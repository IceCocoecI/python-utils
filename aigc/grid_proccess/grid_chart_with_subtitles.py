from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import platform
import subprocess
import logging
import unicodedata
from functools import lru_cache
from typing import List, Tuple, Optional

# 可选依赖（使用本地 try/except 确保模块级导入失败不影响代码运行）
try:
    import langdetect

    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from fontTools.ttLib import TTFont

    HAS_FONTTOOLS = True
except ImportError:
    HAS_FONTTOOLS = False

# --- ⚙️ 全局配置和常量 ---
# 支持语言白名单
SUPPORTED_LANGS = {'en', 'zh-Hans', 'zh-Hant', 'ja', 'ko', 'th', 'vi', 'de', 'es', 'pt'}

# 无空格逐字换行语言
NO_SPACE_WRAP_LANGS = {'zh-Hans', 'zh-Hant', 'ja', 'ko', 'th'}

# Unicode 区块用于粗略脚本检测
UNICODE_BLOCKS = {
    'CJK': (0x4E00, 0x9FFF),
    'HANGUL': (0xAC00, 0xD7AF),
    'THAI': (0x0E00, 0x0E7F),
}

# Noto Sans CJK .ttc 字体索引映射（固定值）
TTC_INDEX_MAP = {
    'zh-Hans': 0,  # 简体中文
    'zh-Hant': 1,  # 繁体中文
    'ja': 2,  # 日文
    'ko': 3,  # 韩文
}

# 常见 Noto CJK/通用字体路径 (Linux 优先)
COMMON_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


# --- 核心函数 ---

def find_split_line(pixel_array: np.ndarray, axis: int, expected_pos: float, search_margin: int) -> int:
    """
    在图像的灰度像素数组中寻找最暗（平均值最低）的分割线。
    axis=0 找水平线 (row)，axis=1 找垂直线 (col)。
    """
    if axis == 1:
        # 寻找垂直分割线 (col)
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[1], int(expected_pos + search_margin))
        roi = pixel_array[:, search_start:search_end]
        if roi.size == 0: return int(expected_pos)
        # 求每列的平均值
        line_averages = np.mean(roi, axis=0)
        min_index = int(np.argmin(line_averages))
        return search_start + min_index
    else:
        # 寻找水平分割线 (row)
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[0], int(expected_pos + search_margin))
        roi = pixel_array[search_start:search_end, :]
        if roi.size == 0: return int(expected_pos)
        # 求每行的平均值
        line_averages = np.mean(roi, axis=1)
        min_index = int(np.argmin(line_averages))
        return search_start + min_index


def detect_text_script(text: str) -> str:
    """粗略地根据 Unicode 区块检测文本的主脚本。"""
    script_counts = {'CJK': 0, 'HANGUL': 0, 'THAI': 0, 'Latin': 0}
    for char in text:
        code_point = ord(char)
        matched = False
        for script, (start, end) in UNICODE_BLOCKS.items():
            if start <= code_point <= end:
                script_counts[script] += 1
                matched = True
                break
        if not matched and not char.isspace() and 'L' in unicodedata.category(char):
            script_counts['Latin'] += 1

    main_script = max(script_counts, key=script_counts.get)
    return main_script if script_counts[main_script] > 0 else 'Latin'


@lru_cache(maxsize=1)
def get_system_fonts() -> List[Tuple[str, str]]:
    """获取系统字体列表 (Family Name, File Path)。优先使用 fc-list。"""
    # 优先使用 fc-list (Linux)
    try:
        result = subprocess.run(['fc-list', ':', 'family,file'],
                                capture_output=True, text=True, timeout=3, check=False)
        if result.returncode == 0 and result.stdout:
            fonts = []
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    # 仅取第一个 Family Name 和第一个 File Path
                    family = parts[0].strip().split(',')[0].strip()
                    file_path = parts[1].strip().split(',')[0].strip()
                    if family and file_path:
                        fonts.append((family, file_path))
            logging.debug(f"通过 fc-list 找到 {len(fonts)} 个字体。")
            return fonts
    except Exception as e:
        logging.debug(f"fc-list 失败或不可用: {e}")

    # Windows 回退
    if platform.system() == "Windows":
        windows_font_path = os.environ.get('WINDIR', 'C:\\Windows') + '\\Fonts'
        if os.path.isdir(windows_font_path):
            win_fonts = [(f.split('.')[0], os.path.join(windows_font_path, f))
                         for f in os.listdir(windows_font_path)
                         if f.lower().endswith(('.ttf', '.ttc', '.otf'))]
            logging.debug(f"通过 Windows 目录找到 {len(win_fonts)} 个字体。")
            return win_fonts

    logging.warning("未找到可用的系统字体列表工具 (fc-list 或 Windows 路径)。")
    return []


# 适配 .ttc 字体的字符检查（指定索引）
def check_font_has_text(font_path: Optional[str], text: str, font_index: int = 0) -> bool:
    """检查字体是否包含文本中的所有非空白字符。"""
    if not HAS_FONTTOOLS or not font_path or not os.path.exists(font_path):
        return True  # 无法检查时，假设支持
    try:
        tt = TTFont(font_path, fontNumber=font_index)
        cmap = tt.getBestCmap()
        tt.close()
        for c in text:
            if c.isspace():
                continue
            if ord(c) not in cmap:
                return False
        return True
    except Exception as e:
        logging.debug(f"检查字体 {font_path}（索引 {font_index}）失败: {e}")
        return True  # 检查失败时，假设支持以避免误判


# 适配 .ttc 字体的诊断（指定索引）
def diagnose_font_issues(text: str, font_path: Optional[str], font_index: int = 0):
    """诊断文本中缺失的字符并给出安装建议。"""
    if not HAS_FONTTOOLS or not font_path or not os.path.exists(font_path):
        return
    try:
        tt = TTFont(font_path, fontNumber=font_index)
        cmap = tt.getBestCmap()
        tt.close()
        missing = [c for c in text if (not c.isspace()) and (ord(c) not in cmap)]
        if missing:
            logging.warning(
                f"检测到{len(missing)}个无法渲染的字符（字体：{os.path.basename(font_path)}，索引：{font_index}）。")
            # 给出安装建议
            if any(0x4E00 <= ord(c) <= 0x9FFF for c in missing):
                logging.warning("（中文/日文/韩文）建议安装 Noto CJK: sudo apt install fonts-noto-cjk -y")
            if any(0x0E00 <= ord(c) <= 0x0E7F for c in missing):
                logging.warning("（泰文）建议安装 Noto Thai: sudo apt install fonts-noto-thai -y")
    except Exception:
        pass


@lru_cache(maxsize=128)
def normalize_lang(text: str) -> str:
    """规范化语言代码，首先尝试 langdetect，回退到脚本检测。"""
    lang = None
    if HAS_LANGDETECT and text.strip():
        try:
            lang = langdetect.detect(text)
        except Exception:
            lang = None

    mapping = {
        'en': 'en', 'de': 'de', 'es': 'es', 'pt': 'pt',
        'ja': 'ja', 'ko': 'ko', 'th': 'th', 'vi': 'vi',
        'zh': 'zh-Hans', 'zh-cn': 'zh-Hans', 'zh-sg': 'zh-Hans',
        'zh-tw': 'zh-Hant', 'zh-hk': 'zh-Hant', 'zh-mo': 'zh-Hant',
    }
    if lang in mapping:
        norm = mapping[lang]
        if norm in SUPPORTED_LANGS:
            return norm

    # 回退基于脚本
    script = detect_text_script(text)
    if script == 'CJK':
        return 'zh-Hans'
    if script == 'HANGUL':
        return 'ko'
    if script == 'THAI':
        return 'th'
    # 默认英文
    return 'en'


def list_existing_paths(paths: List[str]) -> List[str]:
    """过滤只保留实际存在的文件路径。"""
    existing = [p for p in paths if os.path.exists(p)]
    return existing


def try_load_font(path: str, size: int, index: int = 0) -> Optional[ImageFont.FreeTypeFont]:
    """尝试加载字体文件，失败则返回 None。"""
    try:
        # 加载 .ttc 时指定索引
        font = ImageFont.truetype(path, size, index=index)
        logging.debug(f"成功加载字体：{path}（索引 {index}，大小 {size}）")
        return font
    except Exception as e:
        logging.debug(f"加载字体失败：{path}（索引 {index}）：{e}")
        return None


def search_system_font_by_keywords(keywords: List[str], size: int) -> Tuple[
    Optional[ImageFont.FreeTypeFont], Optional[str]]:
    """通过关键词在系统字体中搜索并加载字体。"""
    fonts = get_system_fonts()
    for family, file_path in fonts:
        hay = (family + " " + os.path.basename(file_path)).lower()
        if any(k in hay for k in keywords):
            # CJK 字体在 fc-list 中可能没有包含索引信息，直接尝试加载 0 索引
            f = try_load_font(file_path, size, index=0)
            if f:
                return f, file_path
    return None, None


def get_font_for_language(font_size: int, lang: str, sample_text: str = "") -> Tuple[
    ImageFont.FreeTypeFont, Optional[str]]:
    """根据语言选择合适的字体和路径。"""
    candidates = []
    font_index = TTC_INDEX_MAP.get(lang, 0)

    # 根据语言类型过滤常用路径列表
    if lang in {'zh-Hans', 'zh-Hant', 'ja', 'ko'}:
        candidates = [p for p in COMMON_FONT_PATHS if 'CJK' in p] + candidates
    elif lang == 'th':
        candidates = [p for p in COMMON_FONT_PATHS if 'Thai' in p] + candidates
    else:  # 拉丁/越南语/西文通用
        candidates = [p for p in COMMON_FONT_PATHS if 'CJK' not in p and 'Thai' not in p] + candidates

    # 尝试路径加载
    for path in list_existing_paths(candidates):
        # 如果是 ttc 且是 CJK 语言，使用对应的索引
        current_index = font_index if path.lower().endswith('.ttc') and lang in TTC_INDEX_MAP else 0
        f = try_load_font(path, font_size, index=current_index)
        if f:
            if not sample_text or check_font_has_text(path, sample_text, font_index=current_index):
                logging.info(f"成功选择字体：{os.path.basename(path)}（语言：{lang}，索引：{current_index}）")
                return f, path

    # 系统关键词搜索
    keywords_map = {
        'zh-Hans': ['noto sans cjk', 'sc', 'simplified', 'noto cjk'],
        'zh-Hant': ['noto sans cjk', 'tc', 'traditional', 'noto cjk'],
        'ja': ['noto sans cjk', 'jp', 'source han sans'],
        'ko': ['noto sans cjk', 'kr', 'source han sans'],
        'th': ['noto sans thai', 'thai'],
        'vi': ['noto sans', 'dejavu sans', 'viet', 'liberation sans'],
        'en': ['noto sans', 'dejavu sans', 'arial', 'liberation sans'],
    }
    f, p = search_system_font_by_keywords(keywords_map.get(lang, ['noto sans']), font_size)
    if f:
        logging.info(f"通过关键词搜索选择字体：{os.path.basename(p)}（语言：{lang}）")
        return f, p

    # 最终兜底
    logging.error(f"未找到 {lang} 合适字体，使用默认字体。")
    return ImageFont.load_default(), None


def wrap_text(text: str, max_width: int, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont,
              language: Optional[str] = None) -> List[str]:
    """根据最大宽度和语言类型进行文本换行。"""
    lines: List[str] = []
    current_line = ""

    # 逐字换行（中文/日文/韩文/泰文）
    if language in NO_SPACE_WRAP_LANGS:
        for char in text:
            char_width = draw.textlength(char, font=font)

            # 使用 try/except 捕获当前行宽度
            try:
                current_width = draw.textlength(current_line, font=font)
            except Exception:
                # 兼容旧版 Pillow 或异常情况，使用 approximate size
                current_width = font.getsize(current_line)[0]

            if current_width + char_width <= max_width or not current_line:
                current_line += char
            else:
                if current_line.strip():
                    lines.append(current_line)
                current_line = char
        if current_line.strip():
            lines.append(current_line.rstrip())
        return lines

    # 按空格换行（西文/越南语）
    tokens = text.split()
    current_width = 0
    for token in tokens:
        token_with_space = token + " "
        token_width = draw.textlength(token_with_space, font=font)

        # 换行条件
        if current_width + token_width <= max_width or not current_line:
            current_line += token_with_space
            current_width += token_width
        else:
            if current_line.strip():
                lines.append(current_line.rstrip())
            current_line = token_with_space
            current_width = token_width

    if current_line.strip():
        lines.append(current_line.rstrip())
    return lines


def add_multilingual_captions_to_nx1_image(
        image_path: str, captions: List[str], output_path: str, margin: int = 30,
        font_size: int = 24, font_color: Tuple[int, ...] = (255, 255, 255, 255),
        padding: int = 10, max_width_ratio: float = 0.9
) -> bool:
    """
    为 N x 1 布局的图片（N 个垂直片段）添加居中文本字幕。
    自动检测语言、选择字体并处理 CJK/非 CJK 换行。
    """
    n = len(captions)
    if n < 1:
        logging.error("错误：至少需要一条字幕。")
        return False

    try:
        img = Image.open(image_path)
    except Exception as e:
        logging.error(f"打开图片时出错: {e}")
        return False

    width, height = img.size

    # --- 1. 自动分割线（水平） ---
    img_gray = img.convert('L')
    pixels = np.array(img_gray)
    h_lines = []

    # ✨ 增强日志 - 显示分割检测初始化信息
    logging.info(f"🔍 分割线检测初始化 | 图片尺寸: {width}x{height} | 目标分割数: {n}段 | 边距阈值: {margin}px")

    for i in range(1, n):
        expected_h = height * i / n
        # 寻找最暗的分割线
        h_line = find_split_line(pixels, axis=0, expected_pos=expected_h, search_margin=margin)
        h_lines.append(h_line)

        # ✨ 增强日志 - 显示每条分割线的详细信息
        deviation = abs(h_line - expected_h)
        status = "✅ 成功" if deviation <= margin else "⚠️ 偏差较大"
        logging.info(
            f"{status} 分割线 #{i} | "
            f"预期位置: {expected_h:.1f}px | "
            f"检测位置: {h_line}px | "
            f"偏差: {deviation:.1f}px | "
            f"搜索范围: [{max(0, expected_h - margin):.1f}, {min(height, expected_h + margin):.1f}]"
        )

    # ✨ 增强日志 - 显示完整分割点序列和各段高度
    y_points = [0] + h_lines + [height]
    segments = [y_points[i + 1] - y_points[i] for i in range(len(y_points) - 1)]
    logging.info(f"📊 最终分割点序列: {y_points}")
    logging.info(f"📏 各段高度分布: {segments} | 平均高度: {height / n:.1f}px")

    img_with_captions = img.copy()
    # 颜色带 alpha 时确保 RGBA
    if img_with_captions.mode != 'RGBA' and len(font_color) == 4:
        img_with_captions = img_with_captions.convert('RGBA')
    draw = ImageDraw.Draw(img_with_captions)

    # --- 2. 逐段添加字幕 ---
    for i in range(n):
        top = y_points[i]
        bottom = y_points[i + 1]
        caption = captions[i]

        language = normalize_lang(caption)
        if language not in SUPPORTED_LANGS:
            logging.warning(f"检测到不在白名单内的语言，按 'en' 处理。片段: {caption[:16]}...")
            language = 'en'

        # 为本条字幕选择字体
        font, font_path = get_font_for_language(font_size, language, sample_text=caption)

        # 字体诊断
        font_index = TTC_INDEX_MAP.get(language, 0)
        diagnose_font_issues(caption, font_path, font_index=font_index)

        # 文本换行
        max_caption_width = int(width * max_width_ratio)
        lines = wrap_text(caption, max_caption_width, draw, font, language=language)

        # 计算文本总高度
        total_text_height = 0
        line_heights = []
        for line in lines:
            try:
                # 使用 textbbox 准确计算行高
                bbox = draw.textbbox((0, 0), line, font=font)
                lh = bbox[3] - bbox[1]
            except Exception:
                # 兜底计算
                lh = font_size + 4
            line_heights.append(lh)
            total_text_height += lh

        if len(lines) > 1:
            total_text_height += padding * (len(lines) - 1)

        # 底部内边距显示：文本的顶部 Y 坐标
        text_y_start = bottom - total_text_height - padding
        current_y = text_y_start

        # 绘制文本行
        for j, line in enumerate(lines):
            line_height = line_heights[j]
            try:
                # 重新计算宽度以居中
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
            except Exception:
                line_width = draw.textlength(line, font=font)  # 至少尝试用 textlength
                if line_width == 0:
                    line_width = int(len(line) * font_size * 0.6)  # 最终兜底

            text_x = (width - line_width) // 2  # 居中

            try:
                draw.text((text_x, current_y), line, font=font, fill=font_color)
            except Exception as e:
                logging.error(f"绘制文本时出错: {e}")

            # 更新下一行的 Y 坐标
            current_y += line_height + (padding if j < len(lines) - 1 else 0)

    # --- 3. 保存结果 ---
    result = img_with_captions
    # JPG 不支持 Alpha 通道，需要转 RGB
    if output_path.lower().endswith(('.jpg', '.jpeg')) and result.mode == 'RGBA':
        # 使用白色背景融合 Alpha
        background = Image.new('RGB', result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3])  # 粘贴时使用 Alpha 通道作为蒙版
        result = background

    try:
        result.save(output_path)
        logging.info(f"字幕添加完成！结果已保存至: {output_path}")
        return True
    except Exception as e:
        logging.error(f"保存图片时出错: {e}")
        return False




if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('../test/split_images.log', mode='w', encoding='utf-8')
        ]
    )

    multilingual_captions = [
        "简体中文",
        # "繁體中文，需要單字符換行。",
        "これは日本語の例です。",
        "이것은 한국어 예시입니다.",
        # "นี่คือตัวอย่างภาษาไทย ที่อาจยาวเกินไป",
        # "Đây là ví dụ tiếng Việt có dấu và cần hỗ trợ tốt từ font chữ.",
        # "English, which will be wrapped by spaces automatically.",
        # "Deutsch ist eine schöne Sprache, die Leerzeichen für die Worttrennung verwendet.",
        # "El español es un idioma muy bonito que usa espacios.",
        # "O português é uma língua que usa espaços para a quebra de linha.",
    ]

    # 请替换为你的实际图片路径
    # 注意：此处使用的路径为原代码中的硬编码路径，需要确保它们在您的运行环境中是正确的。
    input_image = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grid/three/1762324473013-SJQJmMMK-0.jpg"
    output_dir = "/assets/grid/output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_image = os.path.join(output_dir, "0009.png")

    logging.info("\n" + "=" * 50)
    logging.info("开始添加精选语种字幕...")
    logging.info("=" * 50)

    # 清除缓存，确保每次运行都重新加载系统字体
    get_system_fonts.cache_clear()
    normalize_lang.cache_clear()

    success = add_multilingual_captions_to_nx1_image(
        image_path=input_image,
        captions=multilingual_captions,
        output_path=output_image,
        margin=8,
        font_size=20,
        font_color=(255, 255, 255, 255),
        padding=25,
        max_width_ratio=0.9
    )

    if not success:
        logging.error("\n" + "=" * 50)
        logging.error("字幕添加失败！请检查字体安装。")
        logging.error("推荐安装（Linux）：sudo apt install fonts-noto-cjk fonts-noto-cjk-extra fonts-noto -y")
        logging.error("=" * 50)
    else:
        logging.info("\n" + "=" * 50)
        logging.info(f"字幕添加成功！请检查 {output_image}")
        logging.info("如泰语/越南语有缺字，请确认 Noto Sans/Thai 已安装。")
        logging.info("=" * 50)
