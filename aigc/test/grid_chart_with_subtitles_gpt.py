from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import platform
import subprocess
import logging
import unicodedata
from functools import lru_cache

# 可选依赖
try:
    import langdetect
    has_langdetect = True
except ImportError:
    has_langdetect = False

try:
    from fontTools.ttLib import TTFont
    has_fonttools = True
except ImportError:
    has_fonttools = False

# 支持语言白名单
SUPPORTED_LANGS = {'en', 'zh-Hans', 'zh-Hant', 'ja', 'ko', 'th', 'vi', 'de', 'es', 'pt'}

# 无空格逐字换行语言（如需越南语逐字，请在此集合中加入 'vi'）
NO_SPACE_WRAP_LANGS = {'zh-Hans', 'zh-Hant', 'ja', 'ko', 'th'}

# Unicode 区块用于粗略脚本检测
UNICODE_BLOCKS = {
    'CJK': (0x4E00, 0x9FFF),
    'HANGUL': (0xAC00, 0xD7AF),
    'THAI': (0x0E00, 0x0E7F),
}

def find_split_line(pixel_array, axis, expected_pos, search_margin):
    if axis == 1:
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[1], int(expected_pos + search_margin))
        roi = pixel_array[:, search_start:search_end]
        if roi.size == 0: return int(expected_pos)
        line_averages = np.mean(roi, axis=0)
        min_index = int(np.argmin(line_averages))
        return search_start + min_index
    else:
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[0], int(expected_pos + search_margin))
        roi = pixel_array[search_start:search_end, :]
        if roi.size == 0: return int(expected_pos)
        line_averages = np.mean(roi, axis=1)
        min_index = int(np.argmin(line_averages))
        return search_start + min_index

def detect_text_script(text):
    script_counts = {'CJK': 0, 'HANGUL': 0, 'THAI': 0, 'Latin': 0}
    for char in text:
        code_point = ord(char)
        matched = False
        for script, (start, end) in UNICODE_BLOCKS.items():
            if start <= code_point <= end:
                script_counts[script] += 1
                matched = True
                break
        if not matched:
            if 'L' in unicodedata.category(char):
                script_counts['Latin'] += 1
    main_script = max(script_counts, key=script_counts.get)
    return main_script if script_counts[main_script] > 0 else 'Latin'

def get_system_fonts():
    # 优先使用 fc-list
    try:
        result = subprocess.run(['fc-list', ':', 'family,file'],
                                capture_output=True, text=True, timeout=2, check=False)
        if result.returncode == 0 and result.stdout:
            fonts = []
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    family = parts[0].strip()
                    file_path = parts[1].strip().split(',')[0]
                    fonts.append((family, file_path))
            return fonts
    except Exception:
        pass

    # Windows 回退
    if platform.system() == "Windows":
        windows_font_path = os.environ.get('WINDIR', 'C:\\Windows') + '\\Fonts'
        if os.path.isdir(windows_font_path):
            return [(f, os.path.join(windows_font_path, f)) for f in os.listdir(windows_font_path)
                    if f.lower().endswith(('.ttf', '.ttc', '.otf'))]
    return []

def check_font_has_text(font_path, text):
    if not has_fonttools or not font_path:
        return True
    try:
        tt = TTFont(font_path)
        cmap = tt.getBestCmap()
        tt.close()
        for c in text:
            if c.isspace():
                continue
            if ord(c) not in cmap:
                return False
        return True
    except Exception:
        return False

def diagnose_font_issues(text, font_path):
    if not has_fonttools or not font_path or not os.path.exists(font_path):
        return
    try:
        tt = TTFont(font_path)
        cmap = tt.getBestCmap()
        tt.close()
        missing = [c for c in text if (not c.isspace()) and (ord(c) not in cmap)]
        if missing:
            logging.warning(f"检测到{len(missing)}个无法渲染的字符。")
            if any(0x4E00 <= ord(c) <= 0x9FFF for c in missing):
                logging.warning("建议安装 Noto CJK: sudo apt install fonts-noto-cjk -y")
            if any(0x0E00 <= ord(c) <= 0x0E7F for c in missing):
                logging.warning("建议安装 Noto Thai: sudo apt install fonts-noto-thai -y")
    except Exception:
        pass

def normalize_lang(text):
    # 首先尝试 langdetect
    lang = None
    if has_langdetect and text.strip():
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

def list_existing_paths(paths):
    return [p for p in paths if os.path.exists(p)]

def try_load_font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return None

def search_system_font_by_keywords(keywords, size):
    fonts = get_system_fonts()
    for family, file_path in fonts:
        hay = (family + " " + os.path.basename(file_path)).lower()
        if any(k in hay for k in keywords):
            f = try_load_font(file_path, size)
            if f:
                return f, file_path
    return None, None


def get_font_for_language(font_size, lang, sample_text=""):
    candidates = []

    # 修改这里：将 truetype 替换为 opentype
    if lang in {'zh-Hans', 'zh-Hant', 'ja', 'ko'}:
        candidates += [
            # 优先使用 opentype 路径
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc",
            # 保留原来的 truetype 路径作为后备
            "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]

    # 其他语言路径保持不变...

    # 泰语
    if lang == 'th':
        candidates += [
            "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.otf",
        ]
    # 拉丁/越南语/西文通用
    candidates += [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    # 尝试路径加载
    for path in list_existing_paths(candidates):
        idx = 0
        # 对 .ttc 如需要可选择索引，这里默认 0 足够
        f = try_load_font(path, font_size, index=idx)
        if f:
            if not sample_text or check_font_has_text(path, sample_text):
                return f, path

    # 系统关键词搜索
    keywords_map = {
        'zh-Hans': ['noto sans cjk', 'cjk', 'sc', 'simplified', 'noto cjk'],
        'zh-Hant': ['noto sans cjk', 'cjk', 'tc', 'traditional', 'noto cjk'],
        'ja':      ['noto sans cjk', 'jp', 'cjk', 'noto cjk', 'source han sans'],
        'ko':      ['noto sans cjk', 'kr', 'cjk', 'noto cjk', 'source han sans'],
        'th':      ['noto sans thai', 'thai'],
        'vi':      ['noto sans', 'dejavu sans', 'viet', 'liberation sans'],
        'en':      ['noto sans', 'dejavu sans', 'arial', 'liberation sans'],
        'de':      ['noto sans', 'dejavu sans', 'arial', 'liberation sans'],
        'es':      ['noto sans', 'dejavu sans', 'arial', 'liberation sans'],
        'pt':      ['noto sans', 'dejavu sans', 'arial', 'liberation sans'],
    }
    f, p = search_system_font_by_keywords(keywords_map.get(lang, ['noto sans']), font_size)
    if f:
        return f, p

    # 最终兜底
    logging.error(f"未找到 {lang} 合适字体，使用默认字体，可能缺字。")
    try:
        return ImageFont.load_default(), None
    except Exception:
        return ImageFont.load_default(), None

def wrap_text(text, max_width, draw, font, language=None):
    lines = []
    current_line = ""
    current_width = 0

    # 逐字换行
    if language in NO_SPACE_WRAP_LANGS:
        for char in text:
            try:
                char_width = draw.textlength(char, font=font)
            except Exception:
                char_width = font.getsize(char)[0]
            if current_width + char_width <= max_width or not current_line:
                current_line += char
                current_width += char_width
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
                current_width = char_width
        if current_line.strip():
            lines.append(current_line.rstrip())
        return lines

    # 按空格换行
    tokens = text.split()
    for token in tokens:
        token_with_space = token + " "
        try:
            token_width = draw.textlength(token_with_space, font=font)
        except Exception:
            token_width = font.getsize(token_with_space)[0]
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

def add_multilingual_captions_to_nx1_image(image_path, captions, output_path, margin=30,
                                           font_size=24, font_color=(255, 255, 255, 255),
                                           padding=10, max_width_ratio=0.9):
    n = len(captions)
    if n < 1:
        logging.error("错误：至少需要一条字幕。")
        return False

    try:
        img = Image.open(image_path)
    except Exception as e:
        logging.error(f"打开图片时出错: {str(e)}")
        return False

    width, height = img.size

    # 自动分割线（水平）
    img_gray = img.convert('L')
    pixels = np.array(img_gray)
    h_lines = []
    for i in range(1, n):
        expected_h = height * i / n
        h_line = find_split_line(pixels, axis=0, expected_pos=expected_h, search_margin=margin)
        h_lines.append(h_line)
    expected_h_lines = [int(height * i / n) for i in range(1, n)]
    if len(h_lines) != n - 1:
        h_lines = expected_h_lines
    y_points = [0] + h_lines + [height]

    img_with_captions = img.copy()
    # 颜色带 alpha 时确保 RGBA
    if img_with_captions.mode != 'RGBA' and len(font_color) == 4:
        img_with_captions = img_with_captions.convert('RGBA')
    draw = ImageDraw.Draw(img_with_captions)

    for i in range(n):
        top = y_points[i]
        bottom = y_points[i + 1]
        caption = captions[i]

        language = normalize_lang(caption)
        if language not in SUPPORTED_LANGS:
            logging.warning(f"检测到不在白名单内的语言，按 en 处理。片段: {caption[:16]}...")
            language = 'en'

        # 为本条字幕选择字体
        font, font_path = get_font_for_language(font_size, language, sample_text=caption)

        # 简单可用性检查
        try:
            _ = draw.textbbox((0, 0), "T", font=font)
        except Exception:
            logging.warning("当前字体不可用，回退默认字体。")
            font = ImageFont.load_default()
            font_path = None

        # 字体覆盖诊断
        if font_path:
            diagnose_font_issues(caption, font_path)

        max_caption_width = int(width * max_width_ratio)
        lines = wrap_text(caption, max_caption_width, draw, font, language=language)

        # 计算文本总高度
        line_heights = []
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                lh = bbox[3] - bbox[1]
            except Exception:
                lh = font_size + 4
            line_heights.append(lh)
        total_text_height = sum(line_heights) + padding * (len(lines) - 1 if len(lines) > 0 else 0)

        # 底部内边距显示
        text_y = bottom - total_text_height - padding
        current_y = text_y

        for j, line in enumerate(lines):
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
            except Exception:
                line_width = int(len(line) * font_size * 0.6)
                line_height = font_size + 4

            text_x = (width - line_width) // 2  # 居中
            try:
                draw.text((text_x, current_y), line, font=font, fill=font_color)
            except Exception as e:
                logging.error(f"绘制文本时出错: {str(e)}")
            current_y += line_height + (padding if j < len(lines) - 1 else 0)

    # 保存结果
    result = img_with_captions
    if output_path.lower().endswith(('.jpg', '.jpeg')) and result.mode == 'RGBA':
        result = result.convert('RGB')

    try:
        result.save(output_path)
        logging.info(f"字幕添加完成！结果已保存至: {output_path}")
        return True
    except Exception as e:
        logging.error(f"保存图片时出错: {str(e)}")
        return False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # 如需更多细节改为 DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('split_images.log', mode='w', encoding='utf-8')
        ]
    )

    multilingual_captions = [
        "简体中文，需要单字符换行。",
        "繁體中文，需要單字符換行。",
        "これは日本語の例です。",
        "이것은 한국어 예시입니다.",
        "นี่คือตัวอย่างภาษาไทย ที่อาจยาวเกินไป",
        "Đây là ví dụ tiếng Việt có dấu và cần hỗ trợ tốt từ font chữ.",
        "English, which will be wrapped by spaces automatically.",
        "Deutsch ist eine schöne Sprache, die Leerzeichen für die Worttrennung verwendet.",
        "El español es un idioma muy bonito que usa espacios.",
        "O português é uma língua que usa espaços para a quebra de linha.",
    ]

    # 请替换为你的实际图片路径
    input_image = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grid/three/002.png"
    output_dir = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/grid/output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_image = os.path.join(output_dir, "multilingual_captions_selected_langs.png")

    logging.info("\n" + "=" * 50)
    logging.info("开始添加精选语种字幕...")
    logging.info("=" * 50)

    success = add_multilingual_captions_to_nx1_image(
        image_path=input_image,
        captions=multilingual_captions,
        output_path=output_image,
        margin=30,
        font_size=32,
        font_color=(255, 255, 255, 255),
        padding=15,
        max_width_ratio=0.9
    )

    if not success:
        logging.error("\n" + "=" * 50)
        logging.error("字幕添加失败！请检查字体安装。")
        logging.error("推荐安装: sudo apt install fonts-noto-cjk fonts-noto-cjk-extra fonts-noto-thai fonts-noto -y")
        logging.error("=" * 50)
    else:
        logging.info("\n" + "=" * 50)
        logging.info(f"字幕添加成功！请检查 {output_image}")
        logging.info("如泰语/越南语有缺字，请确认 Noto Sans/Thai 已安装。")
        logging.info("=" * 50)