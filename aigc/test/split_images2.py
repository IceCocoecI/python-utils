from PIL import Image
import numpy as np
import os


def find_separator_lines(pixel_data, axis, darkness_threshold=50, line_width=1):
    """
    沿着指定轴查找深色分隔线。

    参数:
    - pixel_data: 图像的NumPy数组。
    - axis: 0表示水平扫描（查找水平线），1表示垂直扫描（查找垂直线）。
    - darkness_threshold: 判断像素是否足够暗以至于成为分隔线一部分的亮度阈值。
    - line_width: 分隔线的宽度（像素）。

    返回:
    - 一个包含分隔线中心坐标的列表。
    """
    if axis == 1:  # 垂直扫描
        lanes = np.mean(pixel_data, axis=(0, 2))
    else:  # 水平扫描
        lanes = np.mean(pixel_data, axis=(1, 2))

    lines = []
    is_on_line = False
    line_start = 0

    for i, brightness in enumerate(lanes):
        if brightness < darkness_threshold and not is_on_line:
            is_on_line = True
            line_start = i
        elif brightness >= darkness_threshold and is_on_line:
            is_on_line = False
            # 为了极致的准确性，我们将线的中心作为其坐标
            line_end = i - 1
            line_center = line_start + (line_end - line_start) // 2
            if len(lines) == 0 or line_center > lines[-1] + line_width * 2:
                lines.append(line_center)

    # 处理图像边缘可能存在线的情况
    if is_on_line:
        line_end = len(lanes) - 1
        line_center = line_start + (line_end - line_start) // 2
        if len(lines) == 0 or line_center > lines[-1] + line_width * 2:
            lines.append(line_center)

    return lines


def split_nine_grid_image(image_path, output_dir='output'):
    """
    将带有1像素深色分隔线的九宫格图片精确分割成九个子图。

    参数:
    - image_path: 输入的九宫格图片路径。
    - output_dir: 保存分割后图片的目录。
    """
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_data = np.array(image)
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
        return

    # 查找垂直和水平分隔线
    vertical_lines = find_separator_lines(pixel_data, axis=1)
    horizontal_lines = find_separator_lines(pixel_data, axis=0)

    # 获取图像尺寸
    width, height = image.size

    # 预期应有两条垂直线和两条水平线
    if len(vertical_lines) != 2 or len(horizontal_lines) != 2:
        print("错误：未能检测到预期的两条垂直线和两条水平线。")
        print(f"检测到的垂直线: {vertical_lines}")
        print(f"检测到的水平线: {horizontal_lines}")
        return

    # 分割线间距检查
    if len(vertical_lines) == 2:
        spacing_ratio = (vertical_lines[1] - vertical_lines[0]) / width
        if not (0.3 < spacing_ratio < 0.7):  # 间距应在合理范围内
            print("警告：垂直分隔线间距不合理，可能检测错误")
            return

    # 定义裁剪的边界
    x_coords = [0] + [x + 1 for x in vertical_lines] + [width]
    y_coords = [0] + [y + 1 for y in horizontal_lines] + [height]

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 循环裁剪并保存九张图片
    image_count = 1
    for i in range(3):
        for j in range(3):
            # 定义裁剪框 (left, upper, right, lower)
            left = x_coords[j]
            upper = y_coords[i]
            right = x_coords[j + 1] - 1
            lower = y_coords[i + 1] - 1

            box = (left, upper, right, lower)

            # 裁剪图片
            sub_image = image.crop(box)

            # 保存子图片
            output_filename = os.path.join(output_dir, f'sub_image_{image_count}.png')
            sub_image.save(output_filename)
            print(f"已保存 {output_filename}")

            image_count += 1


# --- 使用示例 ---
# 假设您的九宫格图片名为 "nine_grid.png"
# 分割后的图片将保存在名为 "output" 的文件夹中
split_nine_grid_image('/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/aigc/1759213836085-daNdWygq-0.jpg')