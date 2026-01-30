from PIL import Image
import numpy as np
import os


def find_split_line(pixel_array, axis, expected_pos, search_margin):
    """
    在指定轴上和预期位置附近寻找像素平均值最低的线。
    (这个辅助函数无需修改)
    """
    if axis == 1:  # 搜索垂直线
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[1], int(expected_pos + search_margin))
        roi = pixel_array[:, search_start:search_end]
        if roi.size == 0: return int(expected_pos)
        line_averages = np.mean(roi, axis=0)
        min_index = np.argmin(line_averages)
        return search_start + min_index
    else:  # 搜索水平线
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[0], int(expected_pos + search_margin))
        roi = pixel_array[search_start:search_end, :]
        if roi.size == 0: return int(expected_pos)
        line_averages = np.mean(roi, axis=1)
        min_index = np.argmin(line_averages)
        return search_start + min_index


def split_image_smart_nxm(image_path, output_dir, grid_rows, grid_cols, margin=30):
    """
    智能检测分割线并切割 N*M 宫格图片。

    :param image_path: 输入图片路径
    :param output_dir: 输出图片文件夹
    :param grid_rows: 宫格的行数 (M)
    :param grid_cols: 宫格的列数 (N)
    :param margin: 在预期分割线位置附近的搜索范围（像素）
    """
    if grid_rows < 1 or grid_cols < 1:
        print("错误：行数和列数参数必须大于或等于1。")
        return

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
        return

    width, height = img.size
    print(f"成功加载图片: {image_path} (尺寸: {width}x{height})")

    img_gray = img.convert('L')
    pixels = np.array(img_gray)

    # --- 智能查找所有分割线 ---
    print(f"正在为 {grid_rows}x{grid_cols} 宫格智能检测分割线...")

    # --- 独立查找垂直线和水平线 ---
    # 根据列数(grid_cols)查找垂直分割线
    v_lines = []
    for i in range(1, grid_cols):
        expected_v = width * i / grid_cols
        v_line = find_split_line(pixels, axis=1, expected_pos=expected_v, search_margin=margin)
        v_lines.append(v_line)

    # 根据行数(grid_rows)查找水平分割线
    h_lines = []
    for i in range(1, grid_rows):
        expected_h = height * i / grid_rows
        h_line = find_split_line(pixels, axis=0, expected_pos=expected_h, search_margin=margin)
        h_lines.append(h_line)

    if v_lines:
        print(f"检测到 {len(v_lines)} 条垂直分割线位置: {v_lines}")
    if h_lines:
        print(f"检测到 {len(h_lines)} 条水平分割线位置: {h_lines}")

    x_points = [0] + v_lines + [width]
    y_points = [0] + h_lines + [height]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_cells = grid_rows * grid_cols
    print(f"开始根据检测到的线条切割图片成 {total_cells} 块 ({grid_rows}行 x {grid_cols}列)...")

    # 使用 grid_rows 和 grid_cols 进行循环
    for j in range(grid_rows):
        for i in range(grid_cols):
            left = x_points[i]
            top = y_points[j]
            right = x_points[i + 1]
            bottom = y_points[j + 1]

            cell = img.crop((left, top, right, bottom))

            output_path = os.path.join(output_dir, f'cell_{j + 1}_{i + 1}.png')
            cell.save(output_path)

    print(f"切割完成！所有 {total_cells} 个图片块已保存至: {output_dir}")


# --- 使用示例 ---
if __name__ == "__main__":

    # 1. 设置宫格的行数和列数
    grid_rows_count = 3  # <-- 设置为2, 代表切割成2行
    grid_cols_count = 1  # <-- 设置为3, 代表切割成3列  (最终会得到 2x3=6 个小图)

    # 2. 输入图片路径
    input_image = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/aigc/n_m_grid/003.jpg"

    # 3. 输出文件夹路径
    output_folder = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/aigc/output"  # <-- 图片保存文件夹

    # 4. 设置搜索范围（像素）
    search_pixel_margin = 30

    # --- 执行切割 ---
    split_image_smart_nxm(
        input_image,
        output_folder,
        grid_rows=grid_rows_count,
        grid_cols=grid_cols_count,
        margin=search_pixel_margin
    )