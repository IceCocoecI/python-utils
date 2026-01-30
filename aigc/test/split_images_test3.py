from PIL import Image
import numpy as np
import os
import logging

def find_split_line(pixel_array, axis, expected_pos, search_margin):
    """
    Find the line with the lowest average pixel value near the expected position on the specified axis.
    (This helper function does not need modification)
    """
    if axis == 1:  # Search for vertical line
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[1], int(expected_pos + search_margin))
        roi = pixel_array[:, search_start:search_end]
        if roi.size == 0: return int(expected_pos)
        line_averages = np.mean(roi, axis=0)
        min_index = np.argmin(line_averages)
        return search_start + min_index
    else:  # Search for horizontal line
        search_start = max(0, int(expected_pos - search_margin))
        search_end = min(pixel_array.shape[0], int(expected_pos + search_margin))
        roi = pixel_array[search_start:search_end, :]
        if roi.size == 0: return int(expected_pos)
        line_averages = np.mean(roi, axis=1)
        min_index = np.argmin(line_averages)
        return search_start + min_index


def split_image_smart_nxm(image_path, output_dir, grid_rows, grid_cols, margin=30):
    """
    Smartly detect split lines and cut the image into N*M grid.

    :param image_path: Input image path
    :param output_dir: Output directory for images
    :param grid_rows: Number of rows in the grid (M)
    :param grid_cols: Number of columns in the grid (N)
    :param margin: Search range around the expected split line position (in pixels)
    """
    if grid_rows < 1 or grid_cols < 1:
        logging.error("Error: Row and column parameters must be greater than or equal to 1.")
        return

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        logging.error(f"Error: File not found {image_path}")
        return

    width, height = img.size
    logging.info(f"Successfully loaded image: {image_path} (size: {width}x{height})")

    img_gray = img.convert('L')
    pixels = np.array(img_gray)

    # --- Smartly find all split lines ---
    logging.info(f"Detecting split lines for {grid_rows}x{grid_cols} grid...")

    # --- Independently find vertical and horizontal lines ---
    # Find vertical split lines based on the number of columns (grid_cols)
    v_lines = []
    for i in range(1, grid_cols):
        expected_v = width * i / grid_cols
        v_line = find_split_line(pixels, axis=1, expected_pos=expected_v, search_margin=margin)
        v_lines.append(v_line)

    # Find horizontal split lines based on the number of rows (grid_rows)
    h_lines = []
    for i in range(1, grid_rows):
        expected_h = height * i / grid_rows
        h_line = find_split_line(pixels, axis=0, expected_pos=expected_h, search_margin=margin)
        h_lines.append(h_line)

    if v_lines:
        logging.info(f"Detected {len(v_lines)} vertical split lines at positions: {v_lines}")
    if h_lines:
        logging.info(f"Detected {len(h_lines)} horizontal split lines at positions: {h_lines}")

    x_points = [0] + v_lines + [width]
    y_points = [0] + h_lines + [height]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_cells = grid_rows * grid_cols
    logging.info(f"Starting to split the image into {total_cells} pieces ({grid_rows} rows x {grid_cols} columns) based on detected lines...")

    # Use grid_rows and grid_cols for looping
    for j in range(grid_rows):
        for i in range(grid_cols):
            left = x_points[i]
            top = y_points[j]
            right = x_points[i + 1]
            bottom = y_points[j + 1]

            cell = img.crop((left, top, right, bottom))

            output_path = os.path.join(output_dir, f'cell_{j + 1}_{i + 1}.png')
            cell.save(output_path)

    logging.info(f"Splitting completed! All {total_cells} image pieces have been saved to: {output_dir}")


# --- Usage example ---
if __name__ == "__main__":

    """
    我有多宫格图片，但是分割线并非等比例划分，帮我分割成单张图片
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 1. Set the number of rows and columns in the grid
    grid_rows_count = 2  # <-- Set to 2, representing 2 rows
    grid_cols_count = 2  # <-- Set to 3, representing 3 columns (will result in 2x3=6 small images)

    # 2. Input image path
    input_image = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/aigc/n_m_grid/4.jpg"

    # 3. Output directory path
    output_folder = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/aigc/output"  # <-- Image save directory

    # 4. Set search range (in pixels)
    search_pixel_margin = 30

    # --- Execute splitting ---
    split_image_smart_nxm(
        input_image,
        output_folder,
        grid_rows=grid_rows_count,
        grid_cols=grid_cols_count,
        margin=search_pixel_margin
    )
