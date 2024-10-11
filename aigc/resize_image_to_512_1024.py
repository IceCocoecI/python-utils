import cv2
import logging
import os

logging.basicConfig(level=logging.INFO)

def resize_image_opencv(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像：{image_path}")

        height, width = img.shape[:2]
        min_dimension = min(height, width)

        if min_dimension < 512:
            target_size = 512
        elif min_dimension > 1024:
            target_size = 1024
        else:
            logging.info(f"图像 {image_path} 无需调整大小。")
            return image_path

        scale_factor = target_size / min_dimension
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        interpolation_method = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
        img_resized = cv2.resize(img, new_dimensions, interpolation=interpolation_method)

        cv2.imwrite(image_path, img_resized)
        logging.info(f"图像 {image_path} 成功调整大小为 {new_dimensions}。")
        return image_path

    except Exception as e:
        logging.error(f"处理图像 {image_path} 时发生错误：{str(e)}")
        return None

def process_images(image_paths):
    resized_paths = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            logging.error(f"图像文件 {image_path} 不存在。")
        else:
            resized_path = resize_image_opencv(image_path)
            if resized_path is not None:
                resized_paths.append(resized_path)
            else:
                logging.warning(f"图像 {image_path} 处理失败。")
    return resized_paths

if __name__ == "__main__":
    image_paths = [
        r"D:/Personal_folder/test/1.png",
        r"D:/Personal_folder/test/2.jpg",
        # 添加其他图像路径
    ]

    resized_image_paths = process_images(image_paths)
    logging.info(f"已处理的图像路径: {resized_image_paths}")