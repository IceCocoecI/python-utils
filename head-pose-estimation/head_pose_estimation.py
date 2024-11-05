import os
import cv2
import logging
import detect_max_face
import numpy as np
#from sixdrepnet import SixDRepNet

import traceback
from sixdrepnet.regressor import SixDRepNet_Detector as SixDRepNet

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


"""
进行头部姿态估计，https://github.com/thohemp/6DRepNet
传入路径或者opencv对象
"""
def head_pose_stimation(input_data):
    try:
        model = SixDRepNet()
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
            if img is None:
                raise FileNotFoundError(f"无法读取用于头部姿态估计的图片: {input_data}")
        elif isinstance(input_data, cv2.UMat) or isinstance(input_data, np.ndarray):
            img = input_data
        else:
            raise ValueError("不支持的输入类型。请传入图片路径或者opencv的image对象。")

        pitch, yaw, roll = model.predict(img)
        logging.info(f"Head Pose Estimation Result: Pitch(抬头为正，低头为负)={pitch}, Yaw(右侧为正，左侧为负)={yaw}, Roll(向左肩倾斜为正，向右为负)={roll}")
        return input_data, yaw
    except Exception as e:
        logging.error(f"头部姿态估计过程中出错: {e}")
        raise

"""
图像镜像翻转
"""

def horizontal_flip_cv2(input_data):
    try:

        # 判断输入是路径还是已经读取的图像对象
        if isinstance(input_data, str):
            # 读取图片
            image = cv2.imread(input_data)
            if image is None:
                raise FileNotFoundError(f"无法找到图片文件: {input_data}")
            logging.info("成功读取图片")
            # 进行水平翻转
            flipped_image = cv2.flip(image, 1)
            # 保存图片，覆盖原文件
            cv2.imwrite(input_data, flipped_image)
            logging.info("水平翻转成功并覆盖原文件")
        else:
            image = input_data
            logging.info("成功读取图片")
            # 进行水平翻转
            flipped_image = cv2.flip(image, 1)
            logging.info("水平翻转成功")

        return flipped_image

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"处理图片时发生错误: {e}")


"""
 1、最大人脸裁切
 2、头部姿态估计
 3、判断是否镜像翻转
"""
def process_head_stimation(input, flag, angle):
    try:
        logging.info(f"开始处理图片的姿态，标志为 {flag}")

        # 1 最大人脸检测
        _, cropped_face = detect_max_face.find_largest_face_yolo8(input)

        # 2 头部姿态估计
        _,yaw = head_pose_stimation(cropped_face)
        logging.info(f"图片的头部姿态估计完成，Yaw 值为 {yaw}")

        # 3 判断是否镜像翻转
        # 用户左图：朝向向左 则镜像翻转，朝向向右或其余朝向，不处理
        # 用户右图：朝向向右 则镜像翻转，朝向向左或其余朝向，不处理
        if flag == "left" and yaw > angle:
            logging.info("图片符合【用户左图，朝向向左】的条件，准备进行水平翻转")
            return horizontal_flip_cv2(input)
        elif flag == "right" and yaw < -angle:
            logging.info("图片符合【用户右图：朝向向右】条件，准备进行水平翻转")
            return horizontal_flip_cv2(input)
        else:
            logging.info("图片不符合翻转条件，不进行处理")
            return input
    except Exception as e:
        traceback.print_exc()
        logging.error(f"处理姿态和图片翻转过程中出错: {e}")


"""
测试用，用于在图片中添加文字
"""
def add_text_to_image_center(image_path, text, output_folder):
    try:
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 计算图像的较长边长度
        max_length = max(height, width)

        # 根据图像较长边长度确定字体大小，这里假设文字高度占较长边的1/10
        font_scale = max_length / 500

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2

        cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 255), thickness)

        # 获取原图片文件名（包含扩展名）
        image_filename = os.path.basename(image_path)
        new_image_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(new_image_path, image)
        print(f"已成功在图片 {image_path} 中间添加文字，并保存为 {new_image_path}")
    except Exception as e:
        print(f"处理图片 {image_path} 时出现错误: {e}")


"""
测试用，测试头部姿态估计head_pose_stimation
"""
def head_pose_estimation_test(image_paths):

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # 常见的图片扩展名
    try:
        for root, dirs, files in os.walk(image_paths):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_path = os.path.join(root, file)
                    logging.info(f"正在处理图片: {image_path}")
                    try:
                        # 1 最大人脸检测
                        _, cropped_face = detect_max_face.find_largest_face_yolo8(image_path)
                        # 2、头部姿态估计
                        _, yaw = head_pose_stimation(cropped_face)

                        # 测试 定义输出目录
                        output_dir = "/checkpoints/asset/image"
                        if yaw > 0:
                            # 朝向文字写入图片
                            add_text_to_image_center(image_path, "Left", output_dir)
                        else:
                            add_text_to_image_center(image_path, "Right", output_dir)
                    except Exception as e:
                        logging.error(f"处理图片 {image_path} 的姿态估计或添加文字时出错: {e}")
    except Exception as e:
        logging.error(f"遍历图片文件夹 {image_paths} 时出错: {e}")


if __name__ == "__main__":

    # 测试头部姿态估计，传入图片文件夹
    # image_paths = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/comfyui-v2/checkpoints/asset/image"
    # head_pose_estimation_test(image_paths)

    # 测试整个流程，传入图片和flag
    image_path = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/comfyui-v2/checkpoints/asset/image/0008.jpg"
    flag = "left"

    #image = cv2.imread(image_path)
    path = process_head_stimation(image_path, flag, 0)
