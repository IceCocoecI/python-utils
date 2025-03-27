import cv2
import logging
import time
import os
from deepface import DeepFace
import mediapipe as mp
from PIL import Image, ExifTags
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mp_pose = mp.solutions.pose


def get_exif_rotation(image_path):
    """
    检测图片的 EXIF 方向信息并返回旋转角度（仅支持 90 度和 180 度的情况）
    """
    try:
        img = Image.open(image_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                return 180  # 图片需要旋转 180 度
            elif orientation == 6:
                return -90  # 图片需要旋转 270 度（即逆时针 90 度）
            elif orientation == 8:
                return 90  # 图片需要旋转 90 度
        return 0  # 无需旋转
    except Exception as e:
        logging.warning(f"读取 EXIF 信息失败：{e}")
        return 0


def correct_image_rotation(image, angle):
    """
    按指定角度旋转图像（仅支持 90 度和 180 度）
    """
    if angle == 0:
        return image  # 无需旋转
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        logging.warning("不支持的旋转角度，仅支持 90 度或 180 度旋转")
        return image


def correct_image_rotation_with_yolov8(image_path, output_path):
    """
    校正图像旋转并保存（仅针对 90 度和 180 度的旋转情况）
    """
    try:
        start_time = time.time()

        # 1. 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图片，请检查路径：{image_path}")

        # # 2. 优先使用 EXIF 信息检测旋转角度
        # exif_rotation = get_exif_rotation(image_path)
        # if exif_rotation != 0:
        #     logging.info(f"检测到 EXIF 旋转角度：{exif_rotation}")
        #     corrected_image = correct_image_rotation(image, exif_rotation)
        #     cv2.imwrite(output_path, corrected_image)
        #     logging.info(f"EXIF 校正成功，保存到 {output_path}")
        #     return

        # 3. 如果 EXIF 信息不可用，使用 DeepFace 检测人脸
        # faces = DeepFace.extract_faces(image, detector_backend='yolov8', enforce_detection=False)
        #
        # if len(faces) > 0:
        #     face = faces[0]
        #     left_eye = face['facial_area'].get('left_eye')
        #     right_eye = face['facial_area'].get('right_eye')
        #
        #     if left_eye is None or right_eye is None:
        #         logging.info("imagepath%s 未检测到人脸", image_path)
        #         raise ValueError("未检测到眼睛位置，无法校正图像！")
        #
        #     # 计算眼睛位置的水平差距
        #     if left_eye[1] > right_eye[1]:
        #         # 图像顺时针旋转了 90 度
        #         rotation_angle = -90
        #     elif left_eye[1] < right_eye[1]:
        #         # 图像逆时针旋转了 90 度
        #         rotation_angle = 90
        #     else:
        #         # 如果眼睛水平对齐，可能是 180 度旋转
        #         rotation_angle = 180 if left_eye[0] > right_eye[0] else 0
        #
        #     logging.info(f"基于人脸检测到的旋转角度：{rotation_angle}")
        #     corrected_image = correct_image_rotation(image, rotation_angle)
        #     cv2.imwrite(output_path, corrected_image)
        #     logging.info(f"基于人脸校正成功，保存到 {output_path}")
        #     return

        # 4. 如果 DeepFace 失败，尝试使用备用方案（MediaPipe 提取人体关键点）
        body_keypoints = get_body_keypoints(image)
        if body_keypoints:
            left_shoulder = (int(body_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),
                             int(body_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0]))
            right_shoulder = (int(body_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),
                              int(body_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0]))

            if left_shoulder[1] > right_shoulder[1]:
                # 图像顺时针旋转了 90 度
                rotation_angle = -90
            elif left_shoulder[1] < right_shoulder[1]:
                # 图像逆时针旋转了 90 度
                rotation_angle = 90
            else:
                # 如果肩膀水平对齐，可能是 180 度旋转
                rotation_angle = 180 if left_shoulder[0] > right_shoulder[0] else 0

            logging.info(f"基于人体关键点检测到的旋转角度：{rotation_angle}")
            corrected_image = correct_image_rotation(image, rotation_angle)
            cv2.imwrite(output_path, corrected_image)
            logging.info(f"基于人体关键点校正成功，保存到 {output_path}")
            return

        # 5. 如果所有方法都失败，不调整旋转
        logging.warning("未检测到人脸或人体关键点，图像未校正")
        cv2.imwrite(output_path, image)

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"图像处理完成，耗时 {elapsed_time:.4f} 秒")
    except FileNotFoundError as e:
        logging.error(e)
    except ValueError as e:
        logging.error(e)
    except Exception as e:
        logging.exception("发生未知异常，校正图像失败：")


def get_body_keypoints(image):
    """
    使用 MediaPipe 获取人体关键点
    """
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
    return None


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            correct_image_rotation_with_yolov8(input_path, output_path)


if __name__ == "__main__":
    input_dir = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/comfyui-v2/checkpoints/asset/image/rotate_input"  # 输入图片文件夹路径
    output_dir = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/comfyui-v2/checkpoints/asset/image/rotate_output"  # 输出校正后的图片文件夹路径
    process_images(input_dir, output_dir)