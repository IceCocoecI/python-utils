import os
import cv2
import logging
from deepface import DeepFace

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


"""
使用yolo8-face模型，检测最大人脸
"""
def find_largest_face_yolo8(input_data):
    try:
        # 判断输入是路径还是已经读取的图像对象
        if isinstance(input_data, str):
            logging.info(f"正在读取图片: {input_data}")
            image = cv2.imread(input_data)
            if image is None:
                raise FileNotFoundError(f"无法找到图片文件: {input_data}")
        else:
            image = input_data

        # Use DeepFace to extract faces with yolov8 as detector backend
        face_objs = DeepFace.extract_faces(image, detector_backend='yolov8', enforce_detection=False)

        if not face_objs:
            logging.warning("未检测到人脸")
            return input_data, image

        # Process detected faces
        faces = []
        for face_obj in face_objs:
            x, y, w, h = face_obj["facial_area"]["x"], face_obj["facial_area"]["y"], face_obj["facial_area"]["w"], face_obj["facial_area"]["h"]
            faces.append((x, y, w, h))

        # Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        cropped_face = image[y:y + h, x:x + w]

        return input_data, cropped_face

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"处理图片时发生错误: {e}")



def find_largest_face_yolo8_test(image_path):
    _, cropped_face = find_largest_face_yolo8(image_path)
    if image_path is not None:
        # 获取原图片所在目录
        dir_path = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(dir_path, image_name + 'cropped_face.jpg')

        # 测试 保存裁剪后的人脸
        logging.info("保存裁剪后的最大人脸为 cropped_face.jpg")
        cv2.imwrite(save_path, cropped_face)

        logging.info(f"找到最大人人脸并保存为：{image_path}")
        #logging.info(f"����后的人：{cropped_face.shape}")
    else:
        logging.error("未找到最大人脸")



if __name__ == '__main__':
    image_path = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/face_detect/0001.png"
    find_largest_face_yolo8_test(image_path)