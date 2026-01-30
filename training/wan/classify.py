import os
import shutil
import cv2

def get_video_resolution(video_path):
    """
    获取视频文件的分辨率。

    :param video_path: 视频文件的路径
    :return: 视频的宽度和高度
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def classify_videos_by_resolution(source_dir, dest_dir):
    """
    根据视频分辨率将视频文件分类到不同的子文件夹中。

    :param source_dir: 源视频文件夹路径
    :param dest_dir: 目标文件夹路径
    """
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录下的所有文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:  # 支持常见视频格式
                video_path = os.path.join(root, file)
                width, height = get_video_resolution(video_path)
                if width and height:
                    resolution_folder = f"{width}x{height}"
                    dest_folder = os.path.join(dest_dir, resolution_folder)
                    # 创建分辨率对应的子文件夹
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    # 拷贝视频文件到对应的子文件夹
                    shutil.copy2(video_path, os.path.join(dest_folder, file))
                    print(f"Copied {file} to {dest_folder}")
                else:
                    print(f"Failed to get resolution for {file}")

if __name__ == "__main__":
    source_directory = "/media/cz/新加卷/new_workspace/Training/wan2.1/Flower Fairy/原视频/归档"
    destination_directory = "/media/cz/新加卷/new_workspace/Training/wan2.1/Flower Fairy/classify"
    classify_videos_by_resolution(source_directory, destination_directory)