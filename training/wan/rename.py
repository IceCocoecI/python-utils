import os

def rename_video_files(source_dir):
    """
    重命名指定目录下的视频文件。

    :param source_dir: 包含视频文件的目录路径
    """
    # 定义视频文件扩展名列表
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    # 获取目录下所有视频文件，并按名称排序
    video_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(video_extensions)])

    # 遍历视频文件并进行重命名
    for index, old_file_name in enumerate(video_files, start=1):
        # 生成新的文件名
        new_file_name = f"wan-768-1024-{index:03d}{os.path.splitext(old_file_name)[1]}"
        # 构建旧文件的完整路径
        old_file_path = os.path.join(source_dir, old_file_name)
        # 构建新文件的完整路径
        new_file_path = os.path.join(source_dir, new_file_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)


if __name__ == "__main__":
    # 定义源目录路径
    source_dir = '/media/cz/新加卷/new_workspace/Training/wan2.1/Flower Fairy/2-select-rename'
    try:
        # 调用重命名函数
        rename_video_files(source_dir)
        print("视频文件重命名完成。")
    except Exception as e:
        print(f"重命名过程中出现错误: {e}")