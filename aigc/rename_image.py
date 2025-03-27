import os

# 指定图片文件夹路径
image_folder = r'/home/cz/workspace/benchmark/Image_restore/origin'

# 获取图片文件列表
image_files = os.listdir(image_folder)

# 设置计数器初始值
count = 1

# 遍历图片文件列表
for image_file in image_files:
    # 获取文件扩展名
    file_ext = os.path.splitext(image_file)[1]

    # 构建新的文件名
    new_file_name = 'restore_{:04d}{}'.format(count, file_ext)

    # 构建旧文件路径和新文件路径
    old_file_path = os.path.join(image_folder, image_file)
    new_file_path = os.path.join(image_folder, new_file_name)

    # 重命名文件
    os.rename(old_file_path, new_file_path)

    # 输出修改后的文件名
    print(f'Renamed: {image_file} -> {new_file_name}')

    # 自增计数器
    count += 1
