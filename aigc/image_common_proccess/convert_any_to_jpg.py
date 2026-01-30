import os
from PIL import Image

input_dir = r"D:\Personal_folder\图片\格式转换\input"
output_dir = r"D:\Personal_folder\图片\格式转换\output"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    file_path = os.path.join(input_dir, filename)

    # 只处理文件
    if os.path.isfile(file_path):
        # 打开并转换图片
        with Image.open(file_path) as img:
            # 获取文件名和扩展名
            base_name, ext = os.path.splitext(filename)
            # 转换为jpg并保存
            img.convert("RGB").save(os.path.join(output_dir, f"{base_name}.jpg"), "JPEG")

print("图片格式转换完成。")