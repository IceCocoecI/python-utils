from PIL import Image
import os

# 指定图片文件夹路径
image_folder = r'E:\wyfx\ComfyUIworkspace\肥胖黑人生成\shein\处理3-4'

# 指定目标分辨率
target_width = 960
target_height = 1280

# 获取图片文件列表
image_files = [f for f in os.listdir(image_folder) if
               os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# 遍历图片文件列表
for image_file in image_files:
    # 构建图片文件的完整路径
    image_path = os.path.join(image_folder, image_file)

    try:
        # 打开图片
        image = Image.open(image_path)

        # 获取原始图片的宽度和高度
        width, height = image.size

        # 计算缩放比例
        scale = min(target_width / width, target_height / height)

        # 计算缩放后的宽度和高度
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 缩放图片并保持高画质
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)

        # 创建一个新的空白画布
        new_image = Image.new('RGB', (target_width, target_height), (255, 255, 255))

        # 计算图片在画布上的位置
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2

        # 将缩放后的图片粘贴到画布上
        new_image.paste(image, (x, y))

        # 覆盖源文件
        new_image.save(image_path, format='JPEG', quality=95)

        # 输出处理后的文件名
        print(f'Resized: {image_file}')

    except (IOError, OSError, Image.UnidentifiedImageError):
        # 处理无法识别的图像文件错误
        print(f'Error processing file: {image_file}')
