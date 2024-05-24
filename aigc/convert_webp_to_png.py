from PIL import Image
import os


def convert_webp_to_png(input_path, output_path):
    """
    将 WebP 图像转换为 PNG 图像

    参数:
    input_path (str): 输入的 WebP 图像路径
    output_path (str): 输出的 PNG 图像路径
    """
    try:
        # 打开 WebP 图片
        webp_image = Image.open(input_path)

        # 保存为 PNG 图片
        webp_image.save(output_path, "PNG")
        print(f"转换成功: {output_path}")
    except Exception as e:
        print(f"转换失败: {e}")


def batch_convert_webp_to_png(input_dir, output_dir):
    """
    批量将目录中的 WebP 图像转换为 PNG 图像

    参数:
    input_dir (str): 输入目录路径
    output_dir (str): 输出目录路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".webp"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".webp", ".png"))
            convert_webp_to_png(input_path, output_path)


if __name__ == "__main__":
    # 指定输入和输出目录
    input_dir = r"E:\wyfx\ComfyUIworkspace\肥胖黑人生成\shein\原图"
    output_dir = r"E:\wyfx\ComfyUIworkspace\肥胖黑人生成\shein\原图"

    # 调用批量转换函数
    batch_convert_webp_to_png(input_dir, output_dir)