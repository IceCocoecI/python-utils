import logging
import matplotlib.pyplot as plt
from IPython.display import display
from pymilvus import MilvusClient
from PIL import Image
from milvus.image_research.image_search import FeatureExtractor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        client = MilvusClient(uri="http://localhost:19530")
        extractor = FeatureExtractor("resnet34")

        query_image = "/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/milvus/image_research/images/test/Afghan_hound/n02088094_4261.JPEG"

        results = client.search(
            "image_embeddings",
            data=[extractor(query_image)],
            output_fields=["filename"],
            search_params={"metric_type": "COSINE"}
        )

        images = []
        for result in results:
            for hit in result[:10]:
                try:
                    filename = hit["entity"]["filename"]
                    img = Image.open(filename)
                    img = img.resize((150, 150))
                    images.append(img)
                except Exception as e:
                    logging.error(f"Error opening or resizing image {filename}: {e}")

        width = 150 * 5
        height = 150 * 2
        concatenated_image = Image.new("RGB", (width, height))

        for idx, img in enumerate(images):
            try:
                x = idx % 5
                y = idx // 5
                concatenated_image.paste(img, (x * 150, y * 150))
            except Exception as e:
                logging.error(f"Error pasting image at index {idx}: {e}")

        # 使用matplotlib展示查询图片
        try:
            query_img = Image.open(query_image).resize((150, 150))
            plt.imshow(query_img)
            plt.title("query")
            plt.show()
        except Exception as e:
            logging.error(f"Error showing query image: {e}")

        # 使用matplotlib展示结果图片
        try:
            plt.imshow(concatenated_image)
            plt.title("results")
            plt.show()
        except Exception as e:
            logging.error(f"Error showing results image: {e}")
    except Exception as e:
        logging.error(f"An overall error occurred: {e}")

if __name__ == "__main__":
    main()