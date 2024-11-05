import logging
import torch
from PIL import Image
import timm
import os
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import MilvusClient


# 设置日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_milvus_client():
    try:
        client = MilvusClient(uri="http://localhost:19530")
        if client.has_collection(collection_name="image_embeddings"):
            client.drop_collection(collection_name="image_embeddings")
        client.create_collection(
            collection_name="image_embeddings",
            vector_field_name="vector",
            dimension=512,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE"
        )
        return client
    except Exception as e:
        logging.error(f"Error setting up Milvus client or creating collection: {e}")
        raise


class FeatureExtractor:
    def __init__(self, modelname):
        try:
            # Load the pre - trained model
            self.model = timm.create_model(
                modelname, pretrained=True, num_classes=0, global_pool="avg"
            )
            self.model.eval()

            # Get the input size required by the model
            self.input_size = self.model.default_cfg["input_size"]

            config = resolve_data_config({}, model=modelname)
            # Get the preprocessing function provided by TIMM for the model
            self.preprocess = create_transform(**config)
        except Exception as e:
            logging.error(f"Error initializing FeatureExtractor: {e}")
            raise

    def __call__(self, imagepath):
        try:
            # Preprocess the input image
            input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
            input_image = self.preprocess(input_image)

            # Convert the image to a PyTorch tensor and add a batch dimension
            input_tensor = input_image.unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = self.model(input_tensor)

            # Extract the feature vector
            feature_vector = output.squeeze().numpy()

            return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
        except Exception as e:
            logging.error(f"Error extracting features from {imagepath}: {e}")
            return None


def insert_image_data(client, extractor, root):
    try:
        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".JPEG"):
                    filepath = dirpath + "/" + filename
                    image_embedding = extractor(filepath)
                    if image_embedding is not None:
                        try:
                            client.insert(
                                "image_embeddings",
                                {"vector": image_embedding, "filename": filepath}
                            )
                            logging.info(f"Successfully inserted data for {filepath}")
                        except Exception as e:
                            logging.error(f"Error inserting data for {filepath}: {e}")
    except Exception as e:
        logging.error(f"Error during data insertion process: {e}")


def process():
    client = setup_milvus_client()
    extractor = FeatureExtractor("resnet34")
    root = "./images/train"
    insert_image_data(client, extractor, root)


if __name__ == "__main__":
    process()