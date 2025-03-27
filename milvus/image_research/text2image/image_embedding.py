
# 先设置model_config
model_config = {}
model_config['protected_namespaces'] = ()

import os
import logging
import tensorflow as tf
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from towhee import ops, pipe
import numpy as np
import csv

# Set TensorFlow environment variable to suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up the model configuration
model_config = {}
# Bug fix: Set an empty tuple to avoid naming conflicts
model_config['protected_namespaces'] = ()


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_milvus_collection(collection_name, dim):
    try:
        connections.connect(host='127.0.0.1', port='19530')
        logging.info('Connected to Milvus at.')

        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logging.info(f'Dropped existing collection: {collection_name}.')

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, descrition='ids', is_primary=True, auto_id=False),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='embedding vectors', dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'metric_type': 'L2',
            'index_type': "IVF_FLAT",
            'params': {"nlist": 512}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logging.info(f'Created collection: {collection_name} with IVF_FLAT index.')
        return collection

    except Exception as e:
        logging.error(f'Error creating collection: {str(e)}')
        raise

def read_csv(csv_path, encoding='utf-8-sig'):
    base_path = '/home/cz/software/pycharm-2024.2.3/PycharmProjects/python-utils/assets/milvus/image_research/images/'
    try:
        with open(csv_path, 'r', encoding=encoding) as f:
            data = csv.DictReader(f)
            for line in data:
                line['path'] = os.path.join(base_path, line['path'].lstrip('./'))
                yield int(line['id']), line['path']
                logging.info(f'Read line: id={line["id"]}, path={line["path"]}')

    except FileNotFoundError:
        logging.error(f'File not found: {csv_path}')
        raise
    except Exception as e:
        logging.error(f'Error reading CSV: {str(e)}')
        raise

def process_images(csv_path, collection_name='text_image_search', dim=512):
    logging.info("Starting image processing pipeline.")
    create_milvus_collection(collection_name, dim)

    p3 = (
        pipe.input('csv_file')
       .flat_map('csv_file', ('id', 'path'), read_csv)
       .map('path', 'img', ops.image_decode.cv2('rgb'))
       .map('img', 'vec', ops.image_text_embedding.clip(model_name='clip_vit_base_patch16', modality='image', device=0))
       .map('vec', 'vec', lambda x: x / np.linalg.norm(x))
       .map(('id', 'vec'), (), ops.ann_insert.milvus_client(host='127.0.0.1', port='19530', collection_name=collection_name))
       .output()
    )

    try:
        ret = p3(csv_path)
        logging.info(f'Processed images with result: {ret}')
    except Exception as e:
        logging.error(f'Error during pipeline processing: {str(e)}')

    logging.warning("To resolve warnings, consider setting 'model_config[protected_namespaces] = ()'.")

def main():
    csv_path = '../../../assets/milvus/image_research/images/reverse_image_search.csv'
    process_images(csv_path)

if __name__ == "__main__":
    main()
