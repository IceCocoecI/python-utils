import os


"""
key失效 待验证
"""

os.environ["OPENAI_API_KEY"] = "XXXXX"


from glob import glob

text_lines = []

for file_path in glob("../../assets/milvus/build_rag_with_milvus/milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()

    text_lines += file_text.split("# ")


from openai import OpenAI

openai_client = OpenAI()

def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )


test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])
