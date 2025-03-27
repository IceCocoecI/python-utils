import os


"""
key失效 待验证
"""

os.environ["OPENAI_API_KEY"] = "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"


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
