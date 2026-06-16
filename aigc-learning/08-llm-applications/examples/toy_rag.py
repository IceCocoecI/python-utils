"""Toy RAG pipeline over local markdown files.

This example is intentionally dependency-light: it uses deterministic hashing
embeddings and NumPy cosine search so it runs on CPU without model downloads.

Run:
    conda run -n aigc python aigc-learning/08-llm-applications/examples/toy_rag.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DOC_DIR = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class Document:
    path: Path
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    path: Path
    start_word: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def iter_markdown_files(doc_dir: Path, limit: int | None = None) -> Iterable[Path]:
    files = sorted(p for p in doc_dir.rglob("*.md") if p.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def load_documents(doc_dir: Path, limit: int | None = None) -> list[Document]:
    docs = []
    for path in iter_markdown_files(doc_dir, limit=limit):
        docs.append(Document(path=path, text=path.read_text(encoding="utf-8")))
    if not docs:
        raise FileNotFoundError(f"No markdown files found under {doc_dir}")
    return docs


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = doc.text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            continue
        text = " ".join(window)
        digest = hashlib.sha1(f"{doc.path}:{start}:{text}".encode("utf-8")).hexdigest()[:10]
        chunks.append(Chunk(chunk_id=digest, path=doc.path, start_word=start, text=text))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    if not chunks:
        raise ValueError("No chunks produced")
    return chunks


def hashing_embedding(text: str, dim: int) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def build_index(chunks: list[Chunk], dim: int) -> np.ndarray:
    return np.stack([hashing_embedding(chunk.text, dim) for chunk in chunks])


def search(query: str, chunks: list[Chunk], index: np.ndarray, top_k: int, dim: int) -> list[SearchResult]:
    query_vec = hashing_embedding(query, dim)
    scores = index @ query_vec
    k = min(top_k, len(chunks))
    top_indices = np.argsort(-scores)[:k]
    return [SearchResult(chunk=chunks[i], score=float(scores[i])) for i in top_indices]


def synthesize_answer(query: str, results: list[SearchResult], max_context_chars: int = 900) -> str:
    context_parts = []
    budget = max_context_chars
    for idx, result in enumerate(results, start=1):
        snippet = result.chunk.text.replace("\n", " ")
        snippet = snippet[:budget]
        if not snippet:
            continue
        context_parts.append(f"[{idx}] {result.chunk.path.name}: {snippet}")
        budget -= len(snippet)
        if budget <= 0:
            break
    context = "\n".join(context_parts)
    return (
        f"Question: {query}\n"
        "Toy answer: use the retrieved context below as evidence. "
        "Replace this synthesizer with a real LLM for production RAG.\n\n"
        f"{context}"
    )


def print_results(query: str, results: list[SearchResult]) -> None:
    print(f"Query: {query}")
    for rank, result in enumerate(results, start=1):
        rel_path = result.chunk.path
        preview = result.chunk.text.replace("\n", " ")[:140]
        print(f"{rank}. score={result.score:.3f} file={rel_path} start_word={result.chunk.start_word}")
        print(f"   {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC_DIR)
    parser.add_argument("--query", default="RAG 如何进行切分 检索 评估")
    parser.add_argument("--chunk-size", type=int, default=90)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> tuple[list[Chunk], list[SearchResult], str]:
    docs = load_documents(args.doc_dir, limit=args.limit_files)
    chunks = build_chunks(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    index = build_index(chunks, dim=args.dim)
    results = search(args.query, chunks, index, top_k=args.top_k, dim=args.dim)
    answer = synthesize_answer(args.query, results)
    return chunks, results, answer


def main() -> None:
    args = parse_args()
    chunks, results, answer = run_pipeline(args)

    print(f"Documents dir: {args.doc_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dim: {args.dim}")
    print_results(args.query, results)
    print("\n" + answer)

    if args.self_test:
        assert chunks
        assert results
        assert all(math.isfinite(result.score) for result in results)
        assert "Toy answer" in answer
        print("\nSelf-test passed")


if __name__ == "__main__":
    main()

