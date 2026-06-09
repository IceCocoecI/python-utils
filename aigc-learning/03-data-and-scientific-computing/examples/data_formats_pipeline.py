"""数据格式与小型数据管线示例。

运行：python data_formats_pipeline.py

示例覆盖 JSONL、Parquet、safetensors、HDF5、HF datasets 和 tar shard。
全部使用本地合成数据，不下载外部资源。
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from PIL import Image
from safetensors.torch import load_file, save_file

OUT = Path(__file__).resolve().parent / "outputs" / "data_formats"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def build_records() -> list[dict]:
    return [
        {"id": i, "text": f"sample text {i}", "length": 13, "label": i % 2}
        for i in range(8)
    ]


def demo_jsonl(records: list[dict]) -> Path:
    print("== 1. JSONL：逐行读写 ==")
    path = OUT / "corpus.jsonl"
    write_jsonl(path, records)
    loaded = list(read_jsonl(path))
    assert loaded == records
    print(f"records: {len(loaded)}, first: {loaded[0]}")
    return path


def demo_parquet(records: list[dict]) -> Path:
    print("\n== 2. Parquet：列式存储 ==")
    df = pd.DataFrame(records)
    path = OUT / "corpus.parquet"
    df.to_parquet(path, compression="zstd", index=False)

    compact = pd.read_parquet(path, columns=["id", "length"])
    assert compact.shape == (len(records), 2)
    assert compact["length"].sum() == len(records) * 13
    print(f"columns: {list(compact.columns)}, shape: {compact.shape}")
    return path


def demo_safetensors() -> Path:
    print("\n== 3. safetensors：安全保存张量 ==")
    path = OUT / "tiny_model.safetensors"
    state = {
        "linear.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "linear.bias": torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    }
    save_file(state, path)
    loaded = load_file(path)
    assert torch.equal(loaded["linear.weight"], state["linear.weight"])
    print(f"keys: {sorted(loaded.keys())}")
    return path


def demo_hdf5() -> Path:
    print("\n== 4. HDF5：按索引切片读取数组 ==")
    path = OUT / "embeddings.h5"
    rng = np.random.default_rng(42)
    features = rng.standard_normal((32, 16)).astype(np.float32)
    labels = rng.integers(0, 3, size=32, dtype=np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip")
        f.create_dataset("labels", data=labels)
        f.attrs["encoder"] = "synthetic-demo"

    with h5py.File(path, "r") as f:
        batch = f["features"][4:10]
        assert batch.shape == (6, 16)
        print(f"encoder: {f.attrs['encoder']}, slice shape: {batch.shape}")
    return path


def demo_hf_datasets(records: list[dict]) -> None:
    print("\n== 5. HF datasets：本地 Arrow 数据集 ==")
    ds = Dataset.from_list(records)
    ds = ds.map(lambda x: {"tokens": x["text"].split()})
    split = ds.train_test_split(test_size=0.25, seed=42)
    assert len(split["train"]) == 6
    assert len(split["test"]) == 2
    print(f"features: {list(ds.features)}, split sizes: train={len(split['train'])}, test={len(split['test'])}")


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = color
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def demo_tar_shard() -> Path:
    print("\n== 6. tar shard：WebDataset 的基础形态 ==")
    path = OUT / "shard-000000.tar"
    samples = [
        ("000000", (255, 0, 0), {"caption": "red square"}),
        ("000001", (0, 255, 0), {"caption": "green square"}),
    ]

    with tarfile.open(path, "w") as tar:
        for key, color, meta in samples:
            _add_bytes_to_tar(tar, f"{key}.jpg", _image_bytes(color))
            _add_bytes_to_tar(tar, f"{key}.json", json.dumps(meta).encode("utf-8"))

    with tarfile.open(path) as tar:
        names = sorted(tar.getnames())
    assert names == ["000000.jpg", "000000.json", "000001.jpg", "000001.json"]
    print(f"members: {names}")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    demo_jsonl(records)
    demo_parquet(records)
    demo_safetensors()
    demo_hdf5()
    demo_hf_datasets(records)
    demo_tar_shard()
    print(f"\noutputs written to: {OUT}")


if __name__ == "__main__":
    main()
