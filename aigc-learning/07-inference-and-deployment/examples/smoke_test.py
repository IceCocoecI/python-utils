"""Run all module 07 examples that should work in the current aigc env."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


COMMANDS = [
    [sys.executable, "kv_cache_and_batching.py", "--model", "llama2-7b", "--batch-size", "4"],
    [sys.executable, "diffusion_acceleration_sim.py", "--latent-size", "32", "--batch-size", "1"],
    [sys.executable, "openai_compatible_toy_server.py", "--self-test"],
    [sys.executable, "fastapi_gateway.py", "--self-test"],
    [sys.executable, "demo_apps.py", "--mode", "self-test"],
]


def main() -> None:
    for command in COMMANDS:
        print(f"\n$ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nAll module 07 smoke tests passed", flush=True)


if __name__ == "__main__":
    main()
