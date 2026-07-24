"""Run every offline course example with its self-test flag."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


COURSE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COURSE_ROOT.parent


@dataclass(frozen=True)
class Result:
    script: Path
    returncode: int
    seconds: float
    output: str


def discover_scripts() -> list[Path]:
    examples = sorted(COURSE_ROOT.glob("[0-9][0-9]-*/examples/*.py"))
    labs = sorted(COURSE_ROOT.glob("labs/*/run_demo.py"))
    return [path for path in examples + labs if path.name != "__init__.py"]


def run_script(
    path: Path,
    timeout: float,
    *,
    warnings_as_errors: bool,
) -> Result:
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable]
    if warnings_as_errors:
        command.extend(["-W", "error"])
    command.extend([str(path), "--self-test"])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return Result(
        script=path,
        returncode=completed.returncode,
        seconds=time.perf_counter() - started,
        output=output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--warnings-as-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scripts = discover_scripts()
    if args.list:
        for script in scripts:
            print(script.relative_to(REPO_ROOT))
        return 0
    if not scripts:
        print("No example scripts found.")
        return 1

    failures: list[Result] = []
    for script in scripts:
        result = run_script(
            script,
            args.timeout,
            warnings_as_errors=args.warnings_as_errors,
        )
        status = "PASS" if result.returncode == 0 else "FAIL"
        relative = script.relative_to(REPO_ROOT)
        print(f"[{status}] {relative} ({result.seconds:.2f}s)")
        if args.verbose or result.returncode != 0:
            for line in result.output.splitlines():
                print(f"       {line}")
        if result.returncode != 0:
            failures.append(result)

    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} self-tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
