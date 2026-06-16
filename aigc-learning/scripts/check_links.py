"""Check local Markdown links under aigc-learning.

The checker ignores fenced code blocks and external URLs. It is intentionally
standard-library only so it can run in a minimal learning environment.

Run:
    conda run -n aigc python aigc-learning/scripts/check_links.py
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


@dataclass(frozen=True)
class MissingLink:
    source: Path
    target: str


def strip_fenced_code(text: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def normalize_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if not target or target.startswith(EXTERNAL_PREFIXES):
        return None
    target = target.split("#", 1)[0]
    if not target:
        return None
    return target


def check_file(path: Path) -> list[MissingLink]:
    text = strip_fenced_code(path.read_text(encoding="utf-8"))
    missing: list[MissingLink] = []
    for match in LINK_RE.finditer(text):
        target = normalize_target(match.group(1))
        if target is None:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            missing.append(MissingLink(source=path, target=target))
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    markdown_files = sorted(root.rglob("*.md"))
    missing: list[MissingLink] = []
    for path in markdown_files:
        missing.extend(check_file(path))

    if missing:
        print("Missing local Markdown links:")
        for item in missing:
            print(f"  {item.source.relative_to(root)} -> {item.target}")
        return 1

    print(f"Checked {len(markdown_files)} Markdown files under {root}; all local links exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

