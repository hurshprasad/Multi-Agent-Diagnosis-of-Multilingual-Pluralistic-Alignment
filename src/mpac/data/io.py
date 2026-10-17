"""JSONL read/write helpers for ParallelItem, Generation and OracleLabel records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from pathlib import Path


def write_jsonl(path: str | Path, records: Iterable, append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            d = asdict(r) if is_dataclass(r) else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
