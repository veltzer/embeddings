#!/usr/bin/env python
"""Write the trimmed tables the browser demo loads, into out/web/.

Per language: <lang>.words.json (the most frequent --words words),
<lang>.int8 (each row scaled to fill -127..127, row after row) and
<lang>.scale.f32 (each row's scale, to undo it). That is about a quarter
of float32's size; the ranking a dot product gives barely moves.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vectors import ROOT, load

OUT = ROOT / "out" / "web"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", action="append", required=True, help="a language to write (repeatable)")
    parser.add_argument("--words", type=int, default=50000, help="words per language")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in args.lang:
        table = load(lang)
        vectors = table.vectors[: args.words]
        scale = np.abs(vectors).max(axis=1) / 127
        quantized = np.round(vectors / scale[:, None]).astype(np.int8)
        (OUT / f"{lang}.words.json").write_text(json.dumps(table.words[: args.words], ensure_ascii=False), encoding="utf-8")
        (OUT / f"{lang}.int8").write_bytes(quantized.tobytes())
        (OUT / f"{lang}.scale.f32").write_bytes(scale.astype("<f4").tobytes())
        size = sum(p.stat().st_size for p in OUT.glob(f"{lang}.*"))
        print(f"{lang}: {len(vectors)} words x {vectors.shape[1]}, {size / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
