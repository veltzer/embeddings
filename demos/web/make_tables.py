#!/usr/bin/env python
"""Build the browser demo, page and tables, into out/web/ (the GitHub Pages site).

The tables come from the release [web] in embeddings.toml names, not from
the newest one, so the site is a function of the repository: publishing a
release changes nothing until [web] points at it. Per language:
<lang>.words.json (the most frequent words), <lang>.int8 (each row scaled to
fill -127..127, row after row) and <lang>.scale.f32 (each row's scale, to
undo it). That is about a quarter of float32's size; the ranking a dot
product gives barely moves. site.json tells the page which release and
languages it has.

rsconstruct runs this as an explicit processor, which appends --inputs and
--output-dirs; the inputs are only there for rebuild detection.
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vectors import ROOT, fetch

sys.path.insert(0, str(ROOT / "scripts"))
from common import load_config

HERE = Path(__file__).resolve().parent
PAGE = ["index.html", "demo.js", "demo.css"]


def head(path: Path, words: int) -> tuple[list[str], np.ndarray]:
    """The first words rows of a .vec.gz: the table is sorted most frequent first."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        count, dim = (int(x) for x in f.readline().split())
        found, rows = [], []
        for _, line in zip(range(min(words, count)), f, strict=False):
            word, _, rest = line.partition(" ")
            found.append(word)
            rows.append(np.array(rest.split(), dtype=np.float32))
    vectors = np.stack(rows)
    assert vectors.shape[1] == dim
    return found, vectors


def write_language(out: Path, lang: str, tag: str, words: int) -> None:
    found, vectors = head(fetch(f"{lang}.vec.gz", tag), words)
    scale = np.abs(vectors).max(axis=1) / 127
    quantized = np.round(vectors / scale[:, None]).astype(np.int8)
    (out / f"{lang}.words.json").write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    (out / f"{lang}.int8").write_bytes(quantized.tobytes())
    (out / f"{lang}.scale.f32").write_bytes(scale.astype("<f4").tobytes())
    size = sum(p.stat().st_size for p in out.glob(f"{lang}.*"))
    print(f"{lang}: {len(found)} words x {vectors.shape[1]}, {size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dirs", nargs=1, default=[str(ROOT / "out" / "web")], help="where to write the site")
    parser.add_argument("--inputs", nargs="*", default=[], help="ignored: what rsconstruct tracks")
    args = parser.parse_args()
    config = load_config()
    web = config["web"]
    out = Path(args.output_dirs[0])
    # Rebuilt whole, so a language dropped from [web] leaves nothing behind.
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for name in PAGE:
        shutil.copyfile(HERE / name, out / name)
    for lang in web["languages"]:
        write_language(out, lang, web["release"], web["words"])
    site = {
        "release": web["release"],
        "words": web["words"],
        "languages": [{"code": lang, "name": config["languages"][lang]["name"]} for lang in web["languages"]],
    }
    (out / "site.json").write_text(json.dumps(site, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
