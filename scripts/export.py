#!/usr/bin/env python
"""Assemble the release in out/release/ from every language that has a final.vec.

Each language becomes <lang>.vec.gz, and manifest.json records what made
it: the dump, the corpus, the training parameters, the alignment scores
and the tokenizer version that keys the words. SHA256SUMS covers both.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sys

from common import (
    RELEASE,
    TOKENIZER_VERSION,
    final_vec,
    load_config,
    stats_file,
    train_params,
)

STEPS = ("download", "extract", "train", "align")


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    config = load_config()
    languages = [lang for lang in sorted(config["languages"]) if final_vec(lang).exists()]
    if not languages:
        sys.exit("no language has an out/<lang>/final.vec yet: run the pipeline first")

    # A table already gzipped from the current final.vec is kept as it is:
    # adding a language to a release must not change the bytes (and so the
    # checksums) of the tables already published in it.
    RELEASE.mkdir(parents=True, exist_ok=True)
    for path in RELEASE.iterdir():
        lang = path.name.removesuffix(".vec.gz")
        if lang not in languages or path.stat().st_mtime < final_vec(lang).stat().st_mtime:
            path.unlink()
    manifest = {
        "tokenizer_version": TOKENIZER_VERSION,
        "pivot": config["pivot"],
        "languages": {},
    }
    for lang in languages:
        target = RELEASE / f"{lang}.vec.gz"
        if target.exists():
            print(f"keeping {target.name}", flush=True)
        else:
            print(f"writing {target.name}", flush=True)
            # gzip the text rather than re-encoding the numbers: final.vec is
            # already in the format the release promises. mtime=0 and no
            # file name in the header, so the same table gzips to the same bytes.
            tmp = target.with_name(target.name + ".tmp")
            with final_vec(lang).open("rb") as src, tmp.open("wb") as raw, \
                    gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            tmp.replace(target)
        with final_vec(lang).open(encoding="utf-8") as f:
            words = int(f.readline().split()[0])
        entry = {
            "name": config["languages"][lang]["name"],
            "words": words,
            "train_params": train_params(config, lang),
        }
        for step in STEPS:
            entry[step] = json.loads(stats_file(lang, step).read_text())
        manifest["languages"][lang] = entry
    (RELEASE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    sums = []
    for path in sorted(RELEASE.iterdir()):
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                digest.update(chunk)
        sums.append(f"{digest.hexdigest()}  {path.name}")
    (RELEASE / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"{RELEASE}: " + ", ".join(languages))


if __name__ == "__main__":
    main()
