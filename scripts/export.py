#!/usr/bin/env python3
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

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)
    manifest = {
        "tokenizer_version": TOKENIZER_VERSION,
        "pivot": config["pivot"],
        "languages": {},
    }
    for lang in languages:
        target = RELEASE / f"{lang}.vec.gz"
        print(f"writing {target.name}", flush=True)
        # gzip the text rather than re-encoding the numbers: final.vec is
        # already in the format the release promises.
        with final_vec(lang).open("rb") as src, gzip.open(target, "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
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
