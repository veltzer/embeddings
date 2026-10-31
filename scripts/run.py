#!/usr/bin/env python
"""Run the per-language steps for one or more languages, in order.

download, extract, train and align, each skipped when its output is
already there (pass --force to redo them). The pivot is trained before any
other language is aligned to it. Then run export.py and release.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, corpus_file, final_vec, load_config, model_file, stats_file

STEPS = ("download", "extract", "train", "align")


def done(step: str, lang: str) -> bool:
    if step == "download":
        return stats_file(lang, "download").exists()
    if step == "extract":
        return corpus_file(lang).exists()
    if step == "train":
        return model_file(lang).exists()
    return final_vec(lang).exists()


def run(step: str, lang: str, force: bool) -> None:
    if done(step, lang) and not force:
        print(f"== {lang} {step}: done already", flush=True)
        return
    print(f"== {lang} {step}", flush=True)
    script = Path(ROOT, "scripts", f"{step}.py")
    subprocess.run([sys.executable, str(script), "--lang", lang], check=True)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lang", action="append", required=True, choices=sorted(config["languages"]),
        help="a language to build (repeatable)",
    )
    parser.add_argument("--force", action="store_true", help="redo steps whose output exists")
    args = parser.parse_args()
    pivot = config["pivot"]
    # Aligning needs the pivot's trained vectors, so the pivot goes first.
    languages = sorted(set(args.lang), key=lambda lang: lang != pivot)
    if any(lang != pivot for lang in languages) and pivot not in languages and not done("train", pivot):
        sys.exit(f"aligning needs the pivot ({pivot}) trained first: add --lang {pivot}")
    for lang in languages:
        for step in STEPS:
            run(step, lang, args.force)


if __name__ == "__main__":
    main()
