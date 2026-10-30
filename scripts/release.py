#!/usr/bin/env python3
"""Publish out/release/ as a GitHub release of this repository.

The tag is the date of the newest dump in the release (dumps are dated,
and so is everything trained from them); the notes are built from
manifest.json. When that release exists already (a language added from
dumps of the same date), the files that differ from it are uploaded over
it and its notes rewritten; export.py keeps the tables already in it
byte for byte, so only the new ones go up.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from common import RELEASE


def notes(manifest: dict) -> str:
    out = [
        (
            "fastText word vectors trained on Wikipedia, one table per language, all in one shared space: "
            f"every language is rotated onto `{manifest['pivot']}` (see `scripts/align.py`)."
        ),
        "",
        "| language | dump | corpus tokens | words | precision@1 / @5 vs " + manifest["pivot"] + " |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lang, entry in manifest["languages"].items():
        align = entry["align"]
        scores = "pivot" if align.get("pivot") is True else f"{align['precision_at_1']:.2f} / {align['precision_at_5']:.2f}"
        dump = entry["download"]
        out.append(
            f"| {entry['name']} (`{lang}`) | {dump['wiki']} {dump['date']}, {dump['parts']} of {dump['parts_available']} parts "
            f"| {entry['extract']['tokens']:,} | {entry['train']['vocabulary']:,} trained, "
            f"{entry['words']:,} released | {scores} |"
        )
    out += [
        "",
        (
            f"Words are keyed by tokenizer version {manifest['tokenizer_version']} (`scripts/common.py`): "
            "NFKD, combining marks dropped (niqqud, accents), casefolded. Normalize text the same way before a lookup."
        ),
        "",
        (
            "Vectors are unit length, so a dot product is a cosine. Text is derived from Wikipedia (CC BY-SA 4.0); "
            "the alignment was fitted on MUSE's bilingual dictionaries (CC BY-NC 4.0)."
        ),
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    tag = max(entry["download"]["date"] for entry in manifest["languages"].values())
    assets = sorted(RELEASE.iterdir())
    exists = subprocess.run(["gh", "release", "view", tag], check=False, capture_output=True).returncode == 0
    with tempfile.NamedTemporaryFile("w", suffix=".md") as f:
        f.write(notes(manifest))
        f.flush()
        if not exists:
            subprocess.run(
                ["gh", "release", "create", tag, *map(str, assets),
                 "--title", f"Word vectors from the {tag} dumps", "--notes-file", f.name],
                check=True,
            )
            return
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["gh", "release", "download", tag, "--pattern", "SHA256SUMS", "--dir", tmp], check=True)
            published = (Path(tmp) / "SHA256SUMS").read_text().splitlines()
        changed = [p for p in assets if p.name == "SHA256SUMS" or published_line(p) not in published]
        print(f"release {tag} exists: uploading {', '.join(p.name for p in changed)}", flush=True)
        subprocess.run(["gh", "release", "upload", tag, *map(str, changed), "--clobber"], check=True)
        subprocess.run(["gh", "release", "edit", tag, "--notes-file", f.name], check=True)


def published_line(path: Path) -> str:
    """The SHA256SUMS line of path as export.py writes it."""
    for line in (RELEASE / "SHA256SUMS").read_text().splitlines():
        if line.endswith("  " + path.name):
            return line
    return ""


if __name__ == "__main__":
    main()
