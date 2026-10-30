#!/usr/bin/env python3
"""Publish out/release/ as a GitHub release of this repository.

The tag is the date of the newest dump in the release (dumps are dated,
and so is everything trained from them); the notes are built from
manifest.json.
"""

import argparse
import json
import subprocess
import tempfile

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
    assets = sorted(str(p) for p in RELEASE.iterdir())
    with tempfile.NamedTemporaryFile("w", suffix=".md") as f:
        f.write(notes(manifest))
        f.flush()
        subprocess.run(
            ["gh", "release", "create", tag, *assets, "--title", f"Word vectors from the {tag} dumps", "--notes-file", f.name],
            check=True,
        )


if __name__ == "__main__":
    main()
