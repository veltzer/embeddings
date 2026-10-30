#!/usr/bin/env python3
"""Download a language's Wikipedia articles dump.

Takes the newest dump whose articles job is done, in its numbered parts,
in page-id order, up to the language's max_download_bytes if it has one.
Every part is checked against the sha1 the dump publishes; a part already
on disk with the right sha1 is not downloaded again.
"""

import hashlib
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import DOWNLOAD, lang_parser, load_config, stats_file

DUMPS = "https://dumps.wikimedia.org"
# Wikimedia asks every client to identify itself.
USER_AGENT = "embeddings/1.0 (https://github.com/veltzer/embeddings)"
# The part's first page id, which is what orders the parts: the number after
# "pages-articles" is a stream, and one stream can hold several parts.
_FIRST_PAGE = re.compile(r"\.xml-p(\d+)p\d+\.bz2$")


def first_page(name: str) -> int:
    match = _FIRST_PAGE.search(name)
    return int(match.group(1)) if match else -1


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read()


def newest_done(wiki: str) -> tuple[str, dict[str, Any]]:
    index = get(f"{DUMPS}/{wiki}/").decode()
    for date in sorted(set(re.findall(r'href="(\d{8})/"', index)), reverse=True):
        try:
            status = json.loads(get(f"{DUMPS}/{wiki}/{date}/dumpstatus.json"))
        except urllib.error.HTTPError:
            continue
        job = status["jobs"].get("articlesdump", {})
        if job.get("status") == "done":
            return date, job["files"]
    sys.exit(f"no finished articles dump for {wiki}")


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path, expected_sha1: str) -> None:
    if path.exists() and sha1(path) == expected_sha1:
        print(f"have {path.name}")
        return
    print(f"downloading {path.name}", flush=True)
    tmp = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)
    if sha1(tmp) != expected_sha1:
        tmp.unlink()
        sys.exit(f"{path.name}: sha1 mismatch")
    tmp.replace(path)


def main() -> None:
    args = lang_parser(__doc__.splitlines()[0]).parse_args()
    budget = load_config()["languages"][args.lang].get("max_download_bytes")
    wiki = f"{args.lang}wiki"
    date, files = newest_done(wiki)
    parts = sorted((name for name in files if first_page(name) >= 0), key=first_page)
    chosen: list[str] = []
    total = 0
    for name in parts:
        if budget is not None and chosen and total + files[name]["size"] > budget:
            break
        chosen.append(name)
        total += files[name]["size"]
    print(f"{wiki} {date}: {len(chosen)} of {len(parts)} parts, {total / 1e9:.2f} GB")

    target = DOWNLOAD / args.lang / date
    target.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        fetch(DUMPS + files[name]["url"], target / name, files[name]["sha1"])

    stats = {
        "wiki": wiki,
        "date": date,
        "parts": len(chosen),
        "parts_available": len(parts),
        "bytes": total,
        "files": {name: files[name]["sha1"] for name in chosen},
    }
    out = stats_file(args.lang, "download")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
