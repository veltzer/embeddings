"""What the demos share: getting a language's table and asking it questions.

load() reads the newest GitHub release of this repository, the way any
user of the vectors would: it downloads the assets once, checks them
against SHA256SUMS, and keeps a .npy copy beside them, so only the first
load of a language parses the 160 MB of text (about half a minute). With
local=True it reads the pipeline's own out/<lang>/final.vec instead, to
look at a table before it is released.
"""

import gzip
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# The tokenizer is the contract the tables are keyed by: use the pipeline's
# own rather than a copy that could drift from it.
sys.path.insert(0, str(ROOT / "scripts"))
from common import final_vec, normalize, read_vec, tokenize

REPO = "veltzer/embeddings"
CACHE = ROOT / "download" / "release"

__all__ = ["Table", "load", "normalize", "tokenize"]


@dataclass
class Table:
    lang: str
    words: list[str]
    # unit length rows, so a dot product is a cosine
    vectors: np.ndarray
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.words)}

    def vector(self, word: str) -> np.ndarray | None:
        row = self.index.get(normalize(word))
        return None if row is None else self.vectors[row]

    def nearest(self, query: np.ndarray, k: int = 10, exclude: frozenset[str] | set[str] = frozenset()) -> list[tuple[str, float]]:
        scores = self.vectors @ (query / np.linalg.norm(query))
        found = []
        for i in np.argsort(-scores):
            if self.words[i] not in exclude:
                found.append((self.words[i], float(scores[i])))
                if len(found) == k:
                    break
        return found


def get(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def fetch(name: str, tag: str | None = None) -> Path:
    """Download one asset of a release (once) and verify it: the newest, or the one tagged tag."""
    which = "latest" if tag is None else f"tags/{tag}"
    release = json.loads(get(f"https://api.github.com/repos/{REPO}/releases/{which}"))
    target = CACHE / release["tag_name"]
    target.mkdir(parents=True, exist_ok=True)
    assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
    sums_file = target / "SHA256SUMS"
    # Fetched again when it does not name the file: a release can gain a
    # language after a first download.
    if not sums_file.exists() or f"  {name}\n" not in sums_file.read_text():
        sums_file.write_bytes(get(assets["SHA256SUMS"]))
    sums = {}
    for line in sums_file.read_text().splitlines():
        digest, name_ = line.split()
        sums[name_] = digest
    path = target / name
    if not path.exists():
        print(f"downloading {name} from release {release['tag_name']}", file=sys.stderr, flush=True)
        tmp = path.with_name(name + ".part")
        with urllib.request.urlopen(assets[name]) as response, tmp.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != sums[name]:
            tmp.unlink()
            sys.exit(f"{name}: sha256 mismatch")
        tmp.replace(path)
    return path


def parse(path: Path) -> tuple[list[str], np.ndarray]:
    """A .vec.gz, read in one pass: much faster than read_vec()'s line-by-line parse."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        count, dim = (int(x) for x in f.readline().split())
        words, numbers = [], []
        for line in f:
            word, _, rest = line.partition(" ")
            words.append(word)
            numbers.append(rest)
    vectors = np.fromstring(" ".join(numbers), dtype=np.float32, sep=" ").reshape(count, dim)
    return words, vectors


def load(lang: str, local: bool = False) -> Table:
    if local:
        words, vectors = read_vec(final_vec(lang))
        return Table(lang, words, vectors)
    path = fetch(f"{lang}.vec.gz")
    words_file, vectors_file = path.with_suffix(".words.txt"), path.with_suffix(".npy")
    if not vectors_file.exists():
        print(f"parsing {path.name} (first time only)", file=sys.stderr, flush=True)
        words, vectors = parse(path)
        np.save(vectors_file, vectors)
        words_file.write_text("\n".join(words) + "\n", encoding="utf-8")
    words = words_file.read_text(encoding="utf-8").splitlines()
    return Table(lang, words, np.load(vectors_file))
