"""What every step of the pipeline shares: the config, the paths, the
tokenizer and the .vec file format.

The tokenizer is a contract, not an implementation detail: the vectors are
keyed by its output, so anything that looks words up in them (the site's
search, for one) must normalize its text exactly as tokenize() does.
"""

import argparse
import gzip
import re
import tomllib
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "embeddings.toml"
# Wikipedia dumps, kept so a re-run of extract does not download again.
DOWNLOAD = ROOT / "download"
OUT = ROOT / "out"
RELEASE = OUT / "release"

# Bumped whenever tokenize() changes what it returns: vectors built under one
# version are keyed differently from text normalized under another.
TOKENIZER_VERSION = 1

# A word is letters, with an apostrophe or a double quote allowed between
# letters: "don't", and the Hebrew geresh and gershayim of צ'יפס and צה"ל
# once normalize() has folded them to ASCII. Digits are not words.
_WORD = re.compile(r"[^\W\d_]+(?:['\"][^\W\d_]+)*")
_FOLD = str.maketrans({
    "׳": "'",  # Hebrew geresh
    "״": '"',  # Hebrew gershayim
    "’": "'",  # right single quotation mark, the typographic apostrophe
})


def normalize(text: str) -> str:
    """Compatibility-decompose, drop every combining mark and casefold.

    Dropping the marks is what removes Hebrew niqqud and cantillation (and
    Latin accents: "café" is "cafe"), so pointed and unpointed text meet.
    """
    decomposed = unicodedata.normalize("NFKD", text.translate(_FOLD))
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.casefold()


def tokenize(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def load_config() -> dict[str, Any]:
    with CONFIG_FILE.open("rb") as f:
        return tomllib.load(f)


def train_params(config: dict[str, Any], lang: str) -> dict[str, Any]:
    params = dict(config["train"])
    params.update(config["languages"][lang].get("train", {}))
    return params


def lang_parser(description: str) -> argparse.ArgumentParser:
    """A parser with the --lang every step takes, its choices read from the config."""
    parser = argparse.ArgumentParser(description=description)
    languages = sorted(load_config()["languages"])
    parser.add_argument(
        "--lang", required=True, choices=languages,
        help="language code, one of the [languages] tables in embeddings.toml",
    )
    return parser


def lang_dir(lang: str) -> Path:
    return OUT / lang


def corpus_file(lang: str) -> Path:
    return lang_dir(lang) / "corpus.txt"


def model_file(lang: str) -> Path:
    return lang_dir(lang) / "model.bin"


def trained_vec(lang: str) -> Path:
    """The top words of the trained model, in its own space."""
    return lang_dir(lang) / "trained.vec"


def final_vec(lang: str) -> Path:
    """The vectors as released: rotated into the pivot's space unless lang is the pivot."""
    return lang_dir(lang) / "final.vec"


def stats_file(lang: str, step: str) -> Path:
    """What a step measured, for export to put in the release manifest."""
    return lang_dir(lang) / f"{step}.json"


def write_vec(path: Path, words: list[str], vectors: np.ndarray) -> None:
    """The word2vec text format: a "<count> <dim>" header, then a word and its numbers per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    tmp = path.with_name(path.name + ".tmp")
    with opener(tmp, "wt", encoding="utf-8") as f:
        f.write(f"{len(words)} {vectors.shape[1]}\n")
        for word, row in zip(words, vectors, strict=True):
            f.write(word + " " + " ".join(f"{x:.5f}" for x in row) + "\n")
    tmp.replace(path)


def read_vec(path: Path) -> tuple[list[str], np.ndarray]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        count, dim = (int(x) for x in f.readline().split())
        words: list[str] = []
        vectors = np.empty((count, dim), dtype=np.float32)
        for i, line in enumerate(f):
            word, _, numbers = line.rstrip("\n").partition(" ")
            words.append(word)
            vectors[i] = np.array(numbers.split(), dtype=np.float32)
    if len(words) != count:
        raise ValueError(f"{path}: header says {count} words, found {len(words)}")
    return words, vectors


def unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
