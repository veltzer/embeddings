#!/usr/bin/env python3
"""Put a language's vectors in the shared space: final.vec.

Every language is preprocessed the same way (unit length, mean-centered,
unit length again), which makes the pivot's final.vec the shared space
itself. Any other language is then rotated onto the pivot with orthogonal
Procrustes, fitted on MUSE's 0-5000 dictionary of word pairs and measured
on its 5000-6500 one: precision@1 and @5 is how often a held-out word's
nearest pivot word (or one of its five nearest) is a correct translation.

A rotation keeps every distance inside a language unchanged, so aligning
costs a language's own search nothing.
"""

import json
import urllib.request

import numpy as np
from common import (
    final_vec,
    lang_dir,
    lang_parser,
    load_config,
    normalize,
    read_vec,
    stats_file,
    trained_vec,
    unit,
    write_vec,
)


def preprocess(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = unit(vectors)
    mean = vectors.mean(axis=0)
    return unit(vectors - mean), mean


def dictionary(url: str, src_index: dict[str, int], dst_index: dict[str, int]) -> list[tuple[int, int]]:
    """The pairs of url whose words both have vectors, as (src row, dst row)."""
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    pairs = []
    for line in text.splitlines():
        src, _, dst = line.partition("\t")
        s, d = src_index.get(normalize(src.strip())), dst_index.get(normalize(dst.strip()))
        if s is not None and d is not None:
            pairs.append((s, d))
    return pairs


def procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The rotation W minimizing |src @ W - dst|."""
    u, _, vt = np.linalg.svd(src.T @ dst)
    return u @ vt


def precision(aligned: np.ndarray, pivot: np.ndarray, pairs: list[tuple[int, int]], k: int) -> float:
    gold: dict[int, set[int]] = {}
    for s, d in pairs:
        gold.setdefault(s, set()).add(d)
    sources = sorted(gold)
    scores = aligned[sources] @ pivot.T
    top = np.argpartition(-scores, k, axis=1)[:, :k] if k > 1 else scores.argmax(axis=1)[:, None]
    hits = sum(bool(gold[s] & set(row.tolist())) for s, row in zip(sources, top, strict=True))
    return hits / len(sources)


def main() -> None:
    args = lang_parser(__doc__.splitlines()[0]).parse_args()
    config = load_config()
    pivot_lang = config["pivot"]

    words, vectors = read_vec(trained_vec(args.lang))
    vectors, mean = preprocess(vectors)
    if args.lang == pivot_lang:
        write_vec(final_vec(args.lang), words, vectors)
        np.savez(lang_dir(args.lang) / "transform.npz", mean=mean, rotation=np.eye(len(mean)))
        stats_file(args.lang, "align").write_text(json.dumps({"pivot": True}, indent=2) + "\n")
        print(f"{args.lang} is the pivot: preprocessed, not rotated")
        return

    pivot_words, pivot = read_vec(trained_vec(pivot_lang))
    pivot, _ = preprocess(pivot)
    src_index = {w: i for i, w in enumerate(words)}
    dst_index = {w: i for i, w in enumerate(pivot_words)}
    urls = {name: config["align"][f"{name}_url"].format(src=args.lang, dst=pivot_lang) for name in ("train", "test")}
    train = dictionary(urls["train"], src_index, dst_index)
    test = dictionary(urls["test"], src_index, dst_index)
    print(f"{len(train)} training pairs, {len(test)} test pairs", flush=True)

    s, d = (np.array(column) for column in zip(*train, strict=True))
    rotation = procrustes(vectors[s], pivot[d])
    aligned = unit(vectors @ rotation)
    stats = {
        "pivot": pivot_lang,
        "train_pairs": len(train),
        "test_pairs": len(test),
        "precision_at_1_before": round(precision(vectors, pivot, test, 1), 4),
        "precision_at_1": round(precision(aligned, pivot, test, 1), 4),
        "precision_at_5": round(precision(aligned, pivot, test, 5), 4),
    }
    write_vec(final_vec(args.lang), words, aligned)
    np.savez(lang_dir(args.lang) / "transform.npz", mean=mean, rotation=rotation)
    stats_file(args.lang, "align").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
