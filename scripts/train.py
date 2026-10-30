#!/usr/bin/env python3
"""Train fastText word vectors on a language's corpus.

Writes the full model (model.bin, with the subword table that gives any
word a vector) and the top [export] words of it as trained.vec, in the
model's own space, unit length, most frequent first.
"""

import json
import os
import time

import fasttext
import numpy as np
from common import (
    corpus_file,
    lang_parser,
    load_config,
    model_file,
    stats_file,
    train_params,
    trained_vec,
    unit,
    write_vec,
)


def main() -> None:
    args = lang_parser(__doc__.splitlines()[0]).parse_args()
    config = load_config()
    params = train_params(config, args.lang)
    threads = os.cpu_count() or 1
    print(f"training {args.lang}: {params}, {threads} threads", flush=True)
    start = time.monotonic()
    model = fasttext.train_unsupervised(
        input=str(corpus_file(args.lang)),
        model=params["model"],
        dim=params["dim"],
        epoch=params["epoch"],
        minCount=params["min_count"],
        minn=params["minn"],
        maxn=params["maxn"],
        thread=threads,
    )
    seconds = time.monotonic() - start
    model.save_model(str(model_file(args.lang)))

    # get_words() is most frequent first.
    vocabulary = model.get_words()
    words = vocabulary[: config["export"]["words"]]
    vectors = unit(np.array([model.get_word_vector(w) for w in words], dtype=np.float32))
    write_vec(trained_vec(args.lang), words, vectors)

    stats = {"params": params, "vocabulary": len(vocabulary), "seconds": round(seconds)}
    stats_file(args.lang, "train").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"{len(vocabulary)} words in {seconds / 60:.1f} minutes; kept {len(words)}")


if __name__ == "__main__":
    main()
