#!/usr/bin/env python
"""Which of the given words does not belong with the others?

Each word is scored by its mean cosine to the rest; the lowest is the odd
one out. The words may mix languages, written as lang:word when they are
not in --lang:

    demos/odd_one_out.py --lang en breakfast lunch dinner chess
    demos/odd_one_out.py --lang he כלב חתול סוס שולחן
    demos/odd_one_out.py --lang en dog cat he:סוס he:שולחן
"""

import argparse

import numpy as np
from vectors import Table, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True, help="language of words with no lang: prefix")
    parser.add_argument("words", nargs="+")
    args = parser.parse_args()
    if len(args.words) < 3:
        raise SystemExit("give at least three words")

    tables: dict[str, Table] = {}
    vectors = []
    for item in args.words:
        lang, _, word = item.rpartition(":") if ":" in item else (args.lang, "", item)
        if lang not in tables:
            tables[lang] = load(lang)
        vector = tables[lang].vector(word)
        if vector is None:
            raise SystemExit(f"{item}: not in the table")
        vectors.append(vector)
    matrix = np.array(vectors)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, np.nan)
    fit = np.nanmean(similarity, axis=1)
    for word, score in sorted(zip(args.words, fit, strict=True), key=lambda pair: pair[1]):
        print(f"{score:5.2f}  {word}")
    print(f"odd one out: {args.words[int(np.argmin(fit))]}")


if __name__ == "__main__":
    main()
