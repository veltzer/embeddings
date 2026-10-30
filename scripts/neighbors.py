#!/usr/bin/env python3
"""Print the nearest words to each given word, to eyeball a table.

The query words are looked up in --lang; the neighbors come from --in
(default: the same language), so `--lang he --in en` shows what the
alignment made of a Hebrew word in English.
"""

import numpy as np
from common import final_vec, lang_parser, normalize, read_vec


def main() -> None:
    parser = lang_parser(__doc__.splitlines()[0])
    parser.add_argument("--in", dest="target", help="language to find neighbors in")
    parser.add_argument("-k", type=int, default=10, help="neighbors per word")
    parser.add_argument("words", nargs="+")
    args = parser.parse_args()

    words, vectors = read_vec(final_vec(args.lang))
    target_words, target = (words, vectors) if args.target in (None, args.lang) else read_vec(final_vec(args.target))
    index = {w: i for i, w in enumerate(words)}
    for word in args.words:
        row = index.get(normalize(word))
        if row is None:
            print(f"{word}: not in the table")
            continue
        scores = target @ vectors[row]
        best = [i for i in np.argsort(-scores)[: args.k + 1] if target_words[i] != words[row]][: args.k]
        print(f"{word}: " + ", ".join(f"{target_words[i]} {scores[i]:.2f}" for i in best))


if __name__ == "__main__":
    main()
