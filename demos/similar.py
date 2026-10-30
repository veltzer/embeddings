#!/usr/bin/env python3
"""The nearest words to each given word, in its own language or another.

    demos/similar.py --lang he מוסר שחמט
    demos/similar.py --lang he --in en מוסר שחמט     # what it is in English
    demos/similar.py --lang en --in he chess
"""

import argparse

from vectors import load, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True, help="language of the given words")
    parser.add_argument("--in", dest="target", help="language to find neighbors in (default: --lang)")
    parser.add_argument("-k", type=int, default=8, help="neighbors per word")
    parser.add_argument("--local", action="store_true", help="read out/<lang>/final.vec, not the release")
    parser.add_argument("words", nargs="+")
    args = parser.parse_args()

    table = load(args.lang, args.local)
    target = table if args.target in (None, args.lang) else load(args.target, args.local)
    for word in args.words:
        vector = table.vector(word)
        if vector is None:
            print(f"{word}: not in the table")
            continue
        found = target.nearest(vector, args.k, exclude={normalize(word)} if target is table else set())
        print(f"{word}: " + ", ".join(f"{w} {s:.2f}" for w, s in found))


if __name__ == "__main__":
    main()
