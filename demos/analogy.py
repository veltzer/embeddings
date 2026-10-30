#!/usr/bin/env python3
"""Analogies by vector arithmetic: a is to b as c is to what?

The answer is the word nearest to b - a + c (Mikolov's 3CosAdd), the three
given words excluded. Any of the three may be in another language than the
answer, since every table shares one space:

    demos/analogy.py --lang en man king woman              # queen
    demos/analogy.py --lang en france paris italy          # rome
    demos/analogy.py --lang he איש מלך אישה
    demos/analogy.py --lang en --in he france paris italy

Hebrew answers come out noisier than English ones: a prefixed form of the
right word (כמלכה, ומילאנו) often outranks the word itself.
"""

import argparse

from vectors import load, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True, help="language of a, b and c")
    parser.add_argument("--in", dest="target", help="language of the answer (default: --lang)")
    parser.add_argument("-k", type=int, default=5, help="answers to show")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("c")
    args = parser.parse_args()

    table = load(args.lang)
    target = table if args.target in (None, args.lang) else load(args.target)
    given = [args.a, args.b, args.c]
    vectors = {w: table.vector(w) for w in given}
    missing = [w for w, v in vectors.items() if v is None]
    if missing:
        raise SystemExit(f"not in the table: {', '.join(missing)}")
    a, b, c = (v for v in vectors.values() if v is not None)
    exclude = {normalize(w) for w in given} if target is table else set()
    found = target.nearest(b - a + c, args.k, exclude)
    print(f"{args.a} : {args.b} :: {args.c} : " + ", ".join(f"{w} {s:.2f}" for w, s in found))


if __name__ == "__main__":
    main()
