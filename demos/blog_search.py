#!/usr/bin/env python3
"""Semantic search over the posts of veltzer.org, from the site's own search index.

A post's vector is the average of its words' vectors, weighted by IDF (how
rare the word is across the posts), so "the" and "של" count for nothing and
the words that make a post what it is count most; its title counts three
times. A query is averaged the same way, and posts rank by cosine.

The query may be in another language than the posts, since every table
shares one space:

    demos/blog_search.py --lang en does morality need god
    demos/blog_search.py --lang he האם המוסר זקוק לאלוהים
    demos/blog_search.py --lang en --posts he does morality need god

The last column says whether the site's current search (every query word as
a substring) would have found the post at all.
"""

import argparse
import json
import math
import urllib.request
from collections import Counter

import numpy as np
from vectors import Table, load, normalize, tokenize

SITE = "https://veltzer.org"
TITLE_WEIGHT = 3


def posts(lang: str) -> list[dict]:
    with urllib.request.urlopen(f"{SITE}/{lang}/search.json") as response:
        return json.loads(response.read())["pages"]


def embed(counts: Counter, table: Table, idf: dict[str, float], default_idf: float) -> np.ndarray | None:
    total = np.zeros(table.vectors.shape[1], dtype=np.float32)
    for word, count in counts.items():
        row = table.index.get(word)
        if row is not None:
            total += count * idf.get(word, default_idf) * table.vectors[row]
    norm = np.linalg.norm(total)
    return None if norm == 0 else total / norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", required=True, help="language of the query")
    parser.add_argument("--posts", help="language of the posts to search (default: --lang)")
    parser.add_argument("-k", type=int, default=8, help="posts to show")
    parser.add_argument("query", nargs="+")
    args = parser.parse_args()
    posts_lang = args.posts or args.lang

    pages = posts(posts_lang)
    table = load(posts_lang)
    query_table = table if args.lang == posts_lang else load(args.lang)

    documents = []
    for page in pages:
        counts = Counter(tokenize(page.get("content") or ""))
        for word in tokenize(page["title"]):
            counts[word] += TITLE_WEIGHT
        documents.append(counts)
    df = Counter(word for counts in documents for word in counts)
    idf = {word: math.log(len(documents) / n) for word, n in df.items()}
    # A query word no post contains is as rare as can be.
    rare = math.log(len(documents))

    vectors = []
    for counts in documents:
        vector = embed(counts, table, idf, rare)
        vectors.append(np.zeros(table.vectors.shape[1], dtype=np.float32) if vector is None else vector)
    matrix = np.array(vectors)

    words = tokenize(" ".join(args.query))
    # IDF is measured on the posts, so it only applies to a query in their language.
    query = embed(Counter(words), query_table, idf if query_table is table else {}, 1.0)
    if query is None:
        raise SystemExit("none of the query words is in the table")
    unknown = [w for w in words if w not in query_table.index]
    if unknown:
        print(f"(not in the table, ignored: {', '.join(unknown)})")

    scores = matrix @ query
    for i in np.argsort(-scores)[: args.k]:
        page = pages[i]
        text = normalize(page["title"] + " " + (page.get("content") or ""))
        literal = all(w in text for w in words)
        print(f"{scores[i]:.3f}  {'yes' if literal else 'no ':3}  {page['title']}")
        print(f"       {SITE}{page['path']}")


if __name__ == "__main__":
    main()
