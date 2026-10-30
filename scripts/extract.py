#!/usr/bin/env python3
"""Turn a language's downloaded Wikipedia dump into a training corpus.

The corpus is one paragraph per line, as tokenize() returns it: normalized
words joined by single spaces. Only articles are read (namespace 0, no
redirects); templates, references, tables' markup and file, category and
interwiki links are dropped, and so are lines of fewer than MIN_TOKENS
words, which are headings, captions and list fragments rather than prose.
"""

import bz2
import json
import multiprocessing
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import mwparserfromhell
from common import DOWNLOAD, corpus_file, lang_parser, load_config, stats_file, tokenize

MIN_TOKENS = 5

# Link prefixes whose link is not prose: the wiki's namespace names (read
# from the dump's siteinfo, so they are right in every language) and
# interwiki language codes. Set in each worker by init_worker().
_link_prefixes: frozenset[str] = frozenset()
# "en", "fr", "zh-yue", "be-tarask": an interwiki link. Real titles with a
# colon ("Star Wars: A New Hope", "Spider-Man: Homecoming") do not match.
_LANGUAGE_CODE = re.compile(r"[a-z]{2,3}(?:-[a-z]{2,8})*")
# Tags whose content is not prose. strip_code() keeps a <ref>'s text and
# glues it to the word before it, so these are removed whole first.
_DROP_TAGS = frozenset({
    "ref", "references", "gallery", "math", "chem", "score", "syntaxhighlight",
    "source", "pre", "code", "timeline", "graph", "imagemap", "templatedata",
})


def local(tag: str) -> str:
    """The tag without its "{namespace-uri}" prefix, whatever export version the dump is."""
    return tag.rpartition("}")[2]


def pages(path: Path, prefixes: set[str]) -> Iterator[str]:
    """The wikitext of every article in one dump part; fills prefixes from its siteinfo."""
    with bz2.open(path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        _, root = next(context)
        ns = redirect = None
        for event, elem in context:
            if event != "end":
                continue
            tag = local(elem.tag)
            if tag == "namespace" and elem.text:
                prefixes.add(elem.text.casefold())
            elif tag == "ns":
                ns = elem.text
            elif tag == "redirect":
                redirect = True
            elif tag == "text":
                if ns == "0" and not redirect and elem.text:
                    yield elem.text
            elif tag == "page":
                ns = redirect = None
                root.clear()


def init_worker(prefixes: frozenset[str]) -> None:
    global _link_prefixes
    _link_prefixes = prefixes


def is_prose_link(title: str) -> bool:
    prefix, colon, _ = title.partition(":")
    if not colon:
        return True
    prefix = prefix.strip().casefold()
    return prefix not in _link_prefixes and not _LANGUAGE_CODE.fullmatch(prefix)


def clean(wikitext: str) -> list[str]:
    try:
        code = mwparserfromhell.parse(wikitext)
    except mwparserfromhell.parser.ParserError:  # one page fewer, not a failed run
        return []
    doomed = [tag for tag in code.filter_tags() if str(tag.tag).strip().casefold() in _DROP_TAGS]
    doomed += [link for link in code.filter_wikilinks(recursive=False) if not is_prose_link(str(link.title))]
    for node in doomed:
        try:
            code.remove(node)
        except ValueError:  # inside a node removed before it
            pass
    text = code.strip_code(normalize=True, collapse=True)
    out = []
    for paragraph in text.split("\n"):
        words = tokenize(paragraph)
        if len(words) >= MIN_TOKENS:
            out.append(" ".join(words))
    return out


def main() -> None:
    args = lang_parser(__doc__.splitlines()[0]).parse_args()
    download = json.loads(stats_file(args.lang, "download").read_text())
    parts = [DOWNLOAD / args.lang / download["date"] / name for name in download["files"]]

    # The siteinfo is at the head of every part, so reading the first part's
    # namespaces before starting the workers costs one short parse.
    # The English names are aliases in every wiki; the dump lists only the
    # canonical names, so the language's own aliases come from the config.
    prefixes: set[str] = {"file", "image", "category", "media"}
    prefixes.update(a.casefold() for a in load_config()["languages"][args.lang].get("namespace_aliases", []))
    next(pages(parts[0], prefixes), None)

    out = corpus_file(args.lang)
    tmp = out.with_name(out.name + ".tmp")
    articles = lines = tokens = 0

    def every_page() -> Iterator[str]:
        for part in parts:
            print(f"reading {part.name}", flush=True)
            yield from pages(part, set())

    workers = max(1, (os.cpu_count() or 2) - 1)
    with (
        multiprocessing.Pool(workers, init_worker, (frozenset(prefixes),)) as pool,
        tmp.open("w", encoding="utf-8") as f,
    ):
        # imap, not imap_unordered: the corpus comes out in dump order, so
        # the same dump always gives the same file.
        for paragraphs in pool.imap(clean, every_page(), chunksize=64):
            if paragraphs:
                articles += 1
                lines += len(paragraphs)
                tokens += sum(p.count(" ") + 1 for p in paragraphs)
                f.write("\n".join(paragraphs) + "\n")
    tmp.replace(out)

    stats = {"articles": articles, "lines": lines, "tokens": tokens}
    stats_file(args.lang, "extract").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"{out}: {articles} articles, {lines} lines, {tokens} tokens")


if __name__ == "__main__":
    main()
