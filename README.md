# embeddings

Word embeddings trained on Wikipedia, one table per language, all aligned
into one shared vector space. The tables are published as
[GitHub releases](https://github.com/veltzer/embeddings/releases); nothing
trained is committed here.

The languages are a parameter, not part of the design: each is a table under
`[languages]` in `embeddings.toml`, and every script takes `--lang`. Today
that is Hebrew (`he`), English (`en`) and Russian (`ru`).

## What a release holds

- `<lang>.vec.gz` - the most frequent words of the language (200,000 by
  default) in the word2vec text format: a `<count> <dim>` header line, then
  a word and its numbers per line, most frequent first. Vectors are unit
  length, so a dot product is a cosine.
- `manifest.json` - what made each table: the Wikipedia dump and its parts,
  the corpus size, the fastText parameters, the alignment scores, and the
  tokenizer version.
- `SHA256SUMS`.

## The shared space

Every language is trained on its own Wikipedia, so on its own each table is
in a space of its own: dimension 57 of the Hebrew table and dimension 57 of
the English one mean unrelated things. `scripts/align.py` fixes that by
rotating every language onto the pivot (`pivot` in `embeddings.toml`,
English). The rotation is fitted with orthogonal Procrustes on MUSE's
bilingual dictionary of the 5,000 most common word pairs, and measured on
1,500 pairs it never saw: the release notes give precision@1 and @5, how
often a held-out word's nearest English word (or one of its five nearest)
is a correct translation.

A rotation keeps every distance within a language, so alignment costs
single-language use nothing, and in exchange an English word and its Hebrew
translation land near each other.

## Looking words up

The tables are keyed by the output of `tokenize()` in `scripts/common.py`,
and a lookup must normalize its text the same way: NFKD, every combining
mark dropped (which removes niqqud and Latin accents), Hebrew geresh and
gershayim folded to `'` and `"`, casefolded. Words are runs of letters, with
`'` or `"` allowed between letters; digits are not words. The manifest's
`tokenizer_version` changes whenever that does.

Dropping every combining mark has one cost in Russian: `й` decomposes to
`и` plus a breve, so it is spelled `и` in the tables (`новый` is `новыи`,
and `мой` and `мои` share a row), and `ё` is spelled `е`, as most Russian
text spells it anyway. Look Russian words up through `tokenize()` and this
is invisible; keeping `й` would be a new tokenizer version.

## Demos

`demos/` uses the tables the way anyone would: from the newest release,
downloaded on first use into `download/release/<tag>/`, checked against
`SHA256SUMS` and cached as `.npy`, so only the first run of a language
parses the text (half a minute; later runs load in a fraction of a second).
Run them inside the repository's venv:

```bash
demos/similar.py --lang en chess                   # nearest words
demos/similar.py --lang he --in en שחמט            # ...in the other language
demos/analogy.py --lang en man king woman          # queen
demos/analogy.py --lang en france paris italy      # rome
demos/odd_one_out.py --lang en dog cat he:סוס he:שולחן
demos/blog_search.py --lang en does morality need god
demos/blog_search.py --lang en --posts he does morality need god
```

`blog_search.py` searches the posts of [veltzer.org](https://veltzer.org)
through the site's own `search.json`: each post is the IDF-weighted average
of its words' vectors, and a query in either language ranks the posts of
either. It shows the limit of averaged word vectors as plainly as their
use: a query whose words are close to a post's main words finds it without
sharing a word, while a short abstract question ("can a computer think")
is pulled toward whatever its rarest word is about.

`demos/web/` does nearest words and analogies in the browser, in plain
JavaScript, over the top 50,000 words of each language quantized to int8
(about 16 MB per language). It is published at
[veltzer.org/embeddings](https://veltzer.org/embeddings/): the CI build runs
`make_tables.py`, which writes the page and its tables into `out/web/`, and
deploys that directory to GitHub Pages. The tables come from the release
`[web]` in `embeddings.toml` names, not the newest one, so a new release
reaches the site when that line is changed to it. To try it locally:

```bash
demos/web/make_tables.py                          # writes out/web/
python -m http.server -d out/web                  # then open localhost:8000
```

Hebrew results are noisier than English ones, for one reason: Hebrew
attaches prefixes to the word (ו, ה, ב, ל, מ, ש, כ), so a word's nearest
neighbors are often its own prefixed forms, and an analogy's answer often
comes out with a prefix on it. Russian has a milder version of the same:
its inflected forms (six cases, three genders) are separate words.

## Building it

Run from the repository root, inside the repository's venv (`uv sync`):

```bash
scripts/run.py --lang en --lang he --lang ru     # download, extract, train, align
demos/similar.py --local --lang he מוסר           # eyeball the result
demos/similar.py --local --lang he --in en מוסר
scripts/export.py                                # out/release/
scripts/release.py                    # gh release create
```

`run.py` skips a step whose output is already there (`--force` redoes it):

1. `download.py` - the newest finished Wikipedia articles dump, part by
   part, each checked against its published sha1, into `download/`. A
   language may cap it with `max_download_bytes`; English does, at
   2.5 GB of its ~26 GB, which in the 20261001 dump is its first 5 parts
   (2.1 GB), and so does Russian.
2. `extract.py` - articles only, no redirects; templates, references, file,
   category and interwiki links dropped; one tokenized paragraph per line in
   `out/<lang>/corpus.txt`.
3. `train.py` - fastText skipgram (`[train]` in `embeddings.toml`), the full
   model in `out/<lang>/model.bin` and its top words in `trained.vec`.
4. `align.py` - `final.vec`, in the shared space.

Everything under `download/` and `out/` is ignored by git.

The CI build (`rsconstruct build`) checks the code - ruff, pyrefly, pytest
and the linters - and builds the browser demo from the pinned release. It
does not run the pipeline.

## Licenses

The code is MIT. The vectors are derived from Wikipedia text (CC BY-SA 4.0),
and the alignment was fitted on MUSE's dictionaries (CC BY-NC 4.0).
