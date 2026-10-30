# embeddings

Word embeddings trained on Wikipedia, one table per language, all aligned
into one shared vector space. The tables are published as
[GitHub releases](https://github.com/veltzer/embeddings/releases); nothing
trained is committed here.

The languages are a parameter, not part of the design: each is a table under
`[languages]` in `embeddings.toml`, and every script takes `--lang`. Today
that is Hebrew (`he`) and English (`en`).

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

## Building it

Run from the repository root, inside the repository's venv (`uv sync`):

```bash
scripts/run.py --lang en --lang he    # download, extract, train, align
scripts/neighbors.py --lang he מוסר   # eyeball the result
scripts/neighbors.py --lang he --in en מוסר
scripts/export.py                     # out/release/
scripts/release.py                    # gh release create
```

`run.py` skips a step whose output is already there (`--force` redoes it):

1. `download.py` - the newest finished Wikipedia articles dump, part by
   part, each checked against its published sha1, into `download/`. A
   language may cap it with `max_download_bytes`; English does, at the
   first 2.5 GB of its ~26 GB.
2. `extract.py` - articles only, no redirects; templates, references, file,
   category and interwiki links dropped; one tokenized paragraph per line in
   `out/<lang>/corpus.txt`.
3. `train.py` - fastText skipgram (`[train]` in `embeddings.toml`), the full
   model in `out/<lang>/model.bin` and its top words in `trained.vec`.
4. `align.py` - `final.vec`, in the shared space.

Everything under `download/` and `out/` is ignored by git.

The CI build (`rsconstruct build`) checks the code only - ruff, pyrefly,
pytest and the config linters. It does not run the pipeline.

## Licenses

The code is MIT. The vectors are derived from Wikipedia text (CC BY-SA 4.0),
and the alignment was fitted on MUSE's dictionaries (CC BY-NC 4.0).
