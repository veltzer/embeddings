"""Tests for the wikitext cleaning in extract.py."""

import extract


def setup_module():
    extract.init_worker(frozenset({"file", "category", "קובץ", "קטגוריה"}))


def test_keeps_prose_links_drops_file_category_and_interwiki():
    text = (
        "The [[jazz|music]] of [[New Orleans]] grew from [[Star Wars: A New Hope|nothing at all]] here.\n"
        "[[File:Band.jpg|thumb|A band playing loudly]]\n[[Category:Music]] [[fr:Jazz]]"
    )
    assert extract.clean(text) == ["the music of new orleans grew from nothing at all here"]


def test_drops_templates_and_references():
    text = "ג'אז{{הערה|מקור}} הוא סוגה מוזיקלית<ref>ספר</ref> שנוצרה בארצות הברית [[קובץ:x.jpg|ממוזער|כיתוב]]"
    assert extract.clean(text) == ["ג'אז הוא סוגה מוזיקלית שנוצרה בארצות הברית"]


def test_drops_short_lines():
    assert extract.clean("== Heading ==\nToo short.") == []
