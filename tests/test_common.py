"""Tests for the tokenizer and the .vec format in common.py.

The tokenizer is the contract every consumer of the vectors must match, so
the cases pin down exactly what it folds and what it keeps.
"""

from typing import TYPE_CHECKING

import numpy as np

import common

if TYPE_CHECKING:
    from pathlib import Path


class TestTokenize:
    def test_drops_niqqud(self) -> None:
        assert common.tokenize("שָׁלוֹם עוֹלָם") == ["שלום", "עולם"]

    def test_folds_geresh_and_gershayim(self) -> None:
        assert common.tokenize("ג\u05f3ק של צה\u05f4ל") == ["ג'ק", "של", 'צה"ל']

    def test_keeps_apostrophe_inside_words(self) -> None:
        assert common.tokenize("Don\u2019t stop") == ["don't", "stop"]

    def test_drops_accents_and_case(self) -> None:
        assert common.tokenize("Café NAÏVE") == ["cafe", "naive"]

    def test_drops_digits_and_punctuation(self) -> None:
        assert common.tokenize('In 1948, "he" said: go!') == ["in", "he", "said", "go"]

    def test_maqaf_separates_words(self) -> None:
        assert common.tokenize("בית־ספר") == ["בית", "ספר"]


class TestVec:
    def test_round_trip(self, tmp_path: Path) -> None:
        vectors = np.array([[0.5, -0.25], [1.0, 0.0]], dtype=np.float32)
        for name in ("t.vec", "t.vec.gz"):
            common.write_vec(tmp_path / name, ["אב", "b"], vectors)
            words, read = common.read_vec(tmp_path / name)
            assert words == ["אב", "b"]
            np.testing.assert_allclose(read, vectors)

    def test_unit(self) -> None:
        np.testing.assert_allclose(np.linalg.norm(common.unit(np.array([[3.0, 4.0]])), axis=1), [1.0])
