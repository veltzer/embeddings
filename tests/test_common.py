"""Tests for the tokenizer and the .vec format in common.py.

The tokenizer is the contract every consumer of the vectors must match, so
the cases pin down exactly what it folds and what it keeps.
"""

import common
import numpy as np


class TestTokenize:
    def test_drops_niqqud(self):
        assert common.tokenize("שָׁלוֹם עוֹלָם") == ["שלום", "עולם"]

    def test_folds_geresh_and_gershayim(self):
        assert common.tokenize("צ׳יפס של צה״ל") == ["צ'יפס", "של", 'צה"ל']

    def test_keeps_apostrophe_inside_words(self):
        assert common.tokenize("Don’t stop") == ["don't", "stop"]

    def test_drops_accents_and_case(self):
        assert common.tokenize("Café NAÏVE") == ["cafe", "naive"]

    def test_drops_digits_and_punctuation(self):
        assert common.tokenize('In 1948, "he" said: go!') == ["in", "he", "said", "go"]

    def test_maqaf_separates_words(self):
        assert common.tokenize("בית־ספר") == ["בית", "ספר"]


class TestVec:
    def test_round_trip(self, tmp_path):
        vectors = np.array([[0.5, -0.25], [1.0, 0.0]], dtype=np.float32)
        for name in ("t.vec", "t.vec.gz"):
            common.write_vec(tmp_path / name, ["אב", "b"], vectors)
            words, read = common.read_vec(tmp_path / name)
            assert words == ["אב", "b"]
            np.testing.assert_allclose(read, vectors)

    def test_unit(self):
        np.testing.assert_allclose(np.linalg.norm(common.unit(np.array([[3.0, 4.0]])), axis=1), [1.0])
