"""Tests for the Procrustes fit and its precision measure in align.py."""

import align
import common
import numpy as np


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(0)
    src = common.unit(rng.normal(size=(200, 16)))
    rotation, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    fitted = align.procrustes(src, src @ rotation)
    np.testing.assert_allclose(fitted, rotation, atol=1e-6)


def test_precision_counts_any_gold_translation():
    pivot = np.eye(3)
    aligned = np.array([[1.0, 0.0, 0.0], [0.0, 0.1, 1.0]])
    # row 0's nearest is pivot 0 (gold); row 1's nearest is pivot 2, gold is 1 or 2
    assert align.precision(aligned, pivot, [(0, 0), (1, 1), (1, 2)], 1) == 1.0
    assert align.precision(aligned, pivot, [(0, 1), (1, 2)], 1) == 0.5
