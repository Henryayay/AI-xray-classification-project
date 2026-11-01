"""Ensemble pairing when the two components were repeated a different number of times."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze  # noqa: E402


def test_equal_counts_pair_in_order():
    left, right = ["a0", "a1", "a2"], ["b0", "b1", "b2"]
    assert analyze.pair_components(left, right) == [("a0", "b0"), ("a1", "b1"), ("a2", "b2")]


def test_single_left_run_is_reused_with_every_right_run():
    pairs = analyze.pair_components(["lora"], ["cnn0", "cnn1", "cnn2", "cnn3", "cnn4"])
    assert pairs == [("lora", f"cnn{i}") for i in range(5)]


def test_single_right_run_is_reused_with_every_left_run():
    pairs = analyze.pair_components(["icl0", "icl1", "icl2", "icl3", "icl4"], ["probe"])
    assert pairs == [(f"icl{i}", "probe") for i in range(5)]


def test_unequal_multi_run_counts_follow_the_shorter_list():
    assert analyze.pair_components(["a0", "a1"], ["b0", "b1", "b2", "b3"]) == [
        ("a0", "b0"), ("a1", "b1")]


def test_empty_side_pairs_nothing():
    assert analyze.pair_components([], ["b0"]) == []
    assert analyze.pair_components(["a0"], []) == []
