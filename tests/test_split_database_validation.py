"""Validation tests for split_database (simple random splits)."""

import numpy as np
import pytest

from bnn_aenet.datamodule.aenet.prepare_batches import split_database


def test_split_database_returns_three_partitions():
    train, valid, test = split_database(100, valid_split=0.1, test_split=0.1)
    assert len(train) + len(valid) + len(test) == 100


def test_split_database_no_overlap():
    train, valid, test = split_database(200, valid_split=0.2, test_split=0.2)
    assert len(set(train) & set(valid)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(valid) & set(test)) == 0


def test_split_database_covers_all_indices():
    train, valid, test = split_database(50, valid_split=0.1, test_split=0.1)
    all_idx = sorted(train + valid + test)
    assert all_idx == list(range(50))


def test_split_database_deterministic_with_same_seed():
    a = split_database(100, valid_split=0.1, test_split=0.1, seed=7)
    b = split_database(100, valid_split=0.1, test_split=0.1, seed=7)
    assert a == b


def test_split_database_different_seeds_differ():
    a = split_database(100, valid_split=0.1, test_split=0.1, seed=1)
    b = split_database(100, valid_split=0.1, test_split=0.1, seed=2)
    assert a != b


def test_split_database_empty():
    train, valid, test = split_database(0, valid_split=0.1, test_split=0.1)
    assert train == [] and valid == [] and test == []


def test_split_database_absolute_counts():
    train, valid, test = split_database(100, valid_split=15, test_split=10)
    assert len(valid) == 15
    assert len(test) == 10
    assert len(train) == 75


def test_split_database_invalid_sizes_raises():
    with pytest.raises(ValueError, match="Invalid split sizes"):
        split_database(10, valid_split=0.1, test_split=15)


def test_train_fraction_preserves_test_set():
    """Test set must be identical between 100% and 20% train fractions."""
    full = split_database(7424, valid_split=0.1, test_split=0.1, seed=42, train_fraction=1.0)
    small = split_database(7424, valid_split=0.1, test_split=0.1, seed=42, train_fraction=0.2)
    assert full[2] == small[2], "test sets differ between train_fraction=1.0 and 0.2"


def test_train_fraction_reduces_train_size():
    """20% train_fraction should yield ~20% of the non-test pool."""
    full_train, full_valid, full_test = split_database(
        1000, valid_split=0.1, test_split=0.1, seed=42, train_fraction=1.0
    )
    small_train, small_valid, small_test = split_database(
        1000, valid_split=0.1, test_split=0.1, seed=42, train_fraction=0.2
    )
    full_pool = len(full_train) + len(full_valid)
    small_pool = len(small_train) + len(small_valid)
    assert 0.15 < small_pool / full_pool < 0.25


def test_train_fraction_no_overlap():
    train, valid, test = split_database(
        500, valid_split=0.1, test_split=0.1, seed=42, train_fraction=0.3
    )
    assert len(set(train) & set(valid)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(valid) & set(test)) == 0


def test_train_fraction_subset_of_full():
    """Subsampled train+valid must be a subset of the full train+valid."""
    full_train, full_valid, _ = split_database(
        500, valid_split=0.1, test_split=0.1, seed=42, train_fraction=1.0
    )
    small_train, small_valid, _ = split_database(
        500, valid_split=0.1, test_split=0.1, seed=42, train_fraction=0.5
    )
    full_pool = set(full_train + full_valid)
    small_pool = set(small_train + small_valid)
    assert small_pool.issubset(full_pool)


def test_two_runs_identical_splits_and_shared_test_set():
    """Simulate two independent training runs at 100% and 20%.

    Verifies:
    - Run 1 and run 2 at 100% produce identical train/valid/test indices.
    - Run 1 and run 2 at 20% produce identical train/valid/test indices.
    - The test set is the same across 100% and 20%.
    - The first 5 test indices are printed for manual inspection.
    """
    N = 7424
    kwargs = dict(valid_split=0.1, test_split=0.1, seed=42)

    # --- 100%: two independent runs --------------------------------------
    run1_100 = split_database(N, train_fraction=1.0, **kwargs)
    run2_100 = split_database(N, train_fraction=1.0, **kwargs)

    assert run1_100[0] == run2_100[0], "100% train differs between runs"
    assert run1_100[1] == run2_100[1], "100% valid differs between runs"
    assert run1_100[2] == run2_100[2], "100% test differs between runs"

    # --- 20%: two independent runs ----------------------------------------
    run1_20 = split_database(N, train_fraction=0.2, **kwargs)
    run2_20 = split_database(N, train_fraction=0.2, **kwargs)

    assert run1_20[0] == run2_20[0], "20% train differs between runs"
    assert run1_20[1] == run2_20[1], "20% valid differs between runs"
    assert run1_20[2] == run2_20[2], "20% test differs between runs"

    # --- test set identical across fractions -------------------------------
    assert run1_100[2] == run1_20[2], "test set differs between 100% and 20%"

    # --- print first 5 test indices for manual inspection -----------------
    first5 = run1_100[2][:5]
    print(f"First 5 test indices (shared): {first5}")
    print(f"100%: train={len(run1_100[0])}, valid={len(run1_100[1])}, " f"test={len(run1_100[2])}")
    print(f" 20%: train={len(run1_20[0])}, valid={len(run1_20[1])}, " f"test={len(run1_20[2])}")
