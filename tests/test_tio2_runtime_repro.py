"""Reproducible TiO2 runtime checks using existing code paths."""

from pathlib import Path
from typing import List

import pytest

from bnn_aenet.datamodule.aenet.prepare_batches import read_list_structures, split_database
from bnn_aenet.datamodule.aenet.read_forces_bin import (
    tf_read_footer,
    tf_read_header,
    tf_read_struc_info,
    tff_read_header,
)
from bnn_aenet.datamodule.aenet.read_input import read_train_in

EXPECTED_TIO2_ASCII_STRUCTURES = 7815
EXPECTED_TIO2_FORCE_STRUCTURES = 7815
EXPECTED_TOTAL_FORCE_VECTORS = 165229
EXPECTED_RUNTIME_ENERGY_STRUCTURES = 7424
EXPECTED_RUNTIME_FORCE_STRUCTURES = 391
EXPECTED_RUNTIME_REMOVED_STRUCTURES = 0


def _load_force_vector_count(train_file: str, sys_species: List[str]) -> int:
    """Count total 3D force vectors from the ascii training file."""
    with open(train_file, "r", encoding="utf-8") as tf:
        n_species, n_struc, species_index, e_atomic, *_ = tf_read_header(tf, sys_species)
        tf_read_footer(tf, n_species, species_index)

        total_force_vectors = 0
        for _ in range(n_struc):
            _, _, _, _, _, forces, _ = tf_read_struc_info(tf, species_index, e_atomic)
            total_force_vectors += len(forces)
    return total_force_vectors


def test_tio2_runtime_counts_from_existing_loader() -> None:
    """Check TiO2 data path and runtime structure counts."""
    tin = read_train_in("data/TiO/train_forces.in")

    train_path = Path(tin.train_file)
    force_path = Path(tin.train_forces_file)
    if not train_path.exists() or not force_path.exists():
        pytest.skip("TiO2 database files are not available on this machine.")

    print("INPUT PATHS")
    print(f"  train_file={tin.train_file}")
    print(f"  train_forces_file={tin.train_forces_file}")

    with open(tin.train_file, "r", encoding="utf-8") as tf:
        _, n_ascii, _, _, *_ = tf_read_header(tf, tin.sys_species)
    with open(tin.train_forces_file, "rb") as tff:
        n_force_struct = tff_read_header(tff)

    total_force_vectors = _load_force_vector_count(tin.train_file, tin.sys_species)
    list_energy, list_forces, list_removed, _, _ = read_list_structures(tin)

    print("HEADER COUNTS")
    print(f"  ascii_structures={n_ascii}")
    print(f"  force_structures={n_force_struct}")
    print("FORCE VECTORS")
    print(f"  total_force_vectors={total_force_vectors}")
    print("RUNTIME LIST SIZES")
    print(f"  energy_structures={len(list_energy)}")
    print(f"  force_structures={len(list_forces)}")
    print(f"  removed_structures={len(list_removed)}")

    assert n_ascii == EXPECTED_TIO2_ASCII_STRUCTURES
    assert n_force_struct == EXPECTED_TIO2_FORCE_STRUCTURES
    assert total_force_vectors == EXPECTED_TOTAL_FORCE_VECTORS
    assert len(list_energy) == EXPECTED_RUNTIME_ENERGY_STRUCTURES
    assert len(list_forces) == EXPECTED_RUNTIME_FORCE_STRUCTURES
    assert len(list_removed) == EXPECTED_RUNTIME_REMOVED_STRUCTURES


def test_tio2_random_split_produces_valid_partitions() -> None:
    """Verify random split on runtime sizes produces non-overlapping, complete partitions."""
    for label, size in (
        ("energy", EXPECTED_RUNTIME_ENERGY_STRUCTURES),
        ("forces", EXPECTED_RUNTIME_FORCE_STRUCTURES),
    ):
        train, valid, test = split_database(size, valid_split=0.1, test_split=0.1)
        total = len(train) + len(valid) + len(test)
        print(
            f"SPLIT {label}: train={len(train)} valid={len(valid)} test={len(test)} total={total}"
        )
        assert total == size, f"{label}: expected {size}, got {total}"
        all_idx = sorted(train + valid + test)
        assert all_idx == list(range(size)), f"{label}: indices don't cover full range"
        assert len(set(train) & set(valid)) == 0, f"{label}: train/valid overlap"
        assert len(set(train) & set(test)) == 0, f"{label}: train/test overlap"
        assert len(set(valid) & set(test)) == 0, f"{label}: valid/test overlap"
