"""Count and validate TiO2 structure/force totals through project loaders."""

from pathlib import Path
from typing import List, Tuple

import pytest

from bnn_aenet.datamodule.aenet.prepare_batches import read_list_structures
from bnn_aenet.datamodule.aenet.read_forces_bin import (
    tf_read_footer,
    tf_read_header,
    tf_read_struc_info,
    tff_read_header,
)
from bnn_aenet.datamodule.aenet.read_input import read_train_in


def _count_force_vectors(train_file: str, sys_species: List[str]) -> int:
    with open(train_file, "r", encoding="utf-8") as tf:
        n_species, n_struc, species_index, e_atomic, *_ = tf_read_header(tf, sys_species)
        tf_read_footer(tf, n_species, species_index)

        total_vectors = 0
        for _ in range(n_struc):
            _, _, _, _, _, forces, _ = tf_read_struc_info(tf, species_index, e_atomic)
            total_vectors += len(forces)
    return total_vectors


def _count_split_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _runtime_counts(train_in_file: str) -> Tuple[int, int, int]:
    tin = read_train_in(train_in_file)
    list_energy, list_forces, list_removed, _, _ = read_list_structures(tin)
    return len(list_energy), len(list_forces), len(list_removed)


def test_tio2_structure_force_and_split_counts() -> None:
    """Count structures/forces and check split files match runtime totals."""
    tin = read_train_in("data/TiO/train_forces.in")

    train_path = Path(tin.train_file)
    force_path = Path(tin.train_forces_file)
    if not train_path.exists() or not force_path.exists():
        pytest.skip("TiO2 database files are not available on this machine.")

    with open(tin.train_file, "r", encoding="utf-8") as tf:
        _, n_ascii_structures, _, _, *_ = tf_read_header(tf, tin.sys_species)
    with open(tin.train_forces_file, "rb") as tff:
        n_force_structures = tff_read_header(tff)

    total_force_vectors = _count_force_vectors(tin.train_file, tin.sys_species)
    runtime_energy, runtime_forces, removed = _runtime_counts("data/TiO/train_forces.in")

    energy_train = _count_split_lines(Path("data/TiO/splits/energy/train_indices.txt"))
    energy_valid = _count_split_lines(Path("data/TiO/splits/energy/valid_indices.txt"))
    energy_test = _count_split_lines(Path("data/TiO/splits/energy/test_indices.txt"))
    forces_train = _count_split_lines(Path("data/TiO/splits/forces/train_indices.txt"))
    forces_valid = _count_split_lines(Path("data/TiO/splits/forces/valid_indices.txt"))
    forces_test = _count_split_lines(Path("data/TiO/splits/forces/test_indices.txt"))

    print("HEADER COUNTS")
    print(f"  ascii_structures={n_ascii_structures}")
    print(f"  force_structures={n_force_structures}")
    print(f"  total_force_vectors={total_force_vectors}")
    print("RUNTIME COUNTS")
    print(f"  runtime_energy_structures={runtime_energy}")
    print(f"  runtime_force_structures={runtime_forces}")
    print(f"  removed_structures={removed}")
    print("SPLIT COUNTS")
    print(f"  energy(train,valid,test)=({energy_train},{energy_valid},{energy_test})")
    print(f"  forces(train,valid,test)=({forces_train},{forces_valid},{forces_test})")

    assert n_ascii_structures > 0
    assert n_force_structures > 0
    assert total_force_vectors >= n_force_structures
    assert runtime_energy > 0
    assert runtime_forces > 0
    assert runtime_energy <= n_ascii_structures
    assert runtime_forces <= n_force_structures
    assert energy_train + energy_valid + energy_test == runtime_energy
    assert forces_train + forces_valid + forces_test == runtime_forces
