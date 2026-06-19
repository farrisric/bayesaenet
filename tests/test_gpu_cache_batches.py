"""Regression tests for the ``memory_mode == "gpu"`` batch-caching path in
``GroupedDataset`` (``datamodule/aenet/data_set.py``).

Guards the fix in ``load_batches``: the cached batch must be moved to the device
via ``data = self.batch_data_cpu_to_gpu(data)``. The previous code was
``batch_data_cpu_to_gpu(data)`` -- which both raised ``NameError`` (missing
``self.``) and discarded the moved result (the method returns a new list rather
than mutating in place). These tests are device-agnostic (run on CPU in CI).
"""

import torch

from bnn_aenet.datamodule.aenet.data_set import GroupedDataset


def _bare_dataset(memory_mode, train_forces, train_energy, device="cpu"):
    """A GroupedDataset with only the attributes the cache path touches."""
    ds = GroupedDataset.__new__(GroupedDataset)
    ds.device = torch.device(device)
    ds.memory_mode = memory_mode
    ds.train_forces = train_forces
    ds.train_energy = train_energy
    return ds


def test_batch_data_cpu_to_gpu_moves_energy_data():
    ds = _bare_dataset("gpu", train_forces=False, train_energy=True)

    descrp = [torch.randn(2, 3)]
    energy = torch.randn(2)
    reduce = [torch.randn(2, 2)]
    index = torch.tensor([0, 1])
    n_atom = torch.tensor([4, 4])
    data = [None] * 10 + [descrp, energy, reduce, index, n_atom]

    out = ds.batch_data_cpu_to_gpu(data)

    assert len(out) == 15
    assert torch.equal(out[11], energy)
    assert torch.equal(out[14], n_atom)
    assert isinstance(out[10], list) and torch.equal(out[10][0], descrp[0])
    assert isinstance(out[12], list) and torch.equal(out[12][0], reduce[0])
    assert out[13] is index  # database index is passed through, not moved


def test_load_batches_gpu_uses_moved_return(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tmp_batches").mkdir()

    # Minimal energy-only batch (5 elements -> data[10:15]); only needs to load.
    data_e = [
        [torch.randn(2, 3)],
        torch.randn(2),
        [torch.randn(2, 2)],
        torch.tensor([0, 1]),
        torch.tensor([4, 4]),
    ]
    e_name = "./tmp_batches/test_E_0"
    torch.save(data_e, e_name)
    torch.save(
        {
            "E_batch_names": [e_name],
            "F_batch_names": [None],
            "N_batch": 1,
            "train_energy": True,
            "train_forces": False,
            "N_remove": 0,
            "N_struc_E": 2,
            "N_struc_F": 0,
            "max_nnb": 0,
            "trainset_params": {},
            "setup_params": {},
            "networks_param": {},
        },
        "./tmp_batches/testds_info",
    )

    ds = _bare_dataset("gpu", train_forces=False, train_energy=True)
    ds.dataname = "testds"

    # Sentinel return proves the fixed line both calls via self and *uses* the
    # return value (regression of either bug would leave the real data in place).
    sentinel = [f"S{i}" for i in range(15)]
    monkeypatch.setattr(ds, "batch_data_cpu_to_gpu", lambda data: sentinel)

    ds.load_batches()

    assert ds.E_group_descrp[0] == "S10"
    assert ds.E_group_energy[0] == "S11"
    assert ds.E_group_N_atom[0] == "S14"
    assert ds.F_group_descrp[0] == "S0"
