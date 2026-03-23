"""Tests for hpsearch behavior when learn_noise is enabled."""

from types import SimpleNamespace


def _build_cfg(learn_noise: bool):
    model = SimpleNamespace(
        pretrain_epochs=0,
        lr=1e-4,
        mc_samples_train=1,
        prior_scale=0.2,
        q_scale=1e-3,
        obs_scale=0.5,
        scale_force=0.1,
        learn_noise=learn_noise,
        net=SimpleNamespace(alpha=0.1),
    )
    hpsearch = SimpleNamespace(monitor="total_rmse/val")
    datamodule = SimpleNamespace(batch_size=128)
    paths = SimpleNamespace(output_dir="")
    return SimpleNamespace(model=model, hpsearch=hpsearch, datamodule=datamodule, paths=paths)


class _FakeTrial:
    def __init__(self):
        self.float_calls = []
        self.categorical_calls = []
        self.number = 0

    def suggest_float(self, name, low, high, log=False):
        self.float_calls.append(name)
        return low

    def suggest_categorical(self, name, choices):
        self.categorical_calls.append(name)
        return choices[0]


def test_learn_noise_true_skips_obs_scale_and_scale_force(monkeypatch):
    from bnn_aenet.tasks import hpsearch

    monkeypatch.setattr(
        hpsearch,
        "objective",
        lambda trial, cfg, output_dir: 0.0,
    )

    cfg = _build_cfg(learn_noise=True)
    trial = _FakeTrial()
    hpsearch.objective_bnn_forces_likelihood(trial, cfg, "unused")

    assert "obs_scale" not in trial.float_calls
    assert "scale_force" not in trial.float_calls


def test_learn_noise_false_still_optimizes_obs_scale_and_scale_force(monkeypatch):
    from bnn_aenet.tasks import hpsearch

    monkeypatch.setattr(
        hpsearch,
        "objective",
        lambda trial, cfg, output_dir: 0.0,
    )

    cfg = _build_cfg(learn_noise=False)
    trial = _FakeTrial()
    hpsearch.objective_bnn_forces_likelihood(trial, cfg, "unused")

    assert "obs_scale" in trial.float_calls
    assert "scale_force" in trial.float_calls
