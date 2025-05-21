import yaml
from pathlib import Path

# Exact keys you want in the output
ALLOWED_KEYS = {
    "trainer.min_epochs",
    "experiment",
    "trainer.deterministic",
    "datamodule",
    "datamodule.valid_split",
    "datamodule.batch_size",
    "model.obs_scale",
    "model.lr",
    "model.mc_samples_train",
    "model.prior_scale",
    "model.q_scale",
}

def flatten_dict(d, parent_key=''):
    """Flatten nested dicts using dot notation, skipping model.net."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k

        if new_key.startswith("model.net"):
            continue
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items

def yaml_value_to_str(val):
    """Clean YAML scalar rendering for values."""
    return yaml.safe_dump(val, default_flow_style=True).replace("...", "").strip()

def hparams_to_overrides(hparams_path, overrides_path):
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)

    flattened = []
    for section in ['model', 'datamodule', 'trainer', 'experiment']:
        if section in hparams:
            flattened.extend(flatten_dict(hparams[section], parent_key=section))
        elif section in hparams:  # for flat keys like 'experiment'
            flattened.append((section, hparams[section]))

    # Only keep allowed keys
    filtered = [(k, v) for k, v in flattened if k in ALLOWED_KEYS]

    overrides = [f"- {key}={yaml_value_to_str(value)}" for key, value in filtered]

    overrides_path = Path(overrides_path)
    overrides_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overrides_path, 'w') as f:
        for line in overrides:
            f.write(f"{line}\n")

    return overrides

if __name__ == "__main__":
    import glob
    lrt = 'hom_lrt_hps_100perc/runs/2025-05-09_16-07-29'
    fo = 'hom_fo_hps_100perc/runs/2025-05-09_16-38-44'
    for model in [fo, lrt]:
        pattern = f'/home/g15farris/bin/bayesaenet/bnn_aenet/logs/{model}/*/tensorboard/version_0/hparams.yaml'
        for hpruns in glob.glob(pattern):
            overrides = Path(hpruns).parent.parent.parent / '.hydra' / 'overrides.yaml'
            hparams_to_overrides(hpruns, overrides)
