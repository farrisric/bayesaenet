"""Generate training scripts from best HPS results.

Reads the best trial from each Optuna DB and generates ready-to-submit
SGE training scripts with the optimal hyperparameters.

Usage:
    python -m bnn_aenet.tasks.generate_train_scripts \
        --dataset TiO2_small \
        --output-dir scripts/final/TiO2_small/train
"""

import argparse
from pathlib import Path
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEEDS = [121958, 671155, 131932, 365838, 259178, 644167, 110268, 732180, 54886, 137337]

CONDA_BLOCK = """\
. /etc/profile
__conda_setup="$('/aplic/anaconda/2020.02/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/aplic/anaconda/2024.10/etc/profile.d/conda.sh" ]; then
        . "/aplic/anaconda/2024.10/etc/profile.d/conda.sh"
    else
        export PATH="/aplic/anaconda/2024.10/bin:$PATH"
    fi
fi
unset __conda_setup"""

ENV_BLOCK = """\
export OMP_NUM_THREADS=4
export TMPDIR=/tmp/g15farris
export PYTHONPATH=/home/g15farris/bin/bayesaenet:$PYTHONPATH
cd /home/g15farris/bin/bayesaenet"""

# Map dataset -> datamodule config name
DATAMODULE_MAP = {
    "TiO2_big": "TiO_Forces_Data100",
    "TiO2_small": "TiO_Forces_Data20",
}

# Map method -> (experiment config, needs cuda module, needs mixed precision, queue)
METHOD_CONFIG = {
    "nn": {
        "experiment": "nn_forces",
        "cuda_module": False,
        "mixed_precision": True,
        "queue": "iqtc13.q",
    },
    "lrt": {
        "experiment": "bnn_lrt_forces_aux",
        "cuda_module": True,
        "mixed_precision": False,  # LRT + mixed precision = NaN
        "queue": "iqtc10.q",
    },
    "lrt_likelihood": {
        "experiment": "bnn_lrt_forces_likelihood",
        "cuda_module": True,
        "mixed_precision": False,
        "queue": "iqtc10.q",
    },
    "fo": {
        "experiment": "bnn_fo_forces_aux",
        "cuda_module": True,
        "mixed_precision": True,
        "queue": "iqtc13.q",
    },
    "rad": {
        "experiment": "bnn_rad_forces_aux",
        "cuda_module": True,
        "mixed_precision": True,
        "queue": "iqtc13.q",
    },
    "rad_likelihood": {
        "experiment": "bnn_rad_forces_likelihood",
        "cuda_module": True,
        "mixed_precision": True,
        "queue": "iqtc13.q",
    },
}


def load_best_params(db_path: Path) -> dict:
    """Load best trial parameters from an Optuna DB.
    
    Automatically detects the study name inside the DB.
    """
    import sqlite3
    storage = f"sqlite:///{db_path.as_posix()}"
    
    # Auto-detect study name from the DB
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT study_name FROM studies").fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(f"No studies found in {db_path}")
    
    study_name = rows[0][0]  # Take first study
    study = optuna.load_study(study_name=study_name, storage=storage)
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    best = study.best_trial
    print(f"  Study '{study_name}': {n_complete} completed trials")
    print(f"  Best trial #{best.number}: value={best.value:.2f}")
    print(f"  Params: {best.params}")
    return best.params


def generate_nn_script(params: dict, dataset: str, output_dir: Path) -> Path:
    """Generate NN training script."""
    cfg = METHOD_CONFIG["nn"]
    datamodule = DATAMODULE_MAP[dataset]
    lr = params["lr"]
    # NN HPS doesn't tune batch_size, use default 128
    bs = params.get("batch_size", 128)

    seeds_str = " ".join(str(s) for s in SEEDS)
    precision_line = "        +trainer.precision=16-mixed \\\n" if cfg["mixed_precision"] else ""

    script = f"""#!/bin/bash
#$ -N multi_nn
#$ -q {cfg['queue']}
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/multirun/{dataset}_nn.out
#$ -e /home/g15farris/bin/bayesaenet/logs/multirun/{dataset}_nn.err

{CONDA_BLOCK}

{"module load cuda/12.4" if cfg["cuda_module"] else ""}
conda activate bnn

{ENV_BLOCK}

# Best NN HPS parameters
LR={lr}
BS={bs}

SEEDS=({seeds_str})

for i in $(seq 0 9); do
    echo "=== Starting NN run $i with seed ${{SEEDS[$i]}} at $(date) ==="
    python -m bnn_aenet.tasks.train \\
        experiment={cfg['experiment']} \\
        datamodule={datamodule} \\
        trainer.accelerator=gpu \\
        trainer.devices=1 \\
{precision_line}        trainer.max_epochs=50000 \\
        task_name=nn_train \\
        run_name=nn_train_${{i}} \\
        datamodule.batch_size=${{BS}} \\
        model.optimizer.lr=${{LR}} \\
        callbacks.model_checkpoint.monitor=total_rmse/val \\
        callbacks.early_stopping.monitor=total_rmse/val \\
        callbacks.early_stopping.patience=500 \\
        seed=${{SEEDS[$i]}} \\
        'tags=["{dataset}", "nn", "train"]'
    echo "=== Finished NN run $i at $(date) ==="
done
"""
    out_path = output_dir / "multirun_nn.sh"
    out_path.write_text(script)
    return out_path


def generate_bnn_script(
    method: str, params: dict, dataset: str, output_dir: Path, use_likelihood: bool = False
) -> Path:
    """Generate BNN training script (lrt/fo/rad)."""
    cfg_key = f"{method}_likelihood" if use_likelihood else method
    cfg = METHOD_CONFIG[cfg_key]
    datamodule = DATAMODULE_MAP[dataset]

    lr = params["lr"]
    mc = params.get("mc_samples_train", 2)
    prior_scale = params.get("prior_scale", 0.1)
    q_scale = params.get("q_scale", 0.001)
    obs_scale = params.get("obs_scale", 0.5)
    scale_force = params.get("scale_force", None)
    bs = params.get("batch_size", 256)

    seeds_str = " ".join(str(s) for s in SEEDS)
    precision_line = "        +trainer.precision=16-mixed \\\n" if cfg["mixed_precision"] else ""

    # Display name for logs (strip _likelihood suffix)
    display_name = method.replace("_likelihood", "")

    scale_force_lines = ""
    if scale_force is not None:
        scale_force_lines = f"        model.scale_force=${{SCALE_FORCE}} \\\n"

    script = f"""#!/bin/bash
#$ -N multi_{display_name}
#$ -q {cfg['queue']}
#$ -l iqtcgpu=1
#$ -pe smp 4
#$ -S /bin/bash
#$ -cwd
#$ -o /home/g15farris/bin/bayesaenet/logs/multirun/{dataset}_{display_name}.out
#$ -e /home/g15farris/bin/bayesaenet/logs/multirun/{dataset}_{display_name}.err

{CONDA_BLOCK}

{"module load cuda/12.4" if cfg["cuda_module"] else ""}
conda activate bnn

{ENV_BLOCK}

# Best {display_name.upper()} HPS parameters
LR={lr}
BS={bs}
MC={mc}
PRIOR_SCALE={prior_scale}
Q_SCALE={q_scale}
OBS_SCALE={obs_scale}
"""
    if scale_force is not None:
        script += f"SCALE_FORCE={scale_force}\n\n"

    script += f"""SEEDS=({seeds_str})

for i in $(seq 0 9); do
    echo "=== Starting {display_name.upper()} run $i with seed ${{SEEDS[$i]}} at $(date) ==="
    python -m bnn_aenet.tasks.train \\
        experiment={cfg['experiment']} \\
        datamodule={datamodule} \\
        trainer.accelerator=gpu \\
        trainer.devices=1 \\
{precision_line}        trainer.max_epochs=50000 \\
        task_name={display_name}_train \\
        run_name={display_name}_train_${{i}} \\
        datamodule.batch_size=${{BS}} \\
        model.lr=${{LR}} \\
        model.mc_samples_train=${{MC}} \\
        model.prior_scale=${{PRIOR_SCALE}} \\
        model.q_scale=${{Q_SCALE}} \\
        model.obs_scale=${{OBS_SCALE}} \\
        model.pretrain_epochs=0 \\
{scale_force_lines}        callbacks.model_checkpoint.monitor=total_rmse/val \\
        callbacks.early_stopping.monitor=total_rmse/val \\
        callbacks.early_stopping.patience=500 \\
        seed=${{SEEDS[$i]}} \\
        'tags=["{dataset}", "{display_name}", "train"]'
    echo "=== Finished {display_name.upper()} run $i at $(date) ==="
done
"""
    out_path = output_dir / f"multirun_{display_name}.sh"
    out_path.write_text(script)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate training scripts from best HPS results"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., TiO2_small, TiO2_big)")
    parser.add_argument("--output-dir", required=True, help="Directory to write training scripts")
    parser.add_argument("--results-dir", default=None, help="Path to results dir (default: bnn_aenet/results)")
    parser.add_argument("--methods", nargs="+", default=["nn", "lrt", "fo", "rad"],
                        help="Methods to generate scripts for")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "bnn_aenet" / "results"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset
    db_dir = results_dir / dataset

    if not db_dir.exists():
        print(f"ERROR: No results directory found at {db_dir}")
        return

    if dataset not in DATAMODULE_MAP:
        print(f"ERROR: Unknown dataset '{dataset}'. Known: {list(DATAMODULE_MAP.keys())}")
        return

    # Find DBs -- try both "{method}.db" and "{method}_{suffix}.db" patterns
    print(f"=== Generating training scripts for {dataset} ===")
    print(f"Results dir: {db_dir}")
    print(f"Output dir:  {output_dir}")
    print()

    for method in args.methods:
        # Try to find the DB file
        # For lrt/rad: prefer likelihood DB (bnn_*_forces_likelihood.db) when it exists
        use_likelihood = False
        if method in ("lrt", "rad"):
            likelihood_db = db_dir / f"bnn_{method}_forces_likelihood.db"
            aux_db = db_dir / f"{method}_small.db" if dataset == "TiO2_small" else db_dir / f"{method}.db"
            if likelihood_db.exists():
                db_path = likelihood_db
                use_likelihood = True
            elif aux_db.exists():
                db_path = aux_db
            else:
                db_candidates = list(db_dir.glob(f"{method}*.db"))
                db_path = db_candidates[0] if db_candidates else None
        else:
            db_candidates = list(db_dir.glob(f"{method}*.db"))
            db_path = db_candidates[0] if db_candidates else None

        if db_path is None or not db_path.exists():
            print(f"[{method}] No DB found, skipping")
            continue

        print(f"[{method}] Loading from {db_path.name}" + (" (likelihood)" if use_likelihood else ""))
        try:
            params = load_best_params(db_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if method == "nn":
            out_path = generate_nn_script(params, dataset, output_dir)
        else:
            out_path = generate_bnn_script(method, params, dataset, output_dir, use_likelihood=use_likelihood)

        print(f"  Written: {out_path}")
        print()

    print("=== Done! Review scripts then submit with qsub ===")


if __name__ == "__main__":
    main()
