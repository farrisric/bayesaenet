# bayesaenet — Claude Code Context

## Prompt scope evaluation

Before executing any task, evaluate the prompt on these three dimensions:
1. **Scope** — is it too broad? (e.g. targets "everything", "the whole codebase", "all files")
2. **Specificity** — is there a clear target file, function, or behaviour?
3. **Feature creep** — does it ask for more than one distinct change?

If any dimension raises a concern, stop and report:
- What the concern is and why
- How to split the prompt into smaller focused tasks (with concrete examples)

Only proceed after I confirm.

## Project overview

**bayesaenet** is a research framework for benchmarking **uncertainty quantification (UQ)** methods in **machine learning interatomic potentials (MLIPs)**. It compares:
- **Deep Ensembles (DE)**: Multiple independent deterministic NNs
- **Variational Bayesian Neural Networks (VBNNs)**: Pyro/TyXe ELBO-based inference

Models predict DFT energies and forces for atomic structures (primary dataset: 7,815 TiO₂ structures). The codebase is built on PyTorch Lightning + Hydra + Pyro.

## Dev commands

```bash
# Install
pip install -e .

# Run tests
pytest tests/ -v

# Train (example)
python bnn_aenet/tasks/train.py experiment=bnn_lrt datamodule=TiO trainer=gpu

# Hyperparameter search
python bnn_aenet/tasks/hpsearch.py experiment=bnn_lrt datamodule=TiO trainer=gpu

# Predict
python bnn_aenet/tasks/predict.py ...

# Linting
black bnn_aenet/
isort bnn_aenet/
flake8 bnn_aenet/
```

## Architecture

```
bnn_aenet/
├── models/
│   ├── nets/network.py       # NetAtom: core element-specific atomic energy network
│   ├── nn.py                 # NN, NNBase: deterministic models
│   ├── bnn.py                # BNN: base variational BNN
│   ├── bnn_forces.py         # BNN_Forces: BNN with joint energy+force likelihood (canonical)
│   ├── bnn_forces_hetero.py  # Heteroscedastic variant
│   ├── likelihoods.py        # Custom Pyro model: joint energy+force log-likelihood
│   ├── likelihoods_hetero.py # Heteroscedastic likelihoods
│   └── guides/radial.py      # AutoRadial variational guide
├── datamodule/
│   ├── aenet_datamodule.py   # Lightning DataModule: reads train.in → batches
│   └── aenet/
│       ├── batch_constants.py # BatchIdx: named indices for batch tensors (no magic ints)
│       ├── data_set.py        # GroupedDataset: synchronized force+energy batches
│       └── prepare_batches.py # Batch construction, descriptor sorting by species
├── tasks/
│   ├── train.py              # Main entry point: setup → optional pretrain → Lightning trainer
│   ├── hpsearch.py           # Optuna HPS with per-model objective functions
│   ├── predict.py            # MC inference → Parquet output
│   └── predict_forces.py     # Force predictions
├── utils/
│   ├── metrics.py            # Per-atom RMSE, NLL, calibration error
│   ├── paths.py              # Log/result directory utilities
│   └── miscellaneous.py      # Result saving helpers
├── configs/                  # Hydra YAML configs (model, experiment, hpsearch, datamodule)
├── logs/                     # Training logs and checkpoints (gitignored)
└── results/                  # Optuna SQLite databases
```

### Model hierarchy

```
LightningModule
├── NNBase → NN (deterministic, with force training)
└── BNN (base variational)
    └── BNN_Forces (joint energy+force ELBO) ← canonical Bayesian model
        └── BNN_Forces_Hetero (per-atom/structure noise)
```

## Key design decisions

1. **Alpha fixed at 0.1**: Energy/force trade-off is *not* optimized by Optuna — would bias search toward one modality.
2. **force_weight in training only**: The validation metric excludes `force_weight` for fair HPS comparison across different weight values.
3. **Validation metric**: `total_rmse/val = (1 - alpha) * E_RMSE + alpha * F_RMSE` — no force_weight.
4. **Joint likelihood** (not auxiliary): Forces are integrated directly into the ELBO via `likelihoods.py`. Both energy and forces use the *same* weight sample in one forward pass — this is the theoretically rigorous approach.
5. **Learnable noise**: Setting `learn_noise=True` makes `obs_scale` and `scale_force` trainable Pyro parameters instead of fixed hyperparameters.
6. **LRT incompatible with mixed precision**: Using `fit_context=LRT` (Local Reparameterization Trick) with `torch.autocast` / AMP causes NaN losses. Always use `precision=32` with LRT.

## Batch structure

Batches from `GroupedDataset` are lists of 15 tensors, indexed via `BatchIdx`:

```
Indices 0-9:  Force data (optional — may be empty)
  [0]  F_DESCRP      - list[Tensor] per species, descriptors for force structures
  [5]  F_FORCES      - Tensor (total_atoms, 3)
  [6-9]             - Descriptor derivatives and atom indices

Indices 10-14: Energy data (always present)
  [10] E_DESCRP      - list[Tensor] per species, descriptors for energy structures
  [11] E_ENERGY      - Tensor (batch,) — per-structure total energy
  [14] E_N_ATOM      - Tensor (batch,) — number of atoms per structure

Use BatchIdx.get_force_data(batch) / BatchIdx.get_energy_data(batch) — never use raw ints.
```

## Configuration system (Hydra)

Config resolution order (later overrides earlier):
1. `configs/train.yaml` — base defaults
2. `configs/experiment/<name>.yaml` — model + trainer + callback presets
3. CLI overrides (e.g. `datamodule.batch_size=512`)

Key experiment configs: `bnn_lrt`, `bnn_rad`, `nn`, `bnn_forces_likelihood`

Log/result paths:
- `bnn_aenet/logs/<dataset>/<task>/<model>/run_<N>/`
- `bnn_aenet/results/<dataset>/<model>.db` (Optuna SQLite)

## Key hyperparameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| `lr` | 1e-5 – 1e-3 | Learning rate |
| `pretrain_epochs` | 0 or 5 | Warm-start from deterministic NN |
| `mc_samples_train` | 1 or 2 | MC samples during training |
| `prior_scale` | 0.1 – 0.5 | Prior distribution scale |
| `q_scale` | 1e-5 – 0.005 | Initial variational scale |
| `obs_scale` | 0.1 – 2.0 | Energy observation noise |
| `scale_force` | 0.05 – 2.0 | Force observation noise |
| `batch_size` | 128/256/512/1024 | |
| `learn_noise` | bool | Make obs_scale, scale_force trainable |

## NetAtom internals

- One sub-network per atomic species (element-specific parameters)
- `forward(grp_descrp, logic_reduce)` → structure energies
- `forward_F(grp_descrp, grp_sfderiv_i, grp_sfderiv_j, ...)` → energies + forces via autodiff through descriptors (`dE/dG * dG/dr`)
- Supported activations: tanh, sigmoid, relu, softplus, elu, gelu, silu

## Uncertainty quantification

At prediction time, BNN draws `mc_samples_eval` (default 20) weight samples from the variational posterior:
- **Epistemic uncertainty**: `std(E_preds)` across MC samples
- **Aleatoric uncertainty** (if `learn_noise=True`): learned `obs_scale`
- **Total**: `sqrt(epistemic² + aleatoric²)`

Metrics: NLL, calibration error, sharpness (via `uncertainty_toolbox`)

## Testing

```bash
pytest tests/ -v
# Key test files:
# tests/test_models.py              - model init and forward pass
# tests/test_datamodule.py          - data loading and batches
# tests/test_reproducibility.py     - seed control
# tests/test_tio2_runtime_repro.py  - integration test on TiO2
```

## Datasets

- **TiO₂ small**: 20% of 7,815 DFT structures — fast iteration
- **TiO₂ big**: Full 7,815 structures — benchmarking
- **QM7**: Organic molecules dataset
- Data format: AENET binary format, read via `bnn_aenet/datamodule/aenet/`

## Dependencies

Core: `torch`, `lightning`, `pyro-ppl`, `tyxe` (git), `hydra-core`, `optuna`
UQ: `uncertainty-toolbox`
Data: `fastparquet`, `pyarrow`
Dev: `pytest`, `black`, `isort`, `flake8`

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical label vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.