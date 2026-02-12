# Deprecated: results/bayesian/

This directory was used by older BNN_Forces_Likelihood HPS runs that passed
`tags=["bayesian", "lrt", "forces", "likelihood", "hps"]`, causing DBs to be
stored here instead of the dataset-specific location.

**Dataset used:** `datamodule=TiO_Forces` with `split_config=null`
- Default splits from `data/TiO/splits/` (NOT TiO2 Data20 or Data100 indices)
- Train size: ~600 energy / ~200 forces structures

**Files:**
- `bnn_lrt_forces_likelihood.db` - LRT likelihood HPS (best total_rmse/val ~43.08)
- `bnn_rad_forces_likelihood.db` - RAD likelihood HPS (best total_rmse/val ~45.03)

**For new runs**, use dataset-specific scripts:
- TiO2_big: `scripts/final/TiO2_big/hps/submit_hps_likelihood.sh`
- TiO2_small: `scripts/final/TiO2_small/hps/submit_hps_likelihood.sh`

DBs will be stored in `results/TiO2_big/` and `results/TiO2_small/` respectively.
