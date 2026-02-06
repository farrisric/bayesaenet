# BNN-AENET Improvement Recommendations

This document outlines identified issues and suggested improvements for the BNN-AENET library.

## High Priority Fixes

### 1. Duplicate Assignment (FIXED)
**Location:** `bnn_aenet/models/bnn.py:112`
**Issue:** `self.svi = self.svi = SVI(...)`
**Status:** Fixed

### 2. Magic Numbers in Batch Indices
**Location:** Throughout `bnn_aenet/models/bnn.py`
**Issue:** Hard-coded indices make code hard to understand:
```python
x = batch[10], batch[12]  # What do these mean?
y = batch[11]
```

**Suggestion:** Create batch index constants:
```python
# In bnn_aenet/datamodule/aenet/batch_constants.py
class BatchIdx:
    # Force data indices (when forces enabled)
    F_DESCRP = 0
    F_ENERGY = 1
    F_LOGIC_REDUCE = 2
    F_DB_INDEX = 3
    F_N_ATOM = 4
    F_FORCES = 5
    F_SFDERIV_I = 6
    F_SFDERIV_J = 7
    F_INDICES = 8
    F_INDICES_I = 9
    
    # Energy data indices
    E_DESCRP = 10
    E_ENERGY = 11
    E_LOGIC_REDUCE = 12
    E_DB_INDEX = 13
    E_N_ATOM = 14
```

### 3. Redundant SVI Creation per Step
**Location:** `bnn_aenet/models/bnn.py:121-124, 161-164`
**Issue:** `svi_no_obs` is recreated in every training/validation step:
```python
self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
self.svi_no_obs = SVI(...)  # Created every step
```

**Suggestion:** Create once in `on_fit_start()` and cache:
```python
def on_fit_start(self):
    # ... existing code ...
    self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
    self.svi_no_obs = SVI(
        self.bnn_no_obs, self.bnn.guide, self.optimizer, self.loss
    )

def training_step(self, batch, batch_idx):
    # ... use self.svi_no_obs directly ...
```

### 4. Standardize Alpha Access Pattern
**Location:** `bnn_aenet/models/bnn.py:639, 753, 798`
**Issue:** Inconsistent access:
```python
alpha = self.net.alpha.item() if hasattr(self.net, 'alpha') else 0.5  # Line 639
alpha = getattr(self.net, 'alpha', 0.5)  # Line 798
```

**Suggestion:** Use a property or helper method:
```python
@property
def alpha(self):
    """Get force weighting parameter (0-1)."""
    if hasattr(self.net, 'alpha'):
        a = self.net.alpha
        return a.item() if hasattr(a, 'item') else float(a)
    return 0.5
```

## Medium Priority

### 5. Exception Handling Too Broad
**Location:** `bnn_aenet/models/bnn.py:141-148, 177-184`
**Issue:** Bare `except Exception: pass` swallows all errors:
```python
try:
    if scale.min() > 0:
        rmsce = rms_calibration_error(...)
except Exception:
    pass  # Could hide bugs
```

**Suggestion:**
```python
except (ValueError, RuntimeError) as e:
    if self.trainer.is_global_zero:
        log.debug(f"Calibration metrics skipped: {e}")
```

### 6. Complex Split Index Loading
**Location:** `bnn_aenet/datamodule/aenet/prepare_batches.py:42-214`
**Issue:** Complex fallback logic with many path variants.

**Suggestion:** Refactor into a `SplitIndexLoader` class:
```python
class SplitIndexLoader:
    def __init__(self, data_dir: str, split_config: str = None):
        self.data_dir = data_dir
        self.split_config = split_config
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load or generate train/valid/test indices."""
        paths = self._get_candidate_paths()
        for path in paths:
            indices = self._try_load(path)
            if indices is not None:
                return indices
        return self._generate_random_split()
```

### 7. Add Type Hints
**Issue:** Many methods lack type hints, reducing IDE support.

**Example improvement:**
```python
def compute_force_loss_and_update(
    self,
    batch: List[torch.Tensor],
    alpha: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute force loss and update model parameters."""
```

### 8. Document Radial Guide Behavior
**Location:** `bnn_aenet/models/bnn.py:78`
**Issue:** Silent behavior change when using radial guide:
```python
elif self.hparams.guide == "radial":
    self.fit_ctxt = contextlib.nullcontext  # Not documented
```

**Suggestion:** Add logging:
```python
elif self.hparams.guide == "radial":
    log.info("Using radial guide - disabling fit context (LRT/Flipout not compatible)")
    self.fit_ctxt = contextlib.nullcontext
```

## Low Priority / Future Work

### 9. Force Units Configuration
**Location:** `bnn_aenet/models/bnn.py:567`
**Issue:** Hardcoded unit conversion:
```python
force_rmse = torch.sqrt(...) * 1000  # Convert to mHa/Bohr
```

**Suggestion:** Make configurable via config file.

### 10. Optimize MC Sampling in Prediction
**Location:** `bnn_aenet/models/bnn.py:843-928`
**Issue:** Sequential MC sampling is slow.

**Suggestion:** Vectorize sampling where possible.

### 11. Memory Mode Centralization
**Location:** `bnn_aenet/datamodule/aenet/data_set.py:120-146`
**Issue:** Device handling scattered across codebase.

**Suggestion:** Create `DeviceManager` utility class.

## Architecture Recommendations

### Network Architecture
For TiO2-like datasets (~7000 structures):
- **15:15** - Good baseline, ~900-1000 params per species
- **25:25** - Better capacity, recommended for production
- **50:25** - Good for complex energy landscapes
- **64:32:16** - Deep network, may overfit on small datasets

### Hyperparameter Suggestions
Based on analysis:

| Parameter | Current Range | Suggested Range | Rationale |
|-----------|---------------|-----------------|-----------|
| `lr` | 1e-5 to 1e-2 | 1e-4 to 1e-2 | Lower bound rarely useful |
| `q_scale` | 1e-5 to 0.005 | 1e-4 to 0.001 | Narrower for stability |
| `prior_scale` | 0.1 to 0.5 | 0.2 to 0.4 | Middle range works best |
| `batch_size` | 64-1024 | 128-512 | Very large may hurt generalization |

### Training Recommendations
1. **Pretrain epochs:** 200-500 for BNNs (helps stability)
2. **Early stopping patience:** 500-1000 epochs
3. **MC samples:** Train=1 (speed), Eval=30-50 (accuracy)
4. **Mixed precision:** Use for NN, Flipout, Radial; avoid for LRT

## Code Quality Checklist
- [ ] Add type hints to all public methods
- [ ] Replace magic numbers with constants
- [ ] Add docstrings to all classes
- [ ] Improve exception handling
- [ ] Add unit tests for edge cases
- [ ] Profile and optimize hot paths
- [ ] Add logging at appropriate levels
