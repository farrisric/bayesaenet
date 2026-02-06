# BNN-AENET Improvement Recommendations

This document outlines identified issues and suggested improvements for the BNN-AENET library.

## Implemented Fixes

### 1. Duplicate Assignment (FIXED)
**Location:** `bnn_aenet/models/bnn.py`
**Issue:** `self.svi = self.svi = SVI(...)`
**Status:** ✅ Fixed

### 2. Magic Numbers in Batch Indices (FIXED)
**Location:** Throughout `bnn_aenet/models/bnn.py`
**Issue:** Hard-coded indices like `batch[10]`, `batch[11]` were hard to understand.
**Status:** ✅ Fixed - Now using `BatchIdx` constants from `bnn_aenet/datamodule/aenet/batch_constants.py`:
```python
# Before
x = batch[10], batch[12]
y = batch[11]

# After  
x = batch[BatchIdx.E_DESCRP], batch[BatchIdx.E_LOGIC_REDUCE]
y = batch[BatchIdx.E_ENERGY]
```

### 3. Redundant SVI Creation per Step (FIXED)
**Location:** `bnn_aenet/models/bnn.py`
**Issue:** `svi_no_obs` was recreated in every training/validation step.
**Status:** ✅ Fixed - Now cached in `on_fit_start()`:
```python
def on_fit_start(self):
    # ... existing code ...
    # Cache svi_no_obs for performance
    self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
    self.svi_no_obs = SVI(
        self.bnn_no_obs, self.bnn.guide, self.optimizer, self.loss
    )
```

### 4. Mixed Precision Warning for LRT (FIXED)
**Location:** `bnn_aenet/models/bnn.py`
**Issue:** LRT causes NaN with mixed precision, but no warning was shown.
**Status:** ✅ Fixed - Now warns in `on_fit_start()` if LRT is used with `16-mixed` precision.

### 5. Type Hints Added (FIXED)
**Location:** `bnn_aenet/models/bnn.py`
**Status:** ✅ Fixed - Added type hints to all key methods.

### 6. Exception Handling Improved (FIXED)
**Location:** `bnn_aenet/models/bnn.py`
**Issue:** Bare `except Exception: pass` could hide bugs.
**Status:** ✅ Fixed - Changed to `except (ValueError, RuntimeError):`

## Configuration Recommendations

### Network Architecture: 15:15 vs 25:25

The current default architecture uses **15:15** (two hidden layers with 15 nodes each).
For datasets with more complex energy surfaces, consider using **25:25**:

**Current (data/TiO/train.in):**
```
NETWORKS
  Ti     Ti.pytorch.nn    2    15:tanh    15:tanh
  O       O.pytorch.nn    2    15:tanh    15:tanh
```

**Recommended for complex datasets:**
```
NETWORKS
  Ti     Ti.pytorch.nn    2    25:tanh    25:tanh
  O       O.pytorch.nn    2    25:tanh    25:tanh
```

**Trade-offs:**
- **15:15**: ~2,000 parameters per atom type, faster training, less prone to overfitting
- **25:25**: ~5,000 parameters per atom type, more capacity, better for complex energy surfaces
- **Larger networks (50:50)**: Consider only if you have >100k training samples

### Mixed Precision Guidelines

| Model Type | Mixed Precision | Recommended Setting |
|------------|-----------------|---------------------|
| NN         | ✅ Safe         | `precision=16-mixed` |
| Flipout    | ✅ Safe         | `precision=16-mixed` |
| Radial     | ✅ Safe         | `precision=16-mixed` |
| LRT        | ❌ Causes NaN   | `precision=32-true` or omit |

## Medium Priority (Future Work)

### 4. Standardize Alpha Access Pattern
**Location:** `bnn_aenet/models/bnn.py`
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

## Low Priority (Future Work)

### Add Logging for Skipped Calibration Metrics
**Location:** `bnn_aenet/models/bnn.py`
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
