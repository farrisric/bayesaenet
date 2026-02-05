# Complete GPU Fix - Using Original AENET GPU Support

## Discovery
The original AENET-PyTorch code **already has full GPU support**! We just weren't using it correctly.

## Two Critical Fixes Applied

### Fix 1: train.in Configuration ✅
**File**: `/home/g15farris/bin/bayesaenet/data/QM7/train.in`

**Changed line 10:**
```diff
- MEMORY_MODE cpu
+ MEMORY_MODE gpu
```

This tells the dataloader to prepare batches for GPU memory.

### Fix 2: Device String Compatibility ✅
**File**: `/home/g15farris/bin/bayesaenet/bnn_aenet/datamodule/aenet/data_set.py`

**Changed lines 140-141:**
```diff
- if self.device == "cuda:0":
+ if "cuda" in str(self.device):
      data = self.batch_data_cpu_to_gpu(data)
```

This makes it work with both `"cuda"` and `"cuda:0"` device strings.

### Fix 3: Network Forward Pass ✅ (Already Applied)
**File**: `/home/g15farris/bin/bayesaenet/bnn_aenet/models/nets/network.py`

Added explicit `.to(device)` calls as a safety measure (lines 79, 84, 103).

## How the Original AENET GPU Support Works

From `data_set.py`, the `GroupedDataset.__getitem__()` method:

1. **Loads batch data from memory** (lines 121-126)
2. **Checks if GPU is being used** (line 140)
3. **Moves all tensors to GPU** (line 141) using `batch_data_cpu_to_gpu()`

The `batch_data_cpu_to_gpu()` method (lines 146-174) moves:
- ✅ `group_descrp` - descriptors
- ✅ `logic_tensor_reduce` - reduction tensors  
- ✅ `group_energy` - energies
- ✅ `group_N_atom` - atom counts
- ✅ `group_forces` - forces
- ✅ `group_sfderiv_i/j` - force derivatives
- ✅ All other training data

This is **exactly** what we needed! It was already there in the original code.

## Why It Wasn't Working

1. **train.in had `MEMORY_MODE cpu`** → dataloader didn't prepare for GPU
2. **Device string mismatch** → `"cuda:0"` check failed when Lightning passed `"cuda"`
3. **Extra safety needed** → Added explicit `.to(device)` in network code

## Test the Complete Fix

```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/de
qsub de_train_gpu_multirun.sh
```

## Expected Behavior Now

✅ **Data loads in GPU memory mode**
✅ **All tensors automatically moved to GPU** by `batch_data_cpu_to_gpu()`
✅ **Network operates on GPU** 
✅ **No device mismatch errors**
✅ **Fast GPU training with proper memory management**

## Files Modified

1. ✅ `/home/g15farris/bin/bayesaenet/data/QM7/train.in` - Set MEMORY_MODE to gpu
2. ✅ `/home/g15farris/bin/bayesaenet/bnn_aenet/datamodule/aenet/data_set.py` - Fix device check
3. ✅ `/home/g15farris/bin/bayesaenet/bnn_aenet/models/nets/network.py` - Add safety .to(device)
4. ✅ `/home/g15farris/bin/bayesaenet/scripts/QM7/de/de_train_gpu_multirun.sh` - GPU config

## Bonus: Check Other Datasets

You may want to update MEMORY_MODE in other train.in files too:
- `/home/g15farris/bin/bayesaenet/data/TiO/train.in`
- `/home/g15farris/bin/bayesaenet/data/PdO/train.in`  
- `/home/g15farris/bin/bayesaenet/data/IrO/train.in`
- `/home/g15farris/bin/bayesaenet/data/H2O/train.in`
