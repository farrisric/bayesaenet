# GPU Training Fix V2 - Complete Device Transfer

## Problem
Even after the initial fix, tensors were still on different devices causing:
```
RuntimeError: Expected all tensors to be on the same device, 
but found at least two devices, cuda:0 and cpu!
```

## Root Cause
The **input data** from the dataloader was on CPU and wasn't being transferred to GPU.
Only fixing `logic_reduce` wasn't enough - we also need to move `grp_descrp` to GPU.

## Complete Fix Applied

### File: `bnn_aenet/models/nets/network.py`

#### 1. Fixed `forward()` method (lines 75-85)
```python
# Before:
partial_E_ann[iesp] = self.functions[iesp](grp_descrp[iesp].float())
list_E_ann = list_E_ann + torch.einsum("ij,ki->k", partial_E_ann[iesp], logic_reduce[iesp])

# After:
# Ensure input descriptors are on the same device as the model
grp_descrp_device = grp_descrp[iesp].to(self.device).float()
partial_E_ann[iesp] = self.functions[iesp](grp_descrp_device)

# Ensure logic_reduce is on the same device as the model
logic_reduce_device = logic_reduce[iesp].to(self.device)
list_E_ann = list_E_ann + torch.einsum("ij,ki->k", partial_E_ann[iesp], logic_reduce_device)
```

#### 2. Fixed `forward_F()` method (lines 101-115)
```python
# Before:
group_descrp[iesp].requires_grad_(True)

# After:
# Ensure input descriptors are on the same device as the model
group_descrp[iesp] = group_descrp[iesp].to(self.device)
group_descrp[iesp].requires_grad_(True)
```

## What Changed

1. **Input descriptors (`grp_descrp`)**: Now moved to GPU before processing
2. **Logic reduce tensors (`logic_reduce`)**: Now moved to GPU before einsum operations
3. **Both methods fixed**: `forward()` for energy-only training and `forward_F()` for force training

## Test the Fix

```bash
cd /home/g15farris/bin/bayesaenet/scripts/QM7/de

# Submit multirun job
qsub de_train_gpu_multirun.sh

# Monitor for device errors
tail -f train_de_gpu_multirun.err
```

## Expected Result

✅ No more device mismatch errors
✅ Training should proceed on GPU
✅ You should see progress bars and training metrics

## If Still Not Working

Check:
1. Is GPU actually available? `nvidia-smi` in the job
2. Are there other tensor operations that need device transfer?
3. Check the error log carefully for which specific tensor is on the wrong device
