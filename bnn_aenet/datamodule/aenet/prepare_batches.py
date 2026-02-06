from torch.utils.data import random_split

from .data_set import *
from .data_loader import *
from .read_trainset import *


def read_list_structures(tin):
	"""
	Read Training set files (*.train.ascii and *.train.forces)
	"""
	if tin.train_forces:
		list_structures_forces, list_structures_energy, list_removed, max_nnb, tin = read_train_forces_together(tin)
	else:
		list_structures_energy, list_removed, max_nnb, tin = read_train(tin)
		list_structures_forces = []
		max_nnb = 0

	input_size = tin.networks_param["input_size"]
	for struc in list_structures_energy:
		struc.padding(max_nnb, input_size)
	for struc in list_structures_forces:
		struc.padding(max_nnb, input_size)

	return list_structures_energy, list_structures_forces, list_removed, max_nnb, tin


def get_N_batch(len_dataset, batch_size):
	"""
	Returns the number of batches for a given batch size and dataset size
	"""
	N_batch = int(len_dataset/batch_size)
	residue = len_dataset - N_batch*batch_size

	if residue >= int(batch_size/2) or N_batch == 0:
		if residue != 0:
			N_batch += 1

	return N_batch


def split_database(dataset_size, valid_split, test_split, seed=42, data_dir=None, split_name="default", split_config=None):
	"""
	Returns indices of the structures in the training, validation, and test sets.
	
	Split indices are loaded from the following locations (in order):
	1. If split_config provided: bnn_aenet/data/{dataset}/indices/{split_config}/
	2. data_dir/splits/{split_name}/ (e.g., data/TiO/splits/energy/)
	3. data_dir/splits/
	4. Generate random split and save it
	
	Args:
		dataset_size: Total number of structures
		valid_split: Fraction (0-1) or count (>1) for validation set
		test_split: Fraction (0-1) or count (>1) for test set
		seed: Random seed for reproducible splits (default: 42)
		data_dir: Path to data directory (e.g., directory containing train.in)
		split_name: Name for this split (e.g., "energy", "forces") for separate files
		split_config: Named split configuration (e.g., "Data20", "Data100") stored in bnn_aenet/data/
	
	Returns:
		train_indices, valid_indices, test_indices: Lists of indices
	"""
	import os
	
	indices = list(range(dataset_size))
	
	if len(indices) == 0:
		return [], [], []
	
	# Helper function to try loading indices from a directory
	def try_load_indices(path_indices):
		"""Try to load indices from a directory, supporting both file naming conventions."""
		# Try new format first (train_indices.txt)
		file_patterns = [
			('train_indices.txt', 'valid_indices.txt', 'test_indices.txt'),
			('train_set_idxes.txt', 'valid_set_idxes.txt', 'test_set_idxes.txt'),
		]
		
		for train_name, valid_name, test_name in file_patterns:
			train_file = os.path.join(path_indices, train_name)
			valid_file = os.path.join(path_indices, valid_name)
			test_file = os.path.join(path_indices, test_name)
			
			if os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(test_file):
				try:
					train_idx = np.genfromtxt(train_file).astype(int)
					valid_idx = np.genfromtxt(valid_file).astype(int)
					test_idx = np.genfromtxt(test_file).astype(int)
					
					# Handle single-element arrays
					if train_idx.ndim == 0:
						train_idx = np.array([train_idx])
					if valid_idx.ndim == 0:
						valid_idx = np.array([valid_idx])
					if test_idx.ndim == 0:
						test_idx = np.array([test_idx])
					
					# Validate indices are within range
					all_idx = np.concatenate([train_idx, valid_idx, test_idx])
					if len(all_idx) > 0 and np.max(all_idx) < dataset_size:
						return list(train_idx), list(valid_idx), list(test_idx), path_indices
				except Exception as e:
					pass
		return None, None, None, None
	
	# Build list of paths to try
	possible_paths = []
	
	# Get base_dir from data_dir
	if data_dir is not None:
		if os.path.isfile(data_dir):
			base_dir = os.path.dirname(data_dir)
		else:
			base_dir = data_dir
	else:
		base_dir = None
	
	# 1. If split_config provided, ONLY look in bnn_aenet/data/{dataset}/indices/{split_config}/
	#    This takes exclusive precedence - no fallback to other locations
	if split_config is not None:
		# Get dataset name from data_dir
		if base_dir is not None:
			dataset_name = os.path.basename(base_dir.rstrip('/'))
		else:
			dataset_name = "unknown"
		
		# Find bnn_aenet root directory (this file is at bnn_aenet/datamodule/aenet/prepare_batches.py)
		bnn_aenet_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
		
		# Try various naming conventions (TiO -> TiO2, qm7 -> QM7, etc.)
		name_variants = [
			dataset_name,
			dataset_name.upper(),
			dataset_name.lower(),
			f"{dataset_name}2",  # TiO -> TiO2
			f"{dataset_name.upper()}2",
		]
		
		for name in name_variants:
			possible_paths.append(os.path.join(bnn_aenet_root, "data", name, "indices", split_config))
		
		# When split_config is specified, we ONLY use these paths - no fallback
		for path in possible_paths:
			train_idx, valid_idx, test_idx, loaded_from = try_load_indices(path)
			if train_idx is not None:
				print(f'Loaded split indices from: {loaded_from} (split_config={split_config})')
				print(f'  Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}')
				return train_idx, valid_idx, test_idx
		
		# split_config specified but not found - this is an error
		print(f'ERROR: split_config={split_config} specified but indices not found!')
		print(f'  Searched paths: {possible_paths}')
		print(f'  Falling back to random split...')
		# Clear paths so we fall through to random split generation
		possible_paths = []
	
	# 2. Try data_dir/splits/{split_name}/ (only if split_config not specified)
	if base_dir is not None:
		possible_paths.extend([
			os.path.join(base_dir, "splits", split_name),
			os.path.join(base_dir, "splits"),
		])
	
	# Try each path
	for path in possible_paths:
		train_idx, valid_idx, test_idx, loaded_from = try_load_indices(path)
		if train_idx is not None:
			print(f'Loaded split indices from: {loaded_from}')
			print(f'  Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}')
			return train_idx, valid_idx, test_idx
	
	# Generate random split
	print(f'Generating random split with seed={seed}')
	np.random.seed(seed)
	indices = np.random.permutation(dataset_size)
	
	# Calculate split sizes (valid_split and test_split can be percentages or counts)
	if valid_split > 1:
		n_valid = int(valid_split)
	else:
		n_valid = int(dataset_size * valid_split)
	
	if test_split > 1:
		n_test = int(test_split)
	else:
		n_test = int(dataset_size * test_split)
	
	n_train = dataset_size - n_valid - n_test
	
	train_indices = indices[:n_train]
	valid_indices = indices[n_train:n_train+n_valid]
	test_indices = indices[n_train+n_valid:]
	
	print(f'Split: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}')
	
	# Save indices for reproducibility (if data_dir provided)
	if data_dir is not None:
		if os.path.isfile(data_dir):
			base_dir = os.path.dirname(data_dir)
		else:
			base_dir = data_dir
		
		save_dir = os.path.join(base_dir, "splits", split_name)
		try:
			os.makedirs(save_dir, exist_ok=True)
			np.savetxt(os.path.join(save_dir, 'train_indices.txt'), train_indices, fmt='%d')
			np.savetxt(os.path.join(save_dir, 'valid_indices.txt'), valid_indices, fmt='%d')
			np.savetxt(os.path.join(save_dir, 'test_indices.txt'), test_indices, fmt='%d')
			print(f'Saved split indices to: {save_dir}')
		except Exception as e:
			print(f'Warning: Could not save indices to {save_dir}: {e}')
	
	return list(train_indices), list(valid_indices), list(test_indices)


def create_split_files(data_dir, train_indices, valid_indices, test_indices, split_name="default"):
	"""
	Create split index files in the data directory.
	
	Use this to create custom train/valid/test splits that will be loaded
	automatically by the datamodule.
	
	Args:
		data_dir: Path to data directory (e.g., 'data/TiO/')
		train_indices: List of indices for training set
		valid_indices: List of indices for validation set
		test_indices: List of indices for test set
		split_name: Name for this split (e.g., "energy", "forces")
	
	Example:
		# Create 80/10/10 split for 100 structures
		train = list(range(80))
		valid = list(range(80, 90))
		test = list(range(90, 100))
		create_split_files('data/TiO/', train, valid, test, split_name='energy')
	"""
	import os
	
	save_dir = os.path.join(data_dir, "splits", split_name)
	os.makedirs(save_dir, exist_ok=True)
	
	np.savetxt(os.path.join(save_dir, 'train_indices.txt'), train_indices, fmt='%d')
	np.savetxt(os.path.join(save_dir, 'valid_indices.txt'), valid_indices, fmt='%d')
	np.savetxt(os.path.join(save_dir, 'test_indices.txt'), test_indices, fmt='%d')
	
	print(f'Created split files in: {save_dir}')
	print(f'  Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}')


def select_batch_size(tin, list_structures_energy, list_structures_forces):
	"""
	Select batch size that best matches the requested size, avoiding the last batch being too small
	"""
	N_data_E = len(list_structures_energy)
	N_data_F = len(list_structures_forces)
	
	# Get data_dir and split_config from tin if available
	data_dir = getattr(tin, 'data_dir', None)
	split_config = getattr(tin, 'split_config', None)
	
	train_sampler_E, valid_sampler_E, test_sampler_E = split_database(
		N_data_E, tin.valid_split, tin.test_split, data_dir=data_dir, split_name="energy", split_config=split_config)
	train_sampler_F, valid_sampler_F, test_sampler_F = split_database(
		N_data_F, tin.valid_split, tin.test_split, data_dir=data_dir, split_name="forces", split_config=split_config)

	N_data_train_E = len(train_sampler_E)
	N_data_test_E = len(test_sampler_E)
	N_data_valid_E = len(valid_sampler_E)

	N_data_train_F = len(train_sampler_F)
	N_data_test_F = len(test_sampler_F)
	N_data_valid_F = len(valid_sampler_F)

	forcespercent  = N_data_F/(N_data_F + N_data_E)
	if forcespercent <= 0.5:
		tin.batch_size = round((1 - forcespercent)*tin.batch_size)
		N_batch_train = get_N_batch(N_data_train_E, tin.batch_size)
		N_batch_valid = get_N_batch(N_data_valid_E, tin.batch_size)
		N_batch_test = get_N_batch(N_data_test_E, tin.batch_size)
	else:
		tin.batch_size = forcespercent*tin.batch_size

		N_batch_train = get_N_batch(N_data_train_F, tin.batch_size)
		N_batch_test = get_N_batch(N_data_test_F, tin.batch_size)
		N_batch_valid = get_N_batch(N_data_valid_F, tin.batch_size)

	if N_data_F!= 0 and N_batch_train > N_data_F:
		N_batch_train = N_data_F
	
	if N_data_F!= 0 and N_batch_valid > N_data_F:
		N_batch_valid = N_data_F
	
	if N_data_F!= 0 and N_batch_test > N_data_F:
		N_batch_test = N_data_F
	
	train_set_size = len(train_sampler_E)
 
	return train_set_size, N_batch_train, N_batch_valid, N_batch_test


def select_batches(tin, trainset_params, device, list_structures_energy, list_structures_forces,
				   max_nnb, N_batch_train, N_batch_valid, N_batch_test):
	"""
	Select which structures belong to each batch for training.
	Returns: four objects of the class data_set_loader.PrepDataloader(), for train/test and energy/forces
	"""
	if len(list_structures_energy) != 0:
		dataset_energy = StructureDataset(list_structures_energy, tin.sys_species, tin.networks_param["input_size"], max_nnb)
		dataset_energy_size = len(dataset_energy)

		# Normalize
		E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
		sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
		dataset_energy.normalize_E(trainset_params.E_scaling, trainset_params.E_shift)
		stp_shift, stp_scale = dataset_energy.normalize_stp(sfval_avg, sfval_cov)

		# Split in train/test
		data_dir = getattr(tin, 'data_dir', None)
		split_config = getattr(tin, 'split_config', None)
		train_sampler_E, valid_sampler_E, test_sampler_E = split_database(
			dataset_energy_size, tin.valid_split, tin.test_split, data_dir=data_dir, split_name="energy", split_config=split_config)

		train_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_train,
		                               sampler=train_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="train_energy")
		valid_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_valid,
		                               sampler=valid_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="valid_energy")
		test_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=N_batch_test,
		                               sampler=test_sampler_E, memory_mode=tin.memory_mode, device=device, dataname="test_energy")
		all_energy_data = PrepDataloader(dataset=dataset_energy, train_forces=False, N_batch=len(dataset_energy),
		                               sampler=range(len(dataset_energy)), memory_mode=tin.memory_mode, device=device, dataname="all_energy")
  
	else:
		dataset_energy = None
		train_energy_data, valid_energy_data, test_energy_data = None, None, None


	if len(list_structures_forces) != 0:
		dataset_forces = StructureDataset(list_structures_forces, tin.sys_species, tin.networks_param["input_size"], max_nnb)
		dataset_forces_size = len(dataset_forces)

		# Normalize
		E_scaling, E_shift = tin.trainset_params.E_scaling, tin.trainset_params.E_shift
		sfval_avg, sfval_cov = tin.setup_params.sfval_avg, tin.setup_params.sfval_cov
		dataset_forces.normalize_E(trainset_params.E_scaling, trainset_params.E_shift)
		dataset_forces.normalize_F(trainset_params.E_scaling, trainset_params.E_shift)
		stp_shift, stp_scale = dataset_forces.normalize_stp(sfval_avg, sfval_cov)

		# Split in train/test
		data_dir = getattr(tin, 'data_dir', None)
		split_config = getattr(tin, 'split_config', None)
		train_sampler_F, valid_sampler_F, test_sampler_F = split_database(
			dataset_forces_size, tin.valid_split, tin.test_split, data_dir=data_dir, split_name="forces", split_config=split_config)

		train_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=True, N_batch=N_batch_train,
		                               sampler=train_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="train_forces")
		valid_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=True, N_batch=N_batch_valid,
		                               sampler=valid_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="valid_forces")
		test_forces_data = PrepDataloader(dataset=dataset_forces, train_forces=True, N_batch=N_batch_test,
		                               sampler=test_sampler_F, memory_mode=tin.memory_mode, device=device, dataname="test_forces")
		
	else:
		dataset_forces = None
		train_forces_data, valid_forces_data, test_forces_data = None, None, None

	return train_forces_data, valid_forces_data, test_forces_data, train_energy_data, valid_energy_data, test_energy_data, all_energy_data


def save_datsets(save, train_forces_data, valid_forces_data, train_energy_data, valid_energy_data):
	"""
	Saves datasets created by select_batches
	"""
	torch.save(save, "tmp_batches/trainset_info")
	torch.save(train_forces_data, "tmp_batches/train_forces_data.ph")
	torch.save(valid_forces_data, "tmp_batches/valid_forces_data.ph")
	torch.save(train_energy_data, "tmp_batches/train_energy_data.ph")
	torch.save(valid_energy_data, "tmp_batches/valid_energy_data.ph")
	#torch.save(grouped_train_data, "tmp_batches/grouped_train_data.ph")


def load_datasets():
	"""
	Loads saved datasets instead of preparing them
	"""
	save = torch.load("tmp_batches/trainset_info")
	N_removed, N_struc_E, N_struc_F, max_nnb, tin.trainset_params, tin.setup_params, tin.networks_param = save[:]

	train_forces_data = torch.load("tmp_batches/train_forces_data.ph")
	train_energy_data = torch.load("tmp_batches/train_energy_data.ph")
	train_forces_data.gather_data(tin.memory_mode)
	train_energy_data.gather_data(tin.memory_mode)

	grouped_train_data = GroupedDataset(train_energy_data, train_forces_data,
									 	memory_mode=tin.memory_mode, device=device, dataname="train")
	del train_forces_data
	del train_energy_data

	valid_forces_data = torch.load("tmp_batches/valid_forces_data.ph")
	valid_energy_data = torch.load("tmp_batches/valid_energy_data.ph")
	valid_forces_data.gather_data(tin.memory_mode)
	valid_energy_data.gather_data(tin.memory_mode)

	grouped_valid_data = GroupedDataset(valid_energy_data, valid_forces_data,
										memory_mode=tin.memory_mode, device=device, dataname="valid")

	del valid_forces_data
	del valid_energy_data

	return N_removed, N_struc_E, N_struc_F, max_nnb, grouped_train_data, grouped_valid_data