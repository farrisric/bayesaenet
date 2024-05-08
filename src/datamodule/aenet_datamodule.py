import sys
sys.path.append('/home/g15farris/bin/bayesaenet/src/datamodule/aenet')

import torch 
from torch.utils.data import DataLoader
import pytorch_lightning as L

from src.datamodule.aenet.read_input import read_train_in
from src.datamodule.aenet.prepare_batches import select_batch_size, select_batches, read_list_structures
from src.datamodule.aenet.data_set import GroupedDataset

import numpy as np

class AenetDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_dir: str, 
            device: str = 'cpu', 
            batch_size: int = 128,
            test_split: float = 0.10,
            valid_split: float = 0.10,
        ):
        super().__init__()
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.test_split = test_split

        self.load_db()
        self.input_size = self.tin.networks_param["input_size"]
        self.hidden_size = self.tin.networks_param["hidden_size"]
        self.species = self.tin.sys_species
        self.active_names = self.tin.networks_param["activations"]
        self.alpha = self.tin.alpha
        #override
    
    def load_db(self):
        self.tin = read_train_in(self.data_dir)
        self.tin.batch_size = self.batch_size
        self.tin.test_split = self.test_split
        self.tin.valid_split = self.valid_split
        torch.manual_seed(self.tin.pytorch_seed)
        np.random.seed(self.tin.numpy_seed)
        self.tin.device = self.device
        self.list_structures_energy, self.list_structures_forces, self.list_removed, self.max_nnb, self.tin = read_list_structures(self.tin)

        N_removed = len(self.list_removed)
        N_struc_E = len(self.list_structures_energy)
        N_struc_F = len(self.list_structures_forces)

        self.train_size = int(len(self.list_structures_energy)*(1-self.tin.test_split-self.tin.valid_split))

        N_batch_train, N_batch_valid, N_batch_test = select_batch_size(self.tin, 
                                                         self.list_structures_energy, 
                                                         self.list_structures_forces)
        # Join datasets with forces and only energies in a single torch dataset AND prepare batches
        train_forces_data, valid_forces_data, test_forces_data, train_energy_data, valid_energy_data, test_energy_data = select_batches(
            self.tin, self.tin.trainset_params, self.device, 
            self.list_structures_energy, self.list_structures_forces,
            self.max_nnb, N_batch_train, N_batch_valid, N_batch_test)
        
        self.grouped_train_data = GroupedDataset(train_energy_data, train_forces_data,
									 memory_mode=self.tin.memory_mode, device=self.device, dataname="train")
        self.grouped_valid_data = GroupedDataset(valid_energy_data, valid_forces_data,
									 memory_mode=self.tin.memory_mode, device=self.device, dataname="valid")
        self.grouped_test_data = GroupedDataset(test_energy_data, test_forces_data,
									 memory_mode=self.tin.memory_mode, device=self.device, dataname="test")

        
    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return self.grouped_train_data
    
    def val_dataloader(self):
        return self.grouped_valid_data

    def test_dataloader(self):
        return self.grouped_test_data

    def predict_dataloader(self):
        return self.grouped_test_data

    def teardown(self, stage: str):
        pass

def custom_collate(batch):
    return batch
