import sys
sys.path.append('/home/riccardo/bin/repos/aenet-bayesian/aenet')

import torch 
from torch.utils.data import DataLoader
import lightning as L

from data_classes import *
from read_input import *
from read_trainset import *
from network import *
from prepare_batches import *
from traininit import *
from data_set import *
from data_loader import *
from optimization_step import *
from output_nn import *
from py_aeio import *
import numpy as np

class AenetDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, device: str = 'cpu'):
        super().__init__()
        self.data_dir = data_dir
        self.device = device
        self.load_db()
        self.model = NetAtom(self.tin.networks_param["input_size"], self.tin.networks_param["hidden_size"],
			    self.tin.sys_species, self.tin.networks_param["activations"], self.tin.alpha, self.device).double()
    
    def load_db(self):
        self.tin = read_train_in(self.data_dir)
        torch.manual_seed(self.tin.pytorch_seed)
        np.random.seed(self.tin.numpy_seed)
        self.tin.device = self.device
        
        self.list_structures_energy, self.list_structures_forces, self.list_removed, self.max_nnb, self.tin = read_list_structures(self.tin)

        N_removed = len(self.list_removed)
        N_struc_E = len(self.list_structures_energy)
        N_struc_F = len(self.list_structures_forces)
        self.dataset_size = int(len(self.list_structures_energy)*(1-self.tin.test_split))

    def setup(self, stage: str):
        N_batch_train, N_batch_valid = select_batch_size(self.tin, 
                                                         self.list_structures_energy, 
                                                         self.list_structures_forces)

        # Join datasets with forces and only energies in a single torch dataset AND prepare batches
        train_forces_data, valid_forces_data, train_energy_data, valid_energy_data = select_batches(
            self.tin, self.tin.trainset_params, self.device, 
            self.list_structures_energy, self.list_structures_forces,
            self.max_nnb, N_batch_train, N_batch_valid)

        self.grouped_train_data = GroupedDataset(train_energy_data, train_forces_data,
									 memory_mode=self.tin.memory_mode, device=self.device, dataname="train")
        self.grouped_valid_data = GroupedDataset(valid_energy_data, valid_forces_data,
									 memory_mode=self.tin.memory_mode, device=self.device, dataname="valid")


    def train_dataloader(self):
        return DataLoader(self.grouped_train_data, batch_size=1, shuffle=False,
                                  collate_fn=custom_collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.grouped_valid_data, batch_size=1, shuffle=False,
                                        collate_fn=custom_collate, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.grouped_valid_data, batch_size=1, shuffle=False,
                                        collate_fn=custom_collate, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.grouped_valid_data, batch_size=1, shuffle=False,
                                        collate_fn=custom_collate, num_workers=0)

    def teardown(self, stage: str):
        pass

    def get_model(self):
        return self.model

def custom_collate(batch):
    return batch

if __name__ == '__main__':
    datamodule = AenetDataModule('/home/riccardo/bin/repos/aenet-bayesian/examples/PdO/train.in')
    datamodule.setup('ciao')
    net = datamodule.get_model()
    print(datamodule.dataset_size)