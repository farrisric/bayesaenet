"""Tests for model architectures."""

import pytest
import torch


class TestNetAtom:
    """Tests for the NetAtom network."""
    
    def test_import(self):
        """Test that NetAtom can be imported."""
        from bnn_aenet.models.nets.network import NetAtom
        assert NetAtom is not None
    
    def test_initialization(self, device):
        """Test network initialization with current API."""
        from bnn_aenet.models.nets.network import NetAtom
        
        # Current API requires these parameters
        # input_size is a list: [descriptor_size per species]
        # hidden_size is a 2D list: [[hidden_per_layer] for each species]
        # active_names is a 2D list: [[activation_per_layer] for each species]
        input_size = [50, 40]  # Ti=50, O=40
        hidden_size = [[15, 15], [15, 15]]  # 2 hidden layers per species
        species = ["Ti", "O"]
        active_names = [["tanh", "tanh"], ["tanh", "tanh"]]  # activations per layer per species
        alpha = 0.1
        e_scaling = 1.0
        e_shift = 0.0
        
        net = NetAtom(
            input_size=input_size,
            hidden_size=hidden_size,
            species=species,
            active_names=active_names,
            alpha=alpha,
            device=device,
            e_scaling=e_scaling,
            e_shift=e_shift
        )
        
        assert net is not None
        assert len(net.functions) == len(species)
    
    @pytest.mark.skip(reason="Forward pass requires properly batched data from DataLoader")
    def test_forward(self, device, seed):
        """Test forward pass - requires DataLoader for proper input format."""
        # The NetAtom network expects a specific input format from the DataLoader
        # that includes properly grouped descriptors and logic reduction tensors.
        # This test is skipped; forward pass is tested via integration tests.
        pass


class TestBNNModels:
    """Tests for BNN model wrappers."""
    
    def test_bnn_import(self):
        """Test BNN module imports."""
        from bnn_aenet.models.bnn import BNN, NN, BNN_Forces_Aux, NN_Forces
        assert BNN is not None
        assert NN is not None
        assert BNN_Forces_Aux is not None
        assert NN_Forces is not None
    
    def test_nn_initialization(self, device):
        """Test NN model initialization with current API."""
        from bnn_aenet.models.bnn import NN
        from bnn_aenet.models.nets.network import NetAtom
        
        # First create the network with proper list/2D structure
        net = NetAtom(
            input_size=[50, 40],  # list, not dict
            hidden_size=[[15, 15], [15, 15]],  # 2D: per species
            species=["Ti", "O"],
            active_names=[["tanh", "tanh"], ["tanh", "tanh"]],  # 2D: per species
            alpha=0.1,
            device=device,
            e_scaling=1.0,
            e_shift=0.0
        )
        
        # NN expects a net and optimizer (as a partial/callable)
        model = NN(
            net=net,
            optimizer=lambda params: torch.optim.Adam(params, lr=0.001),
        )
        
        assert model is not None
        assert model.net is not None


class TestConfigs:
    """Tests for Hydra configurations."""
    
    def test_hydra_config_exists(self, project_root):
        """Test that main configs exist."""
        config_dir = project_root / "bnn_aenet" / "configs"
        
        assert (config_dir / "train.yaml").exists()
        assert (config_dir / "model").exists()
        assert (config_dir / "datamodule").exists()
    
    def test_experiment_configs_exist(self, project_root):
        """Test experiment configs exist."""
        exp_dir = project_root / "bnn_aenet" / "configs" / "experiment"
        
        # Check BNN configs
        assert (exp_dir / "bnn_lrt.yaml").exists()
        assert (exp_dir / "bnn_fo.yaml").exists()
        assert (exp_dir / "bnn_rad.yaml").exists()
        assert (exp_dir / "nn.yaml").exists()
    
    def test_final_configs_exist(self, project_root):
        """Test final training configs exist."""
        final_dir = project_root / "bnn_aenet" / "configs" / "experiment" / "final"
        
        # Check final configs
        assert (final_dir / "lrt_qm7.yaml").exists()
        assert (final_dir / "lrt_tio2.yaml").exists()
        assert (final_dir / "fo_qm7.yaml").exists()
        assert (final_dir / "fo_tio2.yaml").exists()
        assert (final_dir / "rad_qm7.yaml").exists()
        assert (final_dir / "rad_tio2.yaml").exists()
        assert (final_dir / "de_qm7.yaml").exists()
        assert (final_dir / "de_tio2.yaml").exists()
