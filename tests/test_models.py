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
        """Test network initialization."""
        from bnn_aenet.models.nets.network import NetAtom
        
        input_size = 50
        hidden = [32, 32]
        output_size = 1
        
        net = NetAtom(
            input_size=input_size,
            hidden=hidden,
            output_size=output_size,
            device=device
        )
        
        assert net is not None
        assert net.fc_in.in_features == input_size
        assert net.fc_in.out_features == hidden[0]
    
    def test_forward(self, device, seed):
        """Test forward pass."""
        from bnn_aenet.models.nets.network import NetAtom
        
        batch_size = 4
        input_size = 50
        hidden = [32, 32]
        
        net = NetAtom(
            input_size=input_size,
            hidden=hidden,
            output_size=1,
            device=device
        )
        net.to(device)
        
        x = torch.randn(batch_size, input_size, device=device)
        logic = torch.ones(batch_size, 1, device=device)
        
        output = net(x, logic)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()


class TestBNNModels:
    """Tests for BNN model wrappers."""
    
    def test_bnn_import(self):
        """Test BNN module imports."""
        from bnn_aenet.models.bnn import BNN, NN
        assert BNN is not None
        assert NN is not None
    
    def test_nn_initialization(self, device):
        """Test NN model initialization."""
        from bnn_aenet.models.bnn import NN
        
        model = NN(
            input_size={"Ti": 50, "O": 40},
            hidden=[32, 32],
            output_size=1,
            lr=0.001,
            weight_decay=1e-5,
            dataset_size=100,
        )
        
        assert model is not None
        assert "Ti" in model.nets
        assert "O" in model.nets


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
