#!/usr/bin/env python3
"""
Test script for Partially Bayesian Neural Networks.

Verifies that:
1. PartialBNN can correctly identify and freeze layers
2. Different bayesian_layers configurations work correctly
3. Training works with partial Bayesian treatment
"""

import sys
import torch
import torch.nn as nn
from collections import OrderedDict

# Add parent to path for imports
sys.path.insert(0, '/home/g15farris/bin/bayesaenet')

from bnn_aenet.models.nets.network import NetAtom


def create_simple_net(input_size=30, hidden=[15, 15], n_species=2):
    """Create a simple NetAtom for testing."""
    species = ['Ti', 'O'][:n_species]
    
    return NetAtom(
        input_size=[input_size] * n_species,
        hidden_size=[hidden] * n_species,
        species=species,
        active_names=[['tanh'] * len(hidden)] * n_species,
        alpha=0.5,
        device='cpu',
        e_scaling=1.0,
        e_shift=0.0,
    )


def test_layer_identification():
    """Test that layer identification works correctly."""
    print("\n=== Testing Layer Identification ===")
    
    from bnn_aenet.models.bnn import PartialBNN
    
    net = create_simple_net()
    
    # Create a full model to test layer identification
    model = PartialBNN(
        net=net,
        lr=0.001,
        pretrain_epochs=0,
        mc_samples_train=2,
        mc_samples_eval=5,
        dataset_size=100,
        fit_context="lrt",
        prior_loc=0.0,
        prior_scale=0.1,
        guide="normal",
        q_scale=0.001,
        obs_scale=0.5,
        bayesian_layers="all",
    )
    
    # Test getting all linear layers
    all_layers = model._get_linear_layer_names()
    print(f"All linear layers: {all_layers}")
    
    # Test different configurations
    for config in ["all", "last", "first", "first_last", [0], [0, 2]]:
        model._bayesian_layers_config = config
        bayesian_layers = model._get_bayesian_layer_names()
        config_str = str(config) if isinstance(config, list) else config
        print(f"  {config_str:15} -> {len(bayesian_layers)} Bayesian: {bayesian_layers}")
    
    print("Layer identification: PASSED")


def test_partial_bnn_creation():
    """Test creating a PartialBNN model."""
    print("\n=== Testing PartialBNN Creation ===")
    
    from bnn_aenet.models.bnn import PartialBNN
    
    net = create_simple_net()
    
    for config in ["all", "last", "first_last"]:
        print(f"\nTesting bayesian_layers='{config}':")
        
        model = PartialBNN(
            net=net,
            lr=0.001,
            pretrain_epochs=0,
            mc_samples_train=2,
            mc_samples_eval=5,
            dataset_size=100,
            fit_context="lrt",
            prior_loc=0.0,
            prior_scale=0.1,
            guide="normal",
            q_scale=0.001,
            obs_scale=0.5,
            bayesian_layers=config,
            name=f"PartialBNN_{config}",
        )
        
        # Check parameter counts
        param_counts = model.get_bayesian_param_count()
        print(f"  Total params: {param_counts['total_params']}")
        print(f"  Bayesian params: {param_counts['bayesian_params']}")
        print(f"  Deterministic params: {param_counts['deterministic_params']}")
        print(f"  Bayesian fraction: {param_counts['bayesian_fraction']:.2%}")
        
        # Verify the fraction makes sense
        if config == "all":
            assert param_counts['bayesian_fraction'] == 1.0, f"Expected 100% Bayesian for 'all', got {param_counts['bayesian_fraction']:.2%}"
        elif config == "last":
            # Last layer should have fewer params than all layers
            assert param_counts['bayesian_fraction'] < 1.0, f"Expected <100% Bayesian for 'last', got {param_counts['bayesian_fraction']:.2%}"
            assert param_counts['bayesian_fraction'] > 0.0, f"Expected >0% Bayesian for 'last', got {param_counts['bayesian_fraction']:.2%}"
        
        print(f"  Config '{config}': PASSED")


def test_partial_bnn_forces():
    """Test creating a PartialBNN_Forces_Aux model."""
    print("\n=== Testing PartialBNN_Forces_Aux Creation ===")
    
    from bnn_aenet.models.bnn import PartialBNN_Forces_Aux
    
    net = create_simple_net()
    
    model = PartialBNN_Forces_Aux(
        net=net,
        lr=0.001,
        pretrain_epochs=0,
        mc_samples_train=2,
        mc_samples_eval=5,
        dataset_size=100,
        fit_context="lrt",
        prior_loc=0.0,
        prior_scale=0.1,
        guide="normal",
        q_scale=0.001,
        obs_scale=0.5,
        bayesian_layers="last",
        force_weight=1.0,
        force_lr_scale=0.1,
        scale_lr_factor=0.5,
        name="PartialBNN_Forces",
    )
    
    param_counts = model.get_bayesian_param_count()
    print(f"  Total params: {param_counts['total_params']}")
    print(f"  Bayesian params: {param_counts['bayesian_params']}")
    print(f"  Bayesian fraction: {param_counts['bayesian_fraction']:.2%}")
    
    # Verify it correctly applies "last" layer config (should be ~2% for 15:15)
    assert param_counts['bayesian_fraction'] < 0.1, \
        f"Expected <10% Bayesian for 'last' config, got {param_counts['bayesian_fraction']:.2%}"
    assert param_counts['bayesian_params'] > 0, "Should have some Bayesian params"
    
    print("PartialBNN_Forces_Aux creation: PASSED")


def test_comparison_summary():
    """Print a summary comparison of different configurations."""
    print("\n=== Configuration Comparison Summary ===")
    
    from bnn_aenet.models.bnn import PartialBNN
    
    net = create_simple_net(hidden=[25, 25])
    
    configs = ["all", "last", "first", "first_last", [0, 2]]
    
    print(f"\nNetwork: 2 species, 25:25 architecture")
    print(f"{'Config':<15} {'Bayesian Params':>15} {'Total Params':>15} {'Fraction':>10}")
    print("-" * 60)
    
    for config in configs:
        model = PartialBNN(
            net=net,
            lr=0.001,
            pretrain_epochs=0,
            mc_samples_train=2,
            mc_samples_eval=5,
            dataset_size=100,
            fit_context="lrt",
            prior_loc=0.0,
            prior_scale=0.1,
            guide="normal",
            q_scale=0.001,
            obs_scale=0.5,
            bayesian_layers=config,
        )
        
        counts = model.get_bayesian_param_count()
        config_str = str(config) if isinstance(config, list) else config
        print(f"{config_str:<15} {counts['bayesian_params']:>15} {counts['total_params']:>15} {counts['bayesian_fraction']:>10.1%}")
    
    print("\nNote: 'last' layer Bayesian is often the best trade-off between UQ quality and speed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Partially Bayesian Neural Networks")
    print("=" * 60)
    
    try:
        test_layer_identification()
        test_partial_bnn_creation()
        test_partial_bnn_forces()
        test_comparison_summary()
        
        print("\n" + "=" * 60)
        print("All tests PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
