import pytest
import numpy as np
from src.sensitivity_analysis import run_hyperparameter_sweep, evaluate_hapke_model

def test_run_hyperparameter_sweep():
    # Mock data loaders
    data_loader = None
    val_loader = None
    
    # Run mock sweep
    results = run_hyperparameter_sweep(None, data_loader, val_loader)
    
    # 3 physics weights * 3 fourier features * 3 da_regs = 27 combinations
    assert len(results) == 27
    assert 'physics_weight' in results[0]
    assert 'fourier_features' in results[0]
    assert 'da_reg' in results[0]
    assert 'val_loss' in results[0]

def test_evaluate_hapke_model():
    # Test with a few known parameters
    # [SSA, g_param]
    params = np.array([
        [0.5, 0.0],
        [0.9, 0.5],
        [0.1, -0.5]
    ])
    
    reflectances = evaluate_hapke_model(params)
    
    assert len(reflectances) == 3
    # Reflectance should be positive
    assert all(r > 0 for r in reflectances)
