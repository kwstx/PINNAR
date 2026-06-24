import torch
from deep_ensemble import DeepEnsemble

def test_deep_ensemble_uncertainties():
    B = 2
    num_bands = 10
    num_endmembers = 4
    patch_size = 16
    
    ensemble = DeepEnsemble(num_models=3, num_bands=num_bands, num_endmembers=num_endmembers)
    
    x = torch.rand(B, num_bands, patch_size, patch_size)
    
    mean_abundances, epistemic_unc, aleatoric_unc = ensemble.predict_with_uncertainty(x)
    
    expected_shape = (B, num_endmembers, patch_size, patch_size)
    
    assert mean_abundances.shape == expected_shape, "Mean abundances shape mismatch"
    assert epistemic_unc.shape == expected_shape, "Epistemic uncertainty shape mismatch"
    assert aleatoric_unc.shape == expected_shape, "Aleatoric uncertainty shape mismatch"
    
    assert torch.all(epistemic_unc >= 0), "Epistemic uncertainty must be non-negative"
    assert torch.all(aleatoric_unc >= 0), "Aleatoric uncertainty must be non-negative"
    
    # Abundances should sum to 1 approximately
    sum_abund = mean_abundances.sum(dim=1)
    assert torch.allclose(sum_abund, torch.ones_like(sum_abund), atol=1e-5), "Mean abundances do not sum to 1"

if __name__ == "__main__":
    test_deep_ensemble_uncertainties()
