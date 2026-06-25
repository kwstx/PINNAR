import torch
from hybrid_model import HybridSpectralSpatialModel

def test_hybrid_model():
    batch_size = 4
    num_bands = 200
    num_endmembers = 5
    patch_size = 32
    
    # Initialize the model
    model = HybridSpectralSpatialModel(num_bands=num_bands, num_endmembers=num_endmembers)
    
    # Create fake batch of 32x32 patches
    x = torch.rand(batch_size, num_bands, patch_size, patch_size)
    
    # Forward pass
    abundances, ssa, log_var, abundance_log_var, domain_logits, hr_abundances, hr_ssa = model(x)
    
    # Assert correct shapes
    assert abundances.shape == (batch_size, num_endmembers, patch_size, patch_size), f"Expected abundances shape (4, 5, 32, 32), got {abundances.shape}"
    assert ssa.shape == (batch_size, num_bands, patch_size, patch_size), f"Expected ssa shape (4, 200, 32, 32), got {ssa.shape}"
    assert log_var.shape == (batch_size, 1, patch_size, patch_size), f"Expected log_var shape (4, 1, 32, 32), got {log_var.shape}"
    assert abundance_log_var.shape == (batch_size, num_endmembers, patch_size, patch_size), f"Expected abundance_log_var shape (4, 5, 32, 32), got {abundance_log_var.shape}"
    assert domain_logits.shape == (batch_size, 3), f"Expected domain_logits shape (4, 3), got {domain_logits.shape}"
    assert hr_abundances.shape == (batch_size, num_endmembers, patch_size*2, patch_size*2), f"Expected hr_abundances shape (4, 5, 64, 64), got {hr_abundances.shape}"
    assert hr_ssa.shape == (batch_size, num_bands, patch_size*2, patch_size*2), f"Expected hr_ssa shape (4, 200, 64, 64), got {hr_ssa.shape}"
    
    # Check physical constraints
    # Abundances sum to 1 over the endmember dimension
    sum_abundances = abundances.sum(dim=1)
    assert torch.allclose(sum_abundances, torch.ones_like(sum_abundances), atol=1e-5), "Abundances do not sum to 1"
    
    # SSA between 0 and 1
    assert torch.all(ssa >= 0) and torch.all(ssa <= 1), "SSA out of bounds [0, 1]"
    
    print("All architecture tests passed successfully!")

if __name__ == '__main__':
    test_hybrid_model()
