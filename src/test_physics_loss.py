import torch
from physics_loss import HapkePhysicsLoss

def test_hapke_physics_loss():
    print("Testing Hapke Physics Loss...")
    
    # 1. Setup mock data
    N_bands = 50
    N_endmembers = 4
    Batch = 8
    
    # Mock optical constants (SSAs should be between 0 and 1)
    endmember_ssas = torch.rand(N_endmembers, N_bands) * 0.8 + 0.1 # Range 0.1 to 0.9
    
    # Initialize loss function
    loss_fn = HapkePhysicsLoss(endmember_ssas, phase_function_g=0.2, lambda_mse=1.0, lambda_sam=1.0)
    
    # Mock network outputs
    # Reflectance should be positive
    predicted_reflectance = torch.rand(Batch, N_bands, requires_grad=True)
    # Abundances should sum to 1. We mock it with softmax.
    raw_abundances = torch.randn(Batch, N_endmembers, requires_grad=True)
    predicted_abundances = torch.softmax(raw_abundances, dim=1)
    
    # Mock illumination geometries
    mu0 = torch.rand(Batch) * 0.5 + 0.5 # cos(inc_angle), range 0.5 to 1.0
    mu = torch.rand(Batch) * 0.5 + 0.5  # cos(em_angle), range 0.5 to 1.0
    phase_angle = torch.rand(Batch) * 1.5 # rad
    
    # 2. Forward pass
    total_loss, hapke_ref = loss_fn(predicted_reflectance, predicted_abundances, mu0, mu, phase_angle)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    assert not torch.isnan(total_loss), "Loss is NaN!"
    
    # 3. Backward pass
    total_loss.backward()
    
    # 4. Check gradients
    print(f"Gradient norm on predicted reflectance: {predicted_reflectance.grad.norm().item():.4f}")
    print(f"Gradient norm on raw abundances: {raw_abundances.grad.norm().item():.4f}")
    assert predicted_reflectance.grad is not None, "Gradients did not flow to reflectance!"
    assert raw_abundances.grad is not None, "Gradients did not flow to abundances!"
    assert not torch.isnan(predicted_reflectance.grad).any(), "NaN in reflectance gradients!"
    assert not torch.isnan(raw_abundances.grad).any(), "NaN in abundance gradients!"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_hapke_physics_loss()
