import torch
from composite_loss import CompositeLoss

def test_composite_loss_forward():
    B = 2
    N_bands = 10
    N_endmembers = 3
    H = W = 8

    # Create dummy endmember ssas
    endmember_ssas = torch.rand(N_endmembers, N_bands)
    
    # Initialize loss
    loss_fn = CompositeLoss(endmember_ssas, init_s_data=1.0, init_s_physics=-1.0, init_s_reg=0.0)

    # Inputs
    predicted_reflectance = torch.rand(B, N_bands, H, W)
    observed_reflectance = torch.rand(B, N_bands, H, W)
    predicted_abundances = torch.rand(B, N_endmembers, H, W)
    
    # Softmax constraint mock
    predicted_abundances = predicted_abundances / predicted_abundances.sum(dim=1, keepdim=True)
    
    log_var = torch.randn(B, 1, H, W)
    abundance_log_var = torch.randn(B, N_endmembers, H, W)
    
    mu0 = torch.rand(B)
    mu = torch.rand(B)
    phase_angle = torch.rand(B)
    
    # Mask labeled (e.g. half the pixels labeled)
    mask_labeled = torch.randint(0, 2, (B, H, W), dtype=torch.bool)
    
    # Abundances labels for a subset
    observed_abundances = torch.rand(B, N_endmembers, H, W)
    observed_abundances = observed_abundances / observed_abundances.sum(dim=1, keepdim=True)
    mask_abundances = torch.randint(0, 2, (B, H, W), dtype=torch.bool)
    
    # Forward pass
    total_loss, loss_dict = loss_fn(
        predicted_reflectance,
        observed_reflectance,
        predicted_abundances,
        log_var,
        abundance_log_var,
        mu0,
        mu,
        phase_angle,
        mask_labeled=mask_labeled,
        observed_abundances=observed_abundances,
        mask_abundances=mask_abundances
    )

    assert total_loss.ndim == 0, "Total loss should be a scalar"
    assert torch.isfinite(total_loss), "Loss must be finite"
    
    print("Test passed. Loss dict:", loss_dict)

if __name__ == "__main__":
    test_composite_loss_forward()
