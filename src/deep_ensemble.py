import torch
import torch.nn as nn
import torch.optim as optim
import random
from hybrid_model import HybridSpectralSpatialModel
from composite_loss import CompositeLoss

class DataAugmentor(nn.Module):
    """
    Applies data augmentations to hyperspectral patches for ensemble diversity.
    """
    def __init__(self, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        if not self.training:
            return x
        
        # 1. Add Gaussian noise to simulate sensor noise
        noise = torch.randn_like(x) * self.noise_std
        x_aug = x + noise
        
        # 2. Random horizontal/vertical flips
        if random.random() > 0.5:
            x_aug = torch.flip(x_aug, dims=[3])
        if random.random() > 0.5:
            x_aug = torch.flip(x_aug, dims=[2])
            
        return torch.clamp(x_aug, 0.0, 1.0)


class DeepEnsemble(nn.Module):
    """
    Wraps multiple independent PINN models to compute epistemic and aleatoric uncertainties.
    """
    def __init__(self, num_models, num_bands, num_endmembers, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            HybridSpectralSpatialModel(num_bands, num_endmembers, **model_kwargs) 
            for _ in range(num_models)
        ])
        
    def forward(self, x):
        """
        Returns a list of outputs from all models in the ensemble.
        Each output is (abundances, ssa, log_var, abundance_log_var)
        """
        return [model(x) for model in self.models]

    @torch.no_grad()
    def predict_with_uncertainty(self, x):
        """
        Computes mean predictions and uncertainties.
        Returns:
            mean_abundances: (B, num_endmembers, H, W)
            epistemic_unc: (B, num_endmembers, H, W) - Variance of predicted means
            aleatoric_unc: (B, num_endmembers, H, W) - Mean of predicted variances
        """
        outputs = self(x)
        
        # Extract abundances and aleatoric variances for each model
        # abundances shape: (num_models, B, num_endmembers, H, W)
        all_abundances = torch.stack([out[0] for out in outputs])
        # abundance_log_vars shape: (num_models, B, num_endmembers, H, W)
        all_abund_log_vars = torch.stack([out[3] for out in outputs])
        
        # Compute mean abundance
        mean_abundances = all_abundances.mean(dim=0)
        
        # Epistemic Uncertainty (Variance of means)
        epistemic_unc = all_abundances.var(dim=0, unbiased=True)
        
        # Aleatoric Uncertainty (Mean of variances)
        # variance = exp(log_var)
        aleatoric_unc = torch.exp(all_abund_log_vars).mean(dim=0)
        
        return mean_abundances, epistemic_unc, aleatoric_unc


def train_ensemble(ensemble, dataloader, endmember_ssas, num_epochs=10, lr=1e-3, noise_std=0.01):
    """
    Trains all models in the ensemble independently.
    """
    device = next(ensemble.parameters()).device
    augmentor = DataAugmentor(noise_std=noise_std)
    
    # We maintain separate optimizers and loss functions for each model
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble.models]
    loss_fns = [CompositeLoss(endmember_ssas).to(device) for _ in ensemble.models]
    for loss_fn, opt in zip(loss_fns, optimizers):
        opt.add_param_group({'params': loss_fn.parameters(), 'lr': lr*10}) # Higher LR for learnable weights
        
    ensemble.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x, y_true, mu0, mu, phase_angle, mask_labeled, obs_abund, mask_abund = [
                b.to(device) if b is not None else None for b in batch
            ]
            
            # Train each model independently
            for i, model in enumerate(ensemble.models):
                optimizers[i].zero_grad()
                
                # Apply data augmentation
                x_aug = augmentor(x)
                
                abundances, ssa, log_var, abundance_log_var = model(x_aug)
                
                # Mock observed_reflectance logic for the data loss
                predicted_reflectance = ssa.mean(dim=(-1, -2)) # This is just a mock for training loop structure
                observed_reflectance = y_true.mean(dim=(-1, -2)) if y_true is not None else None
                
                # For a real implementation, predicted_reflectance comes from HapkePhysicsLoss output or explicitly
                # Here we pass it directly to the composite loss.
                loss, loss_dict = loss_fns[i](
                    predicted_reflectance,
                    observed_reflectance,
                    abundances,
                    log_var,
                    abundance_log_var,
                    mu0,
                    mu,
                    phase_angle,
                    mask_labeled=mask_labeled,
                    observed_abundances=obs_abund,
                    mask_abundances=mask_abund
                )
                
                loss.backward()
                optimizers[i].step()
                
        print(f"Epoch {epoch+1}/{num_epochs} completed.")
        
    return ensemble
