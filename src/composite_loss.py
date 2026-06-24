import torch
import torch.nn as nn
from physics_loss import HapkePhysicsLoss

class CompositeLoss(nn.Module):
    """
    Composite loss function for PINNAR hybrid model.
    Combines Data Loss (MSE + SAM), Physics Loss (Hapke residual), and Uncertainty Regularization.
    Uses Homoscedastic Uncertainty Weighting to balance the three terms dynamically.
    """
    def __init__(self, endmember_ssas, init_s_data=0.0, init_s_physics=0.0, init_s_reg=0.0, phase_function_g=0.0):
        super().__init__()
        # Learnable log-variance parameters for homoscedastic weighting of the tasks
        self.s_data = nn.Parameter(torch.tensor(init_s_data, dtype=torch.float32))
        self.s_physics = nn.Parameter(torch.tensor(init_s_physics, dtype=torch.float32))
        self.s_reg = nn.Parameter(torch.tensor(init_s_reg, dtype=torch.float32))
        
        self.physics_module = HapkePhysicsLoss(endmember_ssas, phase_function_g=phase_function_g)
        self.mse = nn.MSELoss()
        
    def spectral_angle_mapper(self, predicted, target):
        dot_product = torch.sum(predicted * target, dim=1)
        norm_pred = torch.norm(predicted, dim=1)
        norm_target = torch.norm(target, dim=1)
        
        cos_theta = dot_product / (norm_pred * norm_target + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
        return torch.mean(torch.acos(cos_theta))

    def forward(self, predicted_reflectance, observed_reflectance, predicted_abundances, log_var, mu0, mu, phase_angle, mask_labeled=None):
        """
        predicted_reflectance: (B, N_bands) or (B, N_bands, H, W)
        observed_reflectance: (B, N_bands) or (B, N_bands, H, W)
        predicted_abundances: (B, N_endmembers) or (B, N_endmembers, H, W)
        log_var: (B, 1) or (B, 1, H, W) - Predictive log-variance
        mask_labeled: boolean mask indicating which pixels have observed reflectance
        """
        # Flatten spatial dimensions if present
        if predicted_reflectance.dim() == 4:
            B, C, H, W = predicted_reflectance.shape
            predicted_reflectance = predicted_reflectance.permute(0, 2, 3, 1).reshape(-1, C)
            observed_reflectance = observed_reflectance.permute(0, 2, 3, 1).reshape(-1, C)
            predicted_abundances = predicted_abundances.permute(0, 2, 3, 1).reshape(-1, predicted_abundances.shape[1])
            log_var = log_var.permute(0, 2, 3, 1).reshape(-1, 1)
            
            # Broadcast viewing geometry if they are scalar per batch
            if mu0.dim() == 1:
                mu0 = mu0.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                mu = mu.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                phase_angle = phase_angle.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
            
            if mask_labeled is not None:
                mask_labeled = mask_labeled.view(-1)
        
        # 1. Data Loss (only on labeled pixels)
        if mask_labeled is not None and mask_labeled.any():
            pred_labeled = predicted_reflectance[mask_labeled]
            obs_labeled = observed_reflectance[mask_labeled]
            loss_mse = self.mse(pred_labeled, obs_labeled)
            loss_sam = self.spectral_angle_mapper(pred_labeled, obs_labeled)
            l_data = loss_mse + loss_sam
        else:
            # If no mask is provided, assume all are labeled, or fallback
            loss_mse = self.mse(predicted_reflectance, observed_reflectance)
            loss_sam = self.spectral_angle_mapper(predicted_reflectance, observed_reflectance)
            l_data = loss_mse + loss_sam
            mask_labeled = torch.ones(predicted_reflectance.shape[0], dtype=torch.bool, device=predicted_reflectance.device)

        # 2. Physics Loss (on all pixels)
        # We only need the hapke reflectance, not the full loss from the physics module
        # So we pass 0.0 for lambdas to avoid unused computation if possible, but actually we can just extract hapke_ref
        _, hapke_ref = self.physics_module(predicted_reflectance, predicted_abundances, mu0, mu, phase_angle)
        l_physics = self.mse(predicted_reflectance, hapke_ref)
        
        # 3. Uncertainty-aware Regularization
        # Heteroscedastic formulation: exp(-log_var) * MSE + log_var
        # Computed per pixel on labeled data (or all data against physics if unlabeled, but let's use labeled data)
        pred_labeled_reg = predicted_reflectance[mask_labeled]
        obs_labeled_reg = observed_reflectance[mask_labeled]
        log_var_labeled = log_var[mask_labeled]
        
        mse_per_pixel = torch.mean((pred_labeled_reg - obs_labeled_reg)**2, dim=1, keepdim=True)
        l_reg = torch.mean(torch.exp(-log_var_labeled) * mse_per_pixel + log_var_labeled)
        
        # 4. Homoscedastic Task Weighting
        loss_data_weighted = torch.exp(-self.s_data) * l_data + self.s_data
        loss_physics_weighted = torch.exp(-self.s_physics) * l_physics + self.s_physics
        loss_reg_weighted = torch.exp(-self.s_reg) * l_reg + self.s_reg
        
        total_loss = loss_data_weighted + loss_physics_weighted + loss_reg_weighted
        
        return total_loss, {
            'l_data': l_data.item(),
            'l_physics': l_physics.item(),
            'l_reg': l_reg.item(),
            's_data': self.s_data.item(),
            's_physics': self.s_physics.item(),
            's_reg': self.s_reg.item(),
        }
