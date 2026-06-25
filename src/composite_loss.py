import torch
import torch.nn as nn
import torch.nn.functional as F
from physics_loss import HapkePhysicsLoss

class CompositeLoss(nn.Module):
    """
    Composite loss function for PINNAR hybrid model.
    Combines Data Loss (MSE + SAM), Physics Loss (Hapke residual), and Uncertainty Regularization.
    Uses Homoscedastic Uncertainty Weighting to balance the three terms dynamically.
    """
    def __init__(self, endmember_ssas, init_s_data=0.0, init_s_physics=0.0, init_s_reg=0.0, init_s_abundance=0.0, init_s_domain=0.0, init_s_sr=0.0, phase_function_g=0.0, use_physics_loss=True, use_uncertainty_reg=True, use_domain_adapt=True):
        super().__init__()
        self.use_physics_loss = use_physics_loss
        self.use_uncertainty_reg = use_uncertainty_reg
        self.use_domain_adapt = use_domain_adapt
        # Learnable log-variance parameters for homoscedastic weighting of the tasks
        self.s_data = nn.Parameter(torch.tensor(init_s_data, dtype=torch.float32))
        self.s_physics = nn.Parameter(torch.tensor(init_s_physics, dtype=torch.float32))
        self.s_reg = nn.Parameter(torch.tensor(init_s_reg, dtype=torch.float32))
        self.s_abundance = nn.Parameter(torch.tensor(init_s_abundance, dtype=torch.float32))
        self.s_domain = nn.Parameter(torch.tensor(init_s_domain, dtype=torch.float32))
        self.s_sr = nn.Parameter(torch.tensor(init_s_sr, dtype=torch.float32))
        
        self.physics_module = HapkePhysicsLoss(endmember_ssas, phase_function_g=phase_function_g)
        self.mse = nn.MSELoss()
        
    def spectral_angle_mapper(self, predicted, target):
        dot_product = torch.sum(predicted * target, dim=1)
        norm_pred = torch.norm(predicted, dim=1)
        norm_target = torch.norm(target, dim=1)
        
        cos_theta = dot_product / (norm_pred * norm_target + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
        return torch.mean(torch.acos(cos_theta))

    def forward(self, predicted_reflectance, observed_reflectance, predicted_abundances, log_var, abundance_log_var, mu0, mu, phase_angle, mask_labeled=None, observed_abundances=None, mask_abundances=None, domain_logits=None, domain_labels=None, hr_abundances=None, hr_ssa=None):
        """
        predicted_reflectance: (B, N_bands) or (B, N_bands, H, W)
        observed_reflectance: (B, N_bands) or (B, N_bands, H, W)
        predicted_abundances: (B, N_endmembers) or (B, N_endmembers, H, W)
        log_var: (B, 1) or (B, 1, H, W) - Predictive log-variance for reflectance
        abundance_log_var: (B, N_endmembers) or (B, N_endmembers, H, W) - Aleatoric variance for abundances
        mask_labeled: boolean mask indicating which pixels have observed reflectance
        observed_abundances: ground truth abundances for calibration pixels
        mask_abundances: boolean mask indicating which pixels have observed abundances
        """
        # Flatten spatial dimensions if present
        predicted_reflectance_orig_dim = predicted_reflectance.dim()
        if predicted_reflectance_orig_dim == 4:
            B, C, H, W = predicted_reflectance.shape
            observed_reflectance_orig = observed_reflectance
            mu0_orig = mu0
            mu_orig = mu
            phase_angle_orig = phase_angle
            predicted_reflectance = predicted_reflectance.permute(0, 2, 3, 1).reshape(-1, C)
            observed_reflectance = observed_reflectance.permute(0, 2, 3, 1).reshape(-1, C)
            predicted_abundances = predicted_abundances.permute(0, 2, 3, 1).reshape(-1, predicted_abundances.shape[1])
            log_var = log_var.permute(0, 2, 3, 1).reshape(-1, 1)
            abundance_log_var = abundance_log_var.permute(0, 2, 3, 1).reshape(-1, abundance_log_var.shape[1])
            if observed_abundances is not None:
                observed_abundances = observed_abundances.permute(0, 2, 3, 1).reshape(-1, observed_abundances.shape[1])
            
            # Broadcast viewing geometry if they are scalar per batch
            if mu0.dim() == 1:
                mu0 = mu0.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                mu = mu.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                phase_angle = phase_angle.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
            
            if mask_labeled is not None:
                if mask_labeled.dim() == 1:
                    mask_labeled = mask_labeled.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                else:
                    mask_labeled = mask_labeled.view(-1)
            if mask_abundances is not None:
                if mask_abundances.dim() == 1:
                    mask_abundances = mask_abundances.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                else:
                    mask_abundances = mask_abundances.view(-1)
            if domain_labels is not None and domain_labels.dim() == 1:
                domain_labels_pixel = domain_labels.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
            else:
                domain_labels_pixel = domain_labels
        else:
            domain_labels_pixel = domain_labels
        
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
        l_physics = torch.tensor(0.0, device=predicted_reflectance.device)
        hapke_ref = predicted_reflectance # default
        if self.use_physics_loss:
            _, hapke_ref = self.physics_module(predicted_reflectance, predicted_abundances, mu0, mu, phase_angle)
            l_physics = self.mse(predicted_reflectance, hapke_ref)
        
        # 3. Uncertainty-aware Regularization
        # Heteroscedastic formulation: exp(-log_var) * MSE + log_var
        l_reg = torch.tensor(0.0, device=predicted_reflectance.device)
        if self.use_uncertainty_reg:
            pred_labeled_reg = predicted_reflectance[mask_labeled]
            obs_labeled_reg = observed_reflectance[mask_labeled]
            log_var_labeled = log_var[mask_labeled]
            
            mse_per_pixel = torch.mean((pred_labeled_reg - obs_labeled_reg)**2, dim=1, keepdim=True)
            l_reg = torch.mean(torch.exp(-log_var_labeled) * mse_per_pixel + log_var_labeled)
        
        # 4. Supervised Abundance Loss (Aleatoric NLL) on pixels with known mineralogy
        l_abundance = torch.tensor(0.0, device=predicted_reflectance.device)
        if observed_abundances is not None and mask_abundances is not None and mask_abundances.any():
            pred_abund_labeled = predicted_abundances[mask_abundances]
            obs_abund_labeled = observed_abundances[mask_abundances]
            abund_log_var_labeled = abundance_log_var[mask_abundances]
            
            # Heteroscedastic NLL for abundances (per-endmember)
            mse_abund_per_pixel = (pred_abund_labeled - obs_abund_labeled)**2
            l_abundance = torch.mean(torch.exp(-abund_log_var_labeled) * mse_abund_per_pixel + abund_log_var_labeled)
        
        # 5. Domain Adaptation & Domain-Specific Physics Consistency
        l_domain = torch.tensor(0.0, device=predicted_reflectance.device)
        if self.use_domain_adapt and domain_logits is not None and domain_labels is not None:
            l_domain = F.cross_entropy(domain_logits, domain_labels)
            
            # Domain-Specific Physics Consistency (variance of mean residuals across domains)
            if self.use_physics_loss:
                pixel_residuals = torch.mean((predicted_reflectance - hapke_ref)**2, dim=1)
                domain_means = []
                for d in torch.unique(domain_labels_pixel):
                    d_mask = (domain_labels_pixel == d)
                    if d_mask.any():
                        domain_means.append(pixel_residuals[d_mask].mean())
                if len(domain_means) > 1:
                    l_domain_physics = torch.var(torch.stack(domain_means))
                    l_domain = l_domain + 0.1 * l_domain_physics # Weight it slightly
                
        # 6. Super-Resolution Consistency
        l_sr = torch.tensor(0.0, device=predicted_reflectance.device)
        if hr_abundances is not None and hr_ssa is not None and predicted_reflectance_orig_dim == 4:
            B_hr, C_a, H_hr, W_hr = hr_abundances.shape
            hr_mu0 = mu0_orig.unsqueeze(1).unsqueeze(2).expand(B_hr, H_hr, W_hr).reshape(-1)
            hr_mu = mu_orig.unsqueeze(1).unsqueeze(2).expand(B_hr, H_hr, W_hr).reshape(-1)
            hr_phase = phase_angle_orig.unsqueeze(1).unsqueeze(2).expand(B_hr, H_hr, W_hr).reshape(-1)
            
            hr_abundances_flat = hr_abundances.permute(0, 2, 3, 1).reshape(-1, C_a)
            dummy_ref = torch.zeros((B_hr * H_hr * W_hr, 1), device=predicted_reflectance.device)
            _, hr_hapke_ref = self.physics_module(dummy_ref, hr_abundances_flat, hr_mu0, hr_mu, hr_phase)
            
            hr_hapke_ref_img = hr_hapke_ref.reshape(B_hr, H_hr, W_hr, -1).permute(0, 3, 1, 2)
            lr_hapke_ref_img = F.adaptive_avg_pool2d(hr_hapke_ref_img, (H, W))
            
            l_sr = self.mse(lr_hapke_ref_img, observed_reflectance_orig)

        # 7. Homoscedastic Task Weighting
        loss_data_weighted = torch.exp(-self.s_data) * l_data + self.s_data
        loss_physics_weighted = torch.exp(-self.s_physics) * l_physics + self.s_physics
        loss_reg_weighted = torch.exp(-self.s_reg) * l_reg + self.s_reg
        loss_abundance_weighted = torch.exp(-self.s_abundance) * l_abundance + self.s_abundance
        loss_domain_weighted = torch.exp(-self.s_domain) * l_domain + self.s_domain
        loss_sr_weighted = torch.exp(-self.s_sr) * l_sr + self.s_sr
        
        total_loss = loss_data_weighted + loss_physics_weighted + loss_reg_weighted + loss_abundance_weighted + loss_domain_weighted + loss_sr_weighted
        
        
        return total_loss, {
            'l_data': l_data.item(),
            'l_physics': l_physics.item(),
            'l_reg': l_reg.item(),
            'l_abundance': l_abundance.item() if isinstance(l_abundance, torch.Tensor) else l_abundance,
            'l_domain': l_domain.item() if isinstance(l_domain, torch.Tensor) else l_domain,
            'l_sr': l_sr.item() if isinstance(l_sr, torch.Tensor) else l_sr,
            's_data': self.s_data.item(),
            's_physics': self.s_physics.item(),
            's_reg': self.s_reg.item(),
            's_abundance': self.s_abundance.item(),
            's_domain': self.s_domain.item(),
            's_sr': self.s_sr.item(),
        }
