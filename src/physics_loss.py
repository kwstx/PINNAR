import torch
import torch.nn as nn

class HapkePhysicsLoss(nn.Module):
    """
    Physics-informed loss function for hyperspectral data.
    Uses the two-stream Hapke approximation and linear spectral unmixing in SSA space.
    """
    def __init__(self, endmember_ssas, phase_function_g=0.0, lambda_mse=1.0, lambda_sam=1.0):
        super().__init__()
        # endmember_ssas: Tensor of shape (N_endmembers, N_bands)
        self.register_buffer('endmember_ssas', endmember_ssas)
        self.g_param = phase_function_g
        self.lambda_mse = lambda_mse
        self.lambda_sam = lambda_sam
        self.mse = nn.MSELoss()

    def phase_function(self, phase_angle):
        """
        Henyey-Greenstein approximation.
        """
        g = self.g_param
        num = 1 - g**2
        den = (1 + g**2 - 2 * g * torch.cos(phase_angle))**1.5
        return num / den

    def h_function(self, x, w):
        """
        Chandrasekhar H-function approximation.
        """
        gamma = torch.sqrt(1.0 - w + 1e-6)  # Add epsilon for stability
        return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)

    def spectral_angle_mapper(self, predicted, target):
        """
        Computes the Spectral Angle Mapper (SAM) between predicted and target spectra.
        Both inputs have shape (Batch, N_bands).
        Returns the mean SAM across the batch.
        """
        dot_product = torch.sum(predicted * target, dim=1)
        norm_pred = torch.norm(predicted, dim=1)
        norm_target = torch.norm(target, dim=1)
        
        # Add epsilon to prevent division by zero
        cos_theta = dot_product / (norm_pred * norm_target + 1e-8)
        # Clamp to avoid NaN in arccos due to numerical instability
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-8, 1.0 - 1e-8)
        
        return torch.mean(torch.acos(cos_theta))

    def forward(self, predicted_reflectance, predicted_abundances, mu0, mu, phase_angle):
        """
        predicted_reflectance: (Batch, N_bands)
        predicted_abundances: (Batch, N_endmembers)
        mu0: (Batch,) cosine of incidence angle
        mu: (Batch,) cosine of emission angle
        phase_angle: (Batch,) phase angle in radians
        """
        # 1. Unmix in SSA space
        # w_mix shape: (Batch, N_bands)
        w_mix = torch.matmul(predicted_abundances, self.endmember_ssas)
        
        # 2. Compute Hapke Reflectance
        p_g = self.phase_function(phase_angle).unsqueeze(1) # (Batch, 1)
        mu0_exp = mu0.unsqueeze(1) # (Batch, 1)
        mu_exp = mu.unsqueeze(1) # (Batch, 1)
        
        h_mu0 = self.h_function(mu0_exp, w_mix)
        h_mu = self.h_function(mu_exp, w_mix)
        
        hapke_ref = (w_mix / 4.0) * (mu0_exp / (mu0_exp + mu_exp + 1e-6)) * (p_g + h_mu0 * h_mu - 1.0)
        
        # 3. Compute Losses
        loss_mse = self.mse(predicted_reflectance, hapke_ref)
        loss_sam = self.spectral_angle_mapper(predicted_reflectance, hapke_ref)
        
        total_loss = self.lambda_mse * loss_mse + self.lambda_sam * loss_sam
        
        return total_loss, hapke_ref
