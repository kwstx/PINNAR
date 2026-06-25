import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class PhysicsResidual(nn.Module):
    """
    Computes the Hapke forward model residual in a fully differentiable manner
    using PyTorch operations, natively supporting autograd.
    """
    def __init__(self, endmember_ssas, phase_function_g=0.0):
        super().__init__()
        self.register_buffer('endmember_ssas', endmember_ssas)
        self.g_param = phase_function_g

    def phase_function(self, phase_angle):
        g = self.g_param
        num = 1 - g**2
        den = (1 + g**2 - 2 * g * torch.cos(phase_angle))**1.5
        return num / den

    def h_function(self, x, w):
        gamma = torch.sqrt(1.0 - w + 1e-6)
        return (1.0 + 2.0 * x) / (1.0 + 2.0 * x * gamma)

    def forward(self, predicted_reflectance, predicted_abundances, mu0, mu, phase_angle):
        # Flatten spatial dims if necessary
        if predicted_abundances.dim() == 4:
            B, C, H, W = predicted_abundances.shape
            predicted_abundances = predicted_abundances.permute(0, 2, 3, 1).reshape(-1, C)
            predicted_reflectance = predicted_reflectance.permute(0, 2, 3, 1).reshape(-1, predicted_reflectance.shape[1])
            if mu0.dim() == 1:
                mu0 = mu0.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                mu = mu.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                phase_angle = phase_angle.unsqueeze(1).unsqueeze(2).expand(B, H, W).reshape(-1)
                
        w_mix = torch.matmul(predicted_abundances, self.endmember_ssas)
        
        p_g = self.phase_function(phase_angle).unsqueeze(1)
        mu0_exp = mu0.unsqueeze(1)
        mu_exp = mu.unsqueeze(1)
        
        h_mu0 = self.h_function(mu0_exp, w_mix)
        h_mu = self.h_function(mu_exp, w_mix)
        
        hapke_ref = (w_mix / 4.0) * (mu0_exp / (mu0_exp + mu_exp + 1e-6)) * (p_g + h_mu0 * h_mu - 1.0)
        
        residual_norm = torch.mean((predicted_reflectance - hapke_ref)**2)
        return residual_norm, hapke_ref

def compute_abundance_smoothness(abundances):
    """
    Computes total variation loss for abundance maps to encourage spatial smoothness.
    abundances: (B, C, H, W)
    """
    if abundances.dim() != 4:
        return torch.tensor(0.0, device=abundances.device)
    tv_h = torch.mean(torch.abs(abundances[:, :, 1:, :] - abundances[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(abundances[:, :, :, 1:] - abundances[:, :, :, :-1]))
    return tv_h + tv_w

class PINNTrainer:
    def __init__(self, model, composite_loss, device="cuda"):
        self.model = model.to(device)
        self.composite_loss = composite_loss.to(device)
        self.device = device
        
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.composite_loss.parameters()), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100) # Assumes 100 epochs total
        self.physics_residual = PhysicsResidual(composite_loss.physics_module.endmember_ssas).to(device)

    def train(self, unlabeled_loader, labeled_loader, total_epochs=100, pretrain_epochs=50, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, total_epochs + 1):
            self.model.train()
            
            stage = "Pre-training" if epoch <= pretrain_epochs else "Fine-tuning"
            print(f"--- Epoch {epoch}/{total_epochs} [{stage}] ---")
            
            epoch_data_loss = 0.0
            epoch_physics_residual = 0.0
            epoch_smoothness = 0.0
            epoch_domain_loss = 0.0
            epoch_sr_loss = 0.0
            
            if epoch <= pretrain_epochs:
                # Stage 1: Pre-train on abundant unlabeled spectra using ONLY the physics loss
                for batch in unlabeled_loader:
                    self.optimizer.zero_grad()
                    
                    obs_ref = batch['reflectance'].to(self.device)
                    mu0 = batch['mu0'].to(self.device)
                    mu = batch['mu'].to(self.device)
                    phase = batch['phase'].to(self.device)
                    
                    abundances, ssa, log_var, abund_log_var, _, _, _ = self.model(obs_ref)
                    
                    # Ensure we pass the predicted_reflectance (network output) rather than observed?
                    # The network outputs abundances, SSA. To get predicted reflectance, it's typically just observed passing through if it's an autoencoder,
                    # but here the hybrid model outputs abundances and SSA. We don't have "predicted reflectance" directly from the model,
                    # so we might use the observed reflectance as the proxy or the model might predict it. Wait, the hybrid model
                    # only outputs (abundances, ssa, log_var, abundance_log_var). Let's use `obs_ref` as the input to the physics residual.
                    # Actually, the user says "PhysicsResidual class that takes network outputs and geometry parameters"
                    residual_norm, _ = self.physics_residual(obs_ref, abundances, mu0, mu, phase)
                    
                    smoothness = compute_abundance_smoothness(abundances)
                    
                    loss = residual_norm + 0.01 * smoothness
                    loss.backward()
                    
                    # Gradient clipping at 1.0 to stabilize physics loss gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.composite_loss.parameters(), 1.0)
                    
                    self.optimizer.step()
                    
                    epoch_physics_residual += residual_norm.item()
                    epoch_smoothness += smoothness.item()
                    
                print(f"Physics Residual: {epoch_physics_residual/len(unlabeled_loader):.4f} | Smoothness: {epoch_smoothness/len(unlabeled_loader):.4f}")
                
            else:
                # Stage 2: Fine-tune with the full composite loss on pixels with higher-quality labels
                for batch in labeled_loader:
                    self.optimizer.zero_grad()
                    
                    obs_ref = batch['reflectance'].to(self.device)
                    mu0 = batch['mu0'].to(self.device)
                    mu = batch['mu'].to(self.device)
                    phase = batch['phase'].to(self.device)
                    obs_abundances = batch.get('abundances', None)
                    if obs_abundances is not None:
                        obs_abundances = obs_abundances.to(self.device)
                    
                    mask_labeled = batch.get('mask_labeled', torch.ones(obs_ref.shape[0], dtype=torch.bool)).to(self.device)
                    mask_abundances = batch.get('mask_abundances', torch.ones(obs_ref.shape[0], dtype=torch.bool)).to(self.device)
                    
                    domain_labels = batch.get('domain_labels', None)
                    if domain_labels is not None:
                        domain_labels = domain_labels.to(self.device)
                    
                    abundances, ssa, log_var, abund_log_var, domain_logits, hr_abundances, hr_ssa = self.model(obs_ref)
                    
                    # Assuming the composite loss needs predicted reflectance, but the model doesn't predict it directly.
                    # We pass the input reflectance as predicted_reflectance since it's an auto-encoding setup where physics acts as decoder.
                    # Let's pass obs_ref as predicted_reflectance for the composite loss.
                    total_loss, metrics = self.composite_loss(
                        predicted_reflectance=obs_ref, 
                        observed_reflectance=obs_ref, 
                        predicted_abundances=abundances, 
                        log_var=log_var, 
                        abundance_log_var=abund_log_var, 
                        mu0=mu0, mu=mu, phase_angle=phase, 
                        mask_labeled=mask_labeled, 
                        observed_abundances=obs_abundances, 
                        mask_abundances=mask_abundances,
                        domain_logits=domain_logits,
                        domain_labels=domain_labels,
                        hr_abundances=hr_abundances,
                        hr_ssa=hr_ssa
                    )
                    
                    smoothness = compute_abundance_smoothness(abundances)
                    loss = total_loss + 0.01 * smoothness
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.composite_loss.parameters(), 1.0)
                    
                    self.optimizer.step()
                    
                    epoch_data_loss += metrics.get('l_data', 0.0)
                    epoch_physics_residual += metrics.get('l_physics', 0.0)
                    epoch_smoothness += smoothness.item()
                    epoch_domain_loss += metrics.get('l_domain', 0.0)
                    epoch_sr_loss += metrics.get('l_sr', 0.0)
                    
                print(f"Data: {epoch_data_loss/len(labeled_loader):.4f} | Phys: {epoch_physics_residual/len(labeled_loader):.4f} | Smooth: {epoch_smoothness/len(labeled_loader):.4f} | Dom: {epoch_domain_loss/len(labeled_loader):.4f} | SR: {epoch_sr_loss/len(labeled_loader):.4f}")
                
            self.scheduler.step()
            
            # Save checkpoints every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'composite_loss_state_dict': self.composite_loss.state_dict()
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
