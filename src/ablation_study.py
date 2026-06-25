import torch
from hybrid_model import HybridSpectralSpatialModel
from composite_loss import CompositeLoss
from trainer import PINNTrainer
from torch.utils.data import DataLoader, TensorDataset
from validation import compute_rmse, compute_spectral_angle, compute_mae_per_endmember
import numpy as np

def run_ablation(config_name, use_physics_loss=True, use_uncertainty_reg=True, use_domain_adapt=True):
    print(f"\n{'='*50}\nRunning Ablation: {config_name}\n{'='*50}")
    
    device = "cpu" # Use CPU for quick ablation testing
    num_bands = 40
    num_endmembers = 5
    
    # Mock data
    X_train = torch.rand(100, num_bands, 32, 32)
    Y_abund = torch.rand(100, num_endmembers, 32, 32)
    Y_abund = Y_abund / Y_abund.sum(dim=1, keepdim=True)
    mu0 = torch.rand(100)
    mu = torch.rand(100)
    phase = torch.rand(100)
    domain_labels = torch.randint(0, 2, (100,))
    
    dataset = TensorDataset(X_train, Y_abund, mu0, mu, phase, domain_labels)
    # create dictionaries for trainer
    def collate_fn(batch):
        return {
            'reflectance': torch.stack([b[0] for b in batch]),
            'abundances': torch.stack([b[1] for b in batch]),
            'mu0': torch.stack([b[2] for b in batch]),
            'mu': torch.stack([b[3] for b in batch]),
            'phase': torch.stack([b[4] for b in batch]),
            'domain_labels': torch.stack([b[5] for b in batch]),
        }
        
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
    
    # Model & Loss
    model = HybridSpectralSpatialModel(num_bands=num_bands, num_endmembers=num_endmembers)
    endmember_ssas = torch.rand(num_endmembers, num_bands)
    
    comp_loss = CompositeLoss(
        endmember_ssas, 
        use_physics_loss=use_physics_loss, 
        use_uncertainty_reg=use_uncertainty_reg, 
        use_domain_adapt=use_domain_adapt
    )
    
    trainer = PINNTrainer(model, comp_loss, device=device)
    
    # Skip pretraining if no physics loss
    pretrain_epochs = 2 if use_physics_loss else 0
    total_epochs = 4
    
    trainer.train(loader, loader, total_epochs=total_epochs, pretrain_epochs=pretrain_epochs, checkpoint_dir=f"checkpoints_ablation_{config_name}")
    
    # Validation (evaluate on training data for simplicity in this script)
    model.eval()
    with torch.no_grad():
        preds, ssa, log_var, abund_log_var, _, _, _ = model(X_train.to(device))
        
        # Calculate metrics
        rmse = compute_rmse(preds, Y_abund.to(device)) # Here just testing if code runs, conceptually predicted reflectance vs observed, but we only have abundances.
        
        # Mock predicted reflectance
        mock_pred_ref = torch.rand_like(X_train)
        rmse_val = compute_rmse(mock_pred_ref, X_train)
        sam_val = compute_spectral_angle(mock_pred_ref.permute(0, 2, 3, 1).reshape(-1, num_bands), X_train.permute(0, 2, 3, 1).reshape(-1, num_bands))
        mae_val = compute_mae_per_endmember(preds.permute(0, 2, 3, 1).reshape(-1, num_endmembers), Y_abund.permute(0, 2, 3, 1).reshape(-1, num_endmembers))
        
        print(f"\nMetrics for {config_name}:")
        print(f"RMSE: {rmse_val:.4f}")
        print(f"SAM: {sam_val:.4f}")
        print(f"MAE (mean across endmembers): {np.mean(mae_val):.4f}")
        return rmse_val, sam_val, mae_val

if __name__ == "__main__":
    configs = [
        {"name": "Full_Model", "use_physics_loss": True, "use_uncertainty_reg": True, "use_domain_adapt": True},
        {"name": "No_Physics_Loss", "use_physics_loss": False, "use_uncertainty_reg": True, "use_domain_adapt": True},
        {"name": "No_Uncertainty_Reg", "use_physics_loss": True, "use_uncertainty_reg": False, "use_domain_adapt": True},
        {"name": "No_Domain_Adapt", "use_physics_loss": True, "use_uncertainty_reg": True, "use_domain_adapt": False},
    ]
    
    results = {}
    for cfg in configs:
        metrics = run_ablation(cfg["name"], cfg["use_physics_loss"], cfg["use_uncertainty_reg"], cfg["use_domain_adapt"])
        results[cfg["name"]] = metrics
        
    print("\n" + "="*50)
    print("Ablation Study Summary")
    print("="*50)
    for name, (rmse, sam, mae) in results.items():
        print(f"{name:20s} | RMSE: {rmse:.4f} | SAM: {sam:.4f} | MAE(avg): {np.mean(mae):.4f}")
