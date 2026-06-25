import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MockOverlappingDataset(Dataset):
    """
    Mock dataset generating overlapping orbital tracks from two different instruments.
    Returns synchronized patches of the same region observed by two different sensors.
    """
    def __init__(self, instrument_1="CRISM", bands_1=50, instrument_2="M3", bands_2=85, size=100, spatial_size=32):
        self.size = size
        self.inst1 = instrument_1
        self.bands1 = bands_1
        self.inst2 = instrument_2
        self.bands2 = bands_2
        self.spatial_size = spatial_size
        
        # Ground truth abundances (shared across instruments for the overlap)
        self.num_endmembers = 5
        self.gt_abundances = torch.rand(size, self.num_endmembers, spatial_size, spatial_size)
        self.gt_abundances = self.gt_abundances / self.gt_abundances.sum(dim=1, keepdim=True)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random reflectance based loosely on ground truth, with instrument-specific noise/bands
        ref_1 = torch.rand(self.bands1, self.spatial_size, self.spatial_size) * 0.5 + 0.1
        ref_2 = torch.rand(self.bands2, self.spatial_size, self.spatial_size) * 0.5 + 0.1
        
        # Geometry (mocked)
        mu0 = torch.tensor(0.8)
        mu = torch.tensor(0.9)
        phase = torch.tensor(0.3)
        
        return {
            "inst1_id": self.inst1,
            "ref1": ref_1,
            "inst2_id": self.inst2,
            "ref2": ref_2,
            "mu0": mu0,
            "mu": mu,
            "phase": phase,
            "gt_abundances": self.gt_abundances[idx]
        }

def get_mock_overlapping_dataloader(batch_size=8):
    dataset = MockOverlappingDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def validate_cross_mission_consistency(model, dataloader, physics_residual_fn, device="cuda"):
    """
    Validates cross-mission consistency by comparing abundance maps and physics residuals
    for regions observed by multiple instruments.
    """
    model.eval()
    
    total_abundance_diff = 0.0
    total_physics_diff = 0.0
    num_batches = 0
    
    unified_products = []
    
    with torch.no_grad():
        for batch in dataloader:
            inst1_id = batch["inst1_id"][0] # Assuming uniform batch
            inst2_id = batch["inst2_id"][0]
            
            ref1 = batch["ref1"].to(device)
            ref2 = batch["ref2"].to(device)
            
            mu0 = batch["mu0"].to(device)
            mu = batch["mu"].to(device)
            phase = batch["phase"].to(device)
            
            # Forward pass instrument 1
            abund1, ssa1, log_var1, abund_log_var1, _, _, _ = model(ref1, instrument_id=inst1_id)
            
            # Forward pass instrument 2
            abund2, ssa2, log_var2, abund_log_var2, _, _, _ = model(ref2, instrument_id=inst2_id)
            
            # Consistency Metrics
            abundance_diff = torch.mean((abund1 - abund2)**2).item()
            total_abundance_diff += abundance_diff
            
            # Physics Residuals (using the harmonized reflectance as the target for the physics model)
            # The harmonizer maps to a shared band space. For the physics residual, we need 
            # to know the shared band reflectances. Since the user asked to compute physics residual 
            # on overlapping tracks, we can compute it on the unified predictions.
            
            # For this step, we just use the predicted reflectance proxy from the model or mock it
            # The physics residual needs the actual shared reflectance.
            # To strictly follow the pipeline, we get the harmonized reflectances
            harm_ref1 = model.harmonizer(ref1, inst1_id)
            harm_ref2 = model.harmonizer(ref2, inst2_id)
            
            # Reshape for physics residual if needed: (B, C, H, W)
            # PhysicsResidual expects (predicted_reflectance, abundances, mu0, mu, phase)
            res_norm1, _ = physics_residual_fn(harm_ref1, abund1, mu0, mu, phase)
            res_norm2, _ = physics_residual_fn(harm_ref2, abund2, mu0, mu, phase)
            
            physics_diff = torch.abs(res_norm1 - res_norm2).mean().item()
            total_physics_diff += physics_diff
            
            # Create unified product (e.g., averaging the two)
            unified_abund = (abund1 + abund2) / 2.0
            
            # Convert log_var to var, average, and back to log_var (or just average variances)
            var1 = torch.exp(abund_log_var1)
            var2 = torch.exp(abund_log_var2)
            unified_unc = (var1 + var2) / 2.0
            
            unified_products.append({
                "unified_abundance": unified_abund.cpu(),
                "unified_uncertainty": unified_unc.cpu()
            })
            
            num_batches += 1
            
    avg_abundance_diff = total_abundance_diff / num_batches
    avg_physics_diff = total_physics_diff / num_batches
    
    print(f"Cross-Mission Validation - Avg Abundance MSE: {avg_abundance_diff:.6f}")
    print(f"Cross-Mission Validation - Avg Physics Residual Diff: {avg_physics_diff:.6f}")
    
    return unified_products, avg_abundance_diff, avg_physics_diff

if __name__ == "__main__":
    # Simple test
    from harmonization import MultiMissionPINN
    from trainer import PhysicsResidual
    
    print("Testing Validation Pipeline...")
    device = "cpu"
    model = MultiMissionPINN(
        instrument_bands_dict={"CRISM": 50, "M3": 85}, 
        shared_bands=40, 
        num_endmembers=5
    ).to(device)
    
    dataloader = get_mock_overlapping_dataloader(batch_size=2)
    
    # Mock endmember SSAs (N_endmembers, N_bands)
    mock_ssas = torch.rand(5, 40).to(device)
    physics_res = PhysicsResidual(mock_ssas).to(device)
    
    unified_products, _, _ = validate_cross_mission_consistency(model, dataloader, physics_res, device=device)
    print(f"Generated {len(unified_products)} batches of unified products ready for mosaicking.")
