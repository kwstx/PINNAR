import torch
import torch.nn as nn
from hybrid_model import HybridSpectralSpatialModel

class CalibrationHead(nn.Module):
    """
    Non-linear Multi-Layer Perceptron mapping instrument-specific 
    spectral bands to a shared band space.
    """
    def __init__(self, in_bands, shared_bands, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_bands, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, shared_bands, kernel_size=1)
        )
        
    def forward(self, x):
        return self.mlp(x)

class GeometricAligner(nn.Module):
    """
    Aligns geometric parameters from different missions to a standard 
    format (e.g. standardizing angles to radians, computing cosines).
    This ensures the physics model receives correctly formatted geometry.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, incidence, emission, phase, is_degrees=True):
        if is_degrees:
            incidence = torch.deg2rad(incidence)
            emission = torch.deg2rad(emission)
            phase = torch.deg2rad(phase)
            
        mu0 = torch.cos(incidence)
        mu = torch.cos(emission)
        
        # Clamp to avoid numerical issues
        mu0 = torch.clamp(mu0, min=1e-6, max=1.0)
        mu = torch.clamp(mu, min=1e-6, max=1.0)
        
        return mu0, mu, phase

class MultiMissionHarmonizer(nn.Module):
    """
    Routes input cubes to their corresponding calibration heads based on instrument_id.
    """
    def __init__(self, instrument_bands_dict, shared_bands, hidden_dim=64):
        """
        instrument_bands_dict: dict mapping instrument_id (str) to number of native bands (int)
        shared_bands: int, number of bands in the unified space
        """
        super().__init__()
        self.heads = nn.ModuleDict({
            inst_id: CalibrationHead(bands, shared_bands, hidden_dim)
            for inst_id, bands in instrument_bands_dict.items()
        })
        self.geometric_aligner = GeometricAligner()
        
    def forward(self, x, instrument_id):
        """
        x: (B, C_in, H, W)
        instrument_id: str, name of the instrument
        """
        if instrument_id not in self.heads:
            raise ValueError(f"Unknown instrument ID: {instrument_id}. Available: {list(self.heads.keys())}")
            
        return self.heads[instrument_id](x)

class MultiMissionPINN(nn.Module):
    """
    Wrapper encapsulating the Harmonizer and the HybridSpectralSpatialModel.
    """
    def __init__(self, instrument_bands_dict, shared_bands, num_endmembers, **kwargs):
        super().__init__()
        self.harmonizer = MultiMissionHarmonizer(instrument_bands_dict, shared_bands)
        self.pinn_encoder = HybridSpectralSpatialModel(num_bands=shared_bands, num_endmembers=num_endmembers, **kwargs)
        
    def forward(self, x, instrument_id):
        """
        Harmonize spectral response, then encode.
        Returns: abundances, ssa, log_var, abundance_log_var, domain_logits, hr_abundances, hr_ssa
        """
        # 1. Harmonize to shared band space
        harmonized_x = self.harmonizer(x, instrument_id)
        
        # 2. Forward through common PINN encoder
        return self.pinn_encoder(harmonized_x)
