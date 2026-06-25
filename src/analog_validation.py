import torch
import numpy as np
from validation import compute_rmse, compute_spectral_angle, compute_mae_per_endmember

class AnalogValidator:
    """
    Validates model predictions against ground-truth in-situ mineralogy 
    (Apollo soil samples for the Moon or Curiosity/Perseverance for Mars).
    Uses nearest-neighbor pixel matching.
    """
    def __init__(self, ground_truth_coords, ground_truth_abundances, ground_truth_spectra):
        """
        ground_truth_coords: (N, 2) array of lat/lon or local coordinates
        ground_truth_abundances: (N, num_endmembers)
        ground_truth_spectra: (N, num_bands)
        """
        self.gt_coords = ground_truth_coords
        self.gt_abundances = ground_truth_abundances
        self.gt_spectra = ground_truth_spectra
        
    def validate(self, model_coords, predicted_abundances, predicted_spectra):
        """
        Matches each ground truth location to the nearest model pixel and computes metrics.
        model_coords: (M, 2) array of coordinates for model pixels
        predicted_abundances: (M, num_endmembers)
        predicted_spectra: (M, num_bands)
        """
        matched_pred_abundances = []
        matched_pred_spectra = []
        
        # Nearest neighbor matching
        for i, gt_coord in enumerate(self.gt_coords):
            distances = np.linalg.norm(model_coords - gt_coord, axis=1)
            nearest_idx = np.argmin(distances)
            
            matched_pred_abundances.append(predicted_abundances[nearest_idx])
            matched_pred_spectra.append(predicted_spectra[nearest_idx])
            
        matched_pred_abundances = torch.tensor(np.array(matched_pred_abundances), dtype=torch.float32)
        matched_pred_spectra = torch.tensor(np.array(matched_pred_spectra), dtype=torch.float32)
        
        gt_abundances_tensor = torch.tensor(self.gt_abundances, dtype=torch.float32)
        gt_spectra_tensor = torch.tensor(self.gt_spectra, dtype=torch.float32)
        
        rmse = compute_rmse(matched_pred_spectra, gt_spectra_tensor)
        sam = compute_spectral_angle(matched_pred_spectra, gt_spectra_tensor)
        mae_per_endmember = compute_mae_per_endmember(matched_pred_abundances, gt_abundances_tensor)
        
        return {
            "rmse": rmse,
            "sam": sam,
            "mae_per_endmember": mae_per_endmember,
            "matched_pairs": len(self.gt_coords)
        }

if __name__ == "__main__":
    print("Testing Analog Validation...")
    # Mock data
    gt_coords = np.random.rand(10, 2) * 100
    gt_abundances = np.random.rand(10, 5)
    gt_abundances = gt_abundances / np.sum(gt_abundances, axis=1, keepdims=True)
    gt_spectra = np.random.rand(10, 50)
    
    model_coords = np.random.rand(1000, 2) * 100
    predicted_abundances = np.random.rand(1000, 5)
    predicted_abundances = predicted_abundances / np.sum(predicted_abundances, axis=1, keepdims=True)
    predicted_spectra = np.random.rand(1000, 50)
    
    validator = AnalogValidator(gt_coords, gt_abundances, gt_spectra)
    metrics = validator.validate(model_coords, predicted_abundances, predicted_spectra)
    
    print(f"Analog Validation Metrics:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"SAM: {metrics['sam']:.4f}")
    print(f"MAE per Endmember: {metrics['mae_per_endmember']}")
