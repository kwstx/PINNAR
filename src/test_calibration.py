import torch
import numpy as np
from calibration import compute_reliability_diagram

def test_compute_reliability_diagram():
    # Simulate a perfectly calibrated model for abundance predictions
    N = 1000
    
    # Ground truth
    true_abundances = torch.rand(N)
    
    # A perfectly calibrated model would predict a mean around the true value with a correct variance
    # Let's say it predicts true value + some noise, and estimates that noise variance accurately
    noise_std = 0.1
    predicted_means = true_abundances + torch.randn(N) * noise_std
    predicted_variances = torch.ones(N) * (noise_std ** 2)
    
    exp_conf, emp_cov, ece = compute_reliability_diagram(predicted_means, predicted_variances, true_abundances, num_bins=10)
    
    # Check that ECE is small (since it's roughly calibrated)
    assert ece < 0.05, f"Expected small ECE for a calibrated model, got {ece}"
    
    # Check monotonicity of expected confidence
    assert np.all(np.diff(exp_conf) > 0), "Expected confidence should be monotonically increasing"
    
if __name__ == "__main__":
    test_compute_reliability_diagram()
