import torch
import numpy as np

def compute_reliability_diagram(predicted_means, predicted_variances, true_abundances, num_bins=10):
    """
    Computes calibration metrics (Empirical Coverage vs. Expected Confidence) for regression.
    
    Args:
        predicted_means: (N,) tensor of predicted abundance means
        predicted_variances: (N,) tensor of predicted total variances (epistemic + aleatoric)
        true_abundances: (N,) tensor of ground truth abundances
        num_bins: number of confidence intervals to evaluate
        
    Returns:
        expected_confidences: array of expected coverage probabilities
        empirical_coverages: array of actual coverages
        ece: Expected Calibration Error
    """
    predicted_means = predicted_means.detach().cpu().numpy().flatten()
    predicted_std = np.sqrt(predicted_variances.detach().cpu().numpy().flatten())
    true_abundances = true_abundances.detach().cpu().numpy().flatten()
    
    # We will test confidence intervals corresponding to standard normal quantiles.
    # For a Gaussian distribution, the probability of falling within +/- z * std is known.
    # We can evaluate this for z ranging from 0 to 3.
    z_scores = np.linspace(0.1, 3.0, num_bins)
    
    from scipy.stats import norm
    expected_confidences = 2 * norm.cdf(z_scores) - 1.0
    
    empirical_coverages = []
    
    for z in z_scores:
        lower_bound = predicted_means - z * predicted_std
        upper_bound = predicted_means + z * predicted_std
        
        # Check how many true values fall within the bounds
        in_bound = (true_abundances >= lower_bound) & (true_abundances <= upper_bound)
        coverage = np.mean(in_bound)
        empirical_coverages.append(coverage)
        
    empirical_coverages = np.array(empirical_coverages)
    
    # Expected Calibration Error (ECE) for regression
    ece = np.mean(np.abs(empirical_coverages - expected_confidences))
    
    return expected_confidences, empirical_coverages, ece

def calibrate_ensemble(ensemble, dataloader):
    """
    Evaluates the ensemble on a held-out dataset and computes ECE.
    """
    device = next(ensemble.parameters()).device
    ensemble.eval()
    
    all_means = []
    all_vars = []
    all_truths = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, _, _, _, _, _, obs_abund, mask_abund = [
                b.to(device) if b is not None else None for b in batch
            ]
            
            if obs_abund is None or mask_abund is None:
                continue
                
            # Filter batch to only those patches that have at least some labeled pixels
            # For simplicity, we just run the whole batch and mask later
            mean_abundances, epistemic_unc, aleatoric_unc = ensemble.predict_with_uncertainty(x)
            
            # Total variance is epistemic + aleatoric
            total_variance = epistemic_unc + aleatoric_unc
            
            # Mask to labeled pixels
            mask_flat = mask_abund.view(-1)
            mean_flat = mean_abundances.permute(0, 2, 3, 1).reshape(-1, mean_abundances.shape[1])[mask_flat]
            var_flat = total_variance.permute(0, 2, 3, 1).reshape(-1, total_variance.shape[1])[mask_flat]
            obs_flat = obs_abund.permute(0, 2, 3, 1).reshape(-1, obs_abund.shape[1])[mask_flat]
            
            all_means.append(mean_flat)
            all_vars.append(var_flat)
            all_truths.append(obs_flat)
            
    if len(all_means) == 0:
        return None, None, None
        
    all_means = torch.cat(all_means, dim=0)
    all_vars = torch.cat(all_vars, dim=0)
    all_truths = torch.cat(all_truths, dim=0)
    
    exp_conf, emp_cov, ece = compute_reliability_diagram(all_means, all_vars, all_truths)
    return exp_conf, emp_cov, ece
