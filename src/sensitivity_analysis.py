import torch
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from src.physics_loss import HapkePhysicsLoss

def run_hyperparameter_sweep(train_func, data_loader, val_loader):
    """
    Mock function to systematically vary model hyperparameters.
    In a real scenario, this would train models and log results (e.g., via wandb).
    """
    print("Starting Hyperparameter Sweep...")
    results = []
    
    physics_weights = [0.1, 1.0, 10.0]
    fourier_features = [16, 32, 64]
    da_regs = [0.01, 0.1, 1.0]
    
    for w in physics_weights:
        for f in fourier_features:
            for d in da_regs:
                print(f"Training with physics_weight={w}, fourier={f}, da_reg={d}")
                # Mock training result
                val_loss = np.random.uniform(0.01, 0.1)
                results.append({
                    'physics_weight': w,
                    'fourier_features': f,
                    'da_reg': d,
                    'val_loss': val_loss
                })
    
    print("Hyperparameter sweep complete.")
    return results

def evaluate_hapke_model(params):
    """
    Evaluates the Hapke model for given SSA and g_param values.
    params: numpy array of shape (N, 2) where N is number of samples.
            col 0 is SSA, col 1 is g_param.
    """
    N = params.shape[0]
    ssas = torch.tensor(params[:, 0], dtype=torch.float32).view(N, 1) # Single band, single endmember
    g_params = torch.tensor(params[:, 1], dtype=torch.float32)
    
    # Fixed viewing geometry for the sensitivity analysis
    mu0 = torch.cos(torch.tensor(30.0 * np.pi / 180.0)).repeat(N)
    mu = torch.cos(torch.tensor(0.0 * np.pi / 180.0)).repeat(N)
    phase_angle = torch.tensor(30.0 * np.pi / 180.0).repeat(N)
    
    # Mock predicted abundances (pure endmember)
    abundances = torch.ones(N, 1)
    
    reflectances = []
    for i in range(N):
        # We instantiate a temporary physics loss object just to compute the forward reflectance
        loss_fn = HapkePhysicsLoss(endmember_ssas=ssas[i].view(1, 1), phase_function_g=g_params[i].item())
        # The Hapke physics loss internally computes the reflectance. We can call its forward pass.
        # w_mix is just ssas[i]
        # To avoid computing loss, we replicate the forward logic slightly or pass a dummy predicted_reflectance
        dummy_pred = torch.zeros(1, 1)
        _, hapke_ref = loss_fn(dummy_pred, abundances[i].view(1, 1), mu0[i].view(1), mu[i].view(1), phase_angle[i].view(1))
        reflectances.append(hapke_ref.item())
        
    return np.array(reflectances)

def run_sobol_analysis():
    """
    Performs a Global Sensitivity Analysis using Sobol indices on the physics parameters.
    Parameters varied:
    - Single-Scattering Albedo (SSA): [0.1, 0.9]
    - Asymmetry factor (g): [-0.5, 0.5] (backward to forward scattering)
    """
    print("Starting Sobol Global Sensitivity Analysis...")
    problem = {
        'num_vars': 2,
        'names': ['SSA', 'g_param'],
        'bounds': [[0.1, 0.9], [-0.5, 0.5]]
    }
    
    # Generate samples (N=1024 is a common minimum for Sobol)
    # Total samples = N * (2D + 2) -> 1024 * 6 = 6144
    param_values = saltelli.sample(problem, 1024)
    print(f"Generated {param_values.shape[0]} samples.")
    
    # Evaluate model
    Y = evaluate_hapke_model(param_values)
    
    # Calculate Sobol indices
    Si = sobol.analyze(problem, Y, print_to_console=True)
    
    print("Sobol Analysis Complete.")
    return Si

if __name__ == "__main__":
    # If run as a script, execute the Sobol analysis
    run_sobol_analysis()
