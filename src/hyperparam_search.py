import optuna
import torch
import torch.optim as optim
from hybrid_model import HybridSpectralSpatialModel
from composite_loss import CompositeLoss

def objective(trial, val_data, endmember_ssas, num_bands, num_endmembers):
    """
    Optuna objective function for short hyperparameter search 
    to find the optimal initial homoscedastic task weights.
    """
    # Sample initial values for the learnable weights
    init_s_data = trial.suggest_float("init_s_data", -3.0, 3.0)
    init_s_physics = trial.suggest_float("init_s_physics", -3.0, 3.0)
    init_s_reg = trial.suggest_float("init_s_reg", -3.0, 3.0)
    
    # Initialize Model and Loss
    model = HybridSpectralSpatialModel(num_bands=num_bands, num_endmembers=num_endmembers)
    loss_fn = CompositeLoss(
        endmember_ssas=endmember_ssas,
        init_s_data=init_s_data,
        init_s_physics=init_s_physics,
        init_s_reg=init_s_reg
    )
    
    # Freeze model parameters to quickly evaluate the loss function's impact
    # on just finding the optimal balance (or do a few steps of training)
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': loss_fn.parameters(), 'lr': 1e-2}
    ])
    
    model.train()
    loss_fn.train()
    
    # Unpack validation subset (mocked as a single batch for the short search)
    x, y_true, mu0, mu, phase_angle, mask_labeled = val_data
    
    # Short training loop to see if the loss converges well
    for step in range(5):
        optimizer.zero_grad()
        abundances, ssa, log_var = model(x)
        
        # In a real scenario, predicted_reflectance would be computed
        # For this demonstration, we mock predicted_reflectance with model output transformations
        predicted_reflectance = ssa.mean(dim=(-1, -2)) # Mock operation
        observed_reflectance = y_true.mean(dim=(-1, -2)) # Mock operation
        
        # Reshape abundances and log_var to match loss signature
        predicted_abundances = abundances.mean(dim=(-1, -2))
        log_var_flat = log_var.mean(dim=(-1, -2))
        
        loss, loss_dict = loss_fn(
            predicted_reflectance,
            observed_reflectance,
            predicted_abundances,
            log_var_flat,
            mu0,
            mu,
            phase_angle,
            mask_labeled=mask_labeled
        )
        
        loss.backward()
        optimizer.step()
        
    # Evaluate on the validation metric we care about most (e.g., pure L_data + L_physics without homoscedastic weights)
    # The goal is to find initial S values that minimize the unweighted physical and data errors
    model.eval()
    loss_fn.eval()
    with torch.no_grad():
        abundances, ssa, log_var = model(x)
        
        predicted_reflectance = ssa.mean(dim=(-1, -2))
        observed_reflectance = y_true.mean(dim=(-1, -2))
        predicted_abundances = abundances.mean(dim=(-1, -2))
        log_var_flat = log_var.mean(dim=(-1, -2))
        
        _, final_loss_dict = loss_fn(
            predicted_reflectance,
            observed_reflectance,
            predicted_abundances,
            log_var_flat,
            mu0,
            mu,
            phase_angle,
            mask_labeled=mask_labeled
        )
    
    # Return the unweighted sum of components to Optuna to minimize
    validation_score = final_loss_dict['l_data'] + final_loss_dict['l_physics']
    return validation_score


def run_hyperparam_search(val_data, endmember_ssas, num_bands, num_endmembers, n_trials=20):
    """
    Run Optuna study to find the best initial homoscedastic weights.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, val_data, endmember_ssas, num_bands, num_endmembers),
        n_trials=n_trials
    )
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return trial.params

if __name__ == "__main__":
    # Mock data for testing the search
    B, num_bands, num_endmembers = 4, 10, 5
    H, W = 32, 32
    
    val_data = (
        torch.randn(B, num_bands, H, W), # x
        torch.randn(B, num_bands, H, W), # y_true
        torch.ones(B), # mu0
        torch.ones(B), # mu
        torch.ones(B) * 0.5, # phase_angle
        torch.ones(B, dtype=torch.bool) # mask_labeled
    )
    
    endmember_ssas = torch.rand(num_endmembers, num_bands)
    
    best_params = run_hyperparam_search(val_data, endmember_ssas, num_bands, num_endmembers, n_trials=5)
    print("Optimization Complete.")
