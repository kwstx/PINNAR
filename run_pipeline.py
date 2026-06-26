import argparse
import sys
from src.sensitivity_analysis import run_sobol_analysis, run_hyperparameter_sweep
from src.trainer import PreTrainer, FineTuner
from src.hybrid_model import SpatialSpectralAbundanceModel
import torch

def train_model(args):
    print("Starting training pipeline...")
    # Mock parameters
    endmember_ssas = torch.rand(5, 50)  # 5 endmembers, 50 bands
    model = SpatialSpectralAbundanceModel(num_bands=50, num_endmembers=5)
    
    if args.stage == 'pretrain' or args.stage == 'both':
        print("Running pre-training...")
        # pretrainer = PreTrainer(model, ...)
        print("Pre-training completed (mock).")
        
    if args.stage == 'finetune' or args.stage == 'both':
        print("Running fine-tuning...")
        # finetuner = FineTuner(model, ...)
        print("Fine-tuning completed (mock).")
        
    print("Training pipeline finished.")

def run_sensitivity(args):
    print("Running sensitivity analysis...")
    if args.type == 'sobol' or args.type == 'all':
        run_sobol_analysis()
    if args.type == 'sweep' or args.type == 'all':
        # Mock loaders
        run_hyperparameter_sweep(None, None, None)
    print("Sensitivity analysis finished.")

def main():
    parser = argparse.ArgumentParser(description="PINNAR: Physics-Informed Neural Network for Planetary Resource Mapping")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the PINNAR model")
    train_parser.add_argument("--stage", choices=['pretrain', 'finetune', 'both'], default='both', help="Training stage to execute")
    train_parser.add_argument("--config", type=str, help="Path to config file", default="config.yaml")
    
    # Sensitivity command
    sens_parser = subparsers.add_parser("sensitivity", help="Run sensitivity analysis")
    sens_parser.add_argument("--type", choices=['sobol', 'sweep', 'all'], default='all', help="Type of sensitivity analysis to run")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "sensitivity":
        run_sensitivity(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
