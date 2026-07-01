import os
import argparse
import sys
from src.sensitivity_analysis import run_sobol_analysis, run_hyperparameter_sweep
from src.trainer import PINNTrainer
from src.hybrid_model import HybridSpectralSpatialModel
import torch

from torch.utils.data import DataLoader
from src.dataset import CRISMDataset, LabeledCRISMDataset
from src.composite_loss import CompositeLoss
from src.physics_loss import HapkePhysicsLoss

def train_model(args):
    print("Starting training pipeline...")
    
    # Check if train_ids.txt exists, else use dummy FRS0005AA3B
    if os.path.exists('train_ids.txt'):
        with open('train_ids.txt', 'r') as f:
            train_ids = [line.strip() for line in f.readlines() if line.strip()]
    else:
        train_ids = ['FRS0005AA3B']
        
    print(f"Loaded {len(train_ids)} product IDs for training.")
    
    # Initialize DataLoaders
    unlabeled_dataset = CRISMDataset(product_ids=train_ids)
    labeled_dataset = LabeledCRISMDataset(product_ids=train_ids, labels_csv='labels.csv')
    
    # Set batch_size=1 since our patches are large (32x32)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True)
    labeled_loader = DataLoader(labeled_dataset, batch_size=1, shuffle=True)
    
    # Get dynamic number of bands from the actual parsed data
    sample_item = unlabeled_dataset[0]
    num_bands = sample_item['reflectance'].shape[0]
    num_endmembers = 5
    print(f"Initialized model with {num_bands} bands.")
    model = HybridSpectralSpatialModel(num_bands=num_bands, num_endmembers=num_endmembers)
    
    # Initialize random endmember SSAs (simulate lab data)
    endmember_ssas = torch.rand(num_endmembers, num_bands)
    composite_loss = CompositeLoss(endmember_ssas)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.stage in ['pretrain', 'finetune', 'both']:
        # We use PINNTrainer from trainer.py
        from src.trainer import PINNTrainer
        trainer = PINNTrainer(model, composite_loss, device=device)
        
        pretrain_epochs = 2 if args.stage in ['pretrain', 'both'] else 0
        total_epochs = 4 if args.stage in ['finetune', 'both'] else pretrain_epochs
        
        print(f"Running on device: {device}")
        trainer.train(
            unlabeled_loader=unlabeled_loader, 
            labeled_loader=labeled_loader, 
            total_epochs=total_epochs, 
            pretrain_epochs=pretrain_epochs
        )
        
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
