import torch
from hybrid_model import HybridSpectralSpatialModel
from composite_loss import CompositeLoss
from trainer import PINNTrainer
import shutil
import os

def test_trainer():
    # Setup dummy dimensions
    num_bands = 10
    num_endmembers = 4
    batch_size = 2
    patch_size = 32
    
    # Dummy data
    endmember_ssas = torch.rand(num_endmembers, num_bands)
    
    # Initialize model and loss
    model = HybridSpectralSpatialModel(num_bands, num_endmembers)
    composite_loss = CompositeLoss(endmember_ssas)
    
    trainer = PINNTrainer(model, composite_loss, device="cpu")
    
    # Dummy dataloaders
    class DummyLoader:
        def __init__(self, n_batches, has_labels=False):
            self.n_batches = n_batches
            self.has_labels = has_labels
            
        def __iter__(self):
            for _ in range(self.n_batches):
                batch = {
                    'reflectance': torch.rand(batch_size, num_bands, patch_size, patch_size),
                    'mu0': torch.rand(batch_size),
                    'mu': torch.rand(batch_size),
                    'phase': torch.rand(batch_size)
                }
                if self.has_labels:
                    batch['abundances'] = torch.rand(batch_size, num_endmembers, patch_size, patch_size)
                    batch['mask_labeled'] = torch.ones(batch_size * patch_size * patch_size, dtype=torch.bool)
                    batch['mask_abundances'] = torch.ones(batch_size * patch_size * patch_size, dtype=torch.bool)
                yield batch
                
        def __len__(self):
            return self.n_batches

    unlabeled_loader = DummyLoader(2, has_labels=False)
    labeled_loader = DummyLoader(2, has_labels=True)
    
    # Test Pre-training and Fine-tuning for 1 epoch each
    # Setting pretrain_epochs=1, total_epochs=2 to cover both stages quickly
    # Setting total_epochs=10 to test checkpointing
    test_ckpt_dir = "test_checkpoints"
    if os.path.exists(test_ckpt_dir):
        shutil.rmtree(test_ckpt_dir)
        
    trainer.train(unlabeled_loader, labeled_loader, total_epochs=11, pretrain_epochs=5, checkpoint_dir=test_ckpt_dir)
    
    # Check if checkpoints were saved
    ckpt_files = os.listdir(test_ckpt_dir)
    assert len(ckpt_files) == 1
    assert "checkpoint_epoch_10.pt" in ckpt_files
    print("Trainer verification passed!")
    
    shutil.rmtree(test_ckpt_dir)

if __name__ == "__main__":
    test_trainer()
