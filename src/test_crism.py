import torch
import numpy as np
from crism_preprocessing import CRISMPipeline

def main():
    print("Testing CRISM Preprocessing Pipeline...")
    
    # Mock data dimensions
    BANDS = 50
    LINES = 20
    SAMPLES = 20
    
    # Mock wavelengths (0.4 to 4.0 µm)
    wavelengths = torch.linspace(0.4, 4.0, BANDS)
    
    # Mock atmospheric transmission
    transmission = torch.ones(BANDS)
    # Add a pseudo absorption feature at 2.0 µm
    transmission[BANDS // 2] = 0.5 
    
    # Mock data cube
    cube = torch.rand((BANDS, LINES, SAMPLES)) + 1.0 # Avoid zero
    
    # Initialize pipeline
    # We want 10 components, and a spatial context of 5 (5x5 = 25)
    pipeline = CRISMPipeline(
        wavelengths=wavelengths,
        atmospheric_transmission=transmission,
        spatial_context_size=5,
        pca_components=10
    )
    
    # Move to GPU if available for a test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cube = cube.to(device)
    pipeline = pipeline.to(device)
    
    print(f"Input shape: {cube.shape}")
    
    # Run pipeline
    with torch.no_grad():
        out = pipeline(cube)
        
    print(f"Output shape: {out.shape}")
    
    N_pixels = LINES * SAMPLES
    expected_shape = (N_pixels, 10, 25)
    print(f"Expected shape: {expected_shape}")
    
    if out.shape == expected_shape:
        print("Test Passed: Output shape matches expected shape.")
    else:
        print("Test Failed: Output shape mismatch.")

if __name__ == "__main__":
    main()
