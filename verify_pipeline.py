import numpy as np
import os
import matplotlib.pyplot as plt

from src.postprocessing import MarkovRandomFieldFilter, GeoTIFFExporter
from src.visualization import InteractiveVisualizer

def generate_synthetic_data(H=50, W=50, B=20, C=3):
    # Synthetic hyperspectral cube
    wavelengths = np.linspace(400, 2500, B)
    cube = np.random.rand(H, W, B)
    
    # Add some spatial structure to the cube
    for i in range(B):
        cube[:, :, i] += np.sin(np.linspace(0, 10, W)).reshape(1, -1) * 0.5
        
    # Synthetic abundances (should sum to 1)
    abundance = np.random.rand(H, W, C)
    
    # Create some spatial clusters for endmembers
    abundance[10:30, 10:30, 0] += 2.0
    abundance[25:45, 25:45, 1] += 2.0
    abundance[5:20, 35:45, 2] += 2.0
    
    # Normalize
    abundance = abundance / abundance.sum(axis=-1, keepdims=True)
    
    # Synthetic uncertainty
    # Higher uncertainty at the edges of clusters
    uncertainty = np.random.rand(H, W, C) * 0.1 + 0.05
    
    # Synthetic variance
    variance = uncertainty ** 2 + 0.01
    
    return cube, wavelengths, abundance, uncertainty, variance

def main():
    print("Generating synthetic data...")
    cube, wavelengths, abundance, uncertainty, variance = generate_synthetic_data()
    
    print("Applying Markov Random Field filter...")
    mrf = MarkovRandomFieldFilter(beta=1.5, iterations=5, threshold=0.1)
    
    # Optional external data (e.g., using a single band of the cube as 'albedo' proxy)
    external_data = cube[:, :, 10:11] 
    
    smoothed_abundance = mrf.apply(abundance, uncertainty, external_data=external_data)
    
    print(f"Original abundance shape: {abundance.shape}")
    print(f"Smoothed abundance shape: {smoothed_abundance.shape}")
    
    print("Exporting to GeoTIFF...")
    exporter = GeoTIFFExporter(crs_epsg=4326) # WGS84 for testing
    os.makedirs("output", exist_ok=True)
    out_path = "output/synthetic_resource_map.tif"
    
    # Create mock metadata
    metadata = {
        'WAVELENGTHS': ','.join([f"{w:.1f}" for w in wavelengths]),
        'SENSOR': 'SYNTHETIC_TEST',
        'UNITS': 'Reflectance'
    }
    
    exporter.export(
        out_path, 
        smoothed_abundance, 
        uncertainty, 
        variance,
        metadata=metadata
    )
    
    print(f"File exists: {os.path.exists(out_path)}")
    
    print("Launching Interactive Visualizer...")
    # NOTE: In a non-interactive environment, this will block until closed.
    visualizer = InteractiveVisualizer(
        cube, 
        wavelengths, 
        smoothed_abundance, 
        uncertainty, 
        class_names=["Olivine", "Pyroxene", "Plagioclase"]
    )
    visualizer.show()

if __name__ == "__main__":
    main()
