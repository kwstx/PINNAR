import h5py
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_to_hdf5(output_path, dataset_name, cube_data, wavelengths, metadata):
    """
    Write the resampled spectral cube and metadata to a common HDF5 data lake format.
    The cube is written in chunks to support streaming and efficient reading later.
    
    Parameters:
    - output_path: str, path to the HDF5 file.
    - dataset_name: str, name of the dataset/group within the HDF5 file (e.g. 'CRISM_OBS_01').
    - cube_data: 3D numpy array, the resampled spectral cube.
    - wavelengths: 1D numpy array, the shared spectral grid.
    - metadata: dict, containing spatial and observation geometry and SNR tags.
    """
    logger.info(f"Writing to HDF5: {output_path}, Group: {dataset_name}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        # Open in append mode so we can add multiple observations to the same data lake file
        with h5py.File(output_path, 'a') as hf:
            # Create a group for this specific observation
            if dataset_name in hf:
                logger.warning(f"Dataset {dataset_name} already exists. Overwriting.")
                del hf[dataset_name]
            
            group = hf.create_group(dataset_name)
            
            # Write the spectral cube using chunking and compression
            # Chunking strategy: chunk along spatial dimensions, full spectral depth if possible,
            # or small spatial chunks. Let's use auto chunking or specify a reasonable chunk size.
            bands, lines, samples = cube_data.shape
            chunk_shape = (bands, min(lines, 64), min(samples, 64))
            
            cube_ds = group.create_dataset(
                'spectral_cube', 
                data=cube_data, 
                chunks=chunk_shape, 
                compression='gzip',
                compression_opts=4
            )
            
            # Write wavelengths
            group.create_dataset('wavelengths', data=wavelengths)
            
            # Write metadata as attributes
            for key, value in metadata.items():
                if value is not None:
                    # h5py attributes need to be compatible types (str, int, float, array)
                    if isinstance(value, dict):
                        # Serialize nested dicts as strings or create subgroups
                        group.attrs[key] = str(value)
                    else:
                        group.attrs[key] = value
                        
            # Specific required metadata tagging for SNR preservation
            if 'snr_metadata' in metadata:
                cube_ds.attrs['snr_characteristics'] = str(metadata['snr_metadata'])
                
        logger.info(f"Successfully wrote {dataset_name} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to HDF5: {e}")
        return False

if __name__ == "__main__":
    # Mock test
    mock_cube = np.random.rand(261, 100, 100)
    mock_wavs = np.linspace(0.4, 3.0, 261)
    meta = {
        'incidence_angle': 45.0,
        'emission_angle': 10.0,
        'phase_angle': 35.0,
        'snr_metadata': 'SNR>100 for visible bands',
        'mission': 'Mock_Mission'
    }
    write_to_hdf5("test_lake.h5", "OBS_001", mock_cube, mock_wavs, meta)
    print("Mock write successful.")
