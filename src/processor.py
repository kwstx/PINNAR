import numpy as np
from scipy.interpolate import CubicSpline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared spectral grid: 0.4 to 3.0 um, e.g., with 10nm (0.01 um) resolution
SHARED_WAVELENGTHS = np.arange(0.4, 3.01, 0.01)

def resample_cube(spectral_cube, original_wavelengths, target_wavelengths=SHARED_WAVELENGTHS):
    """
    Resample a spectral cube to a shared spectral grid using cubic spline interpolation.
    
    Parameters:
    - spectral_cube: 3D numpy array (bands, lines, samples) or (lines, samples, bands).
                     Assuming (bands, lines, samples) for this implementation.
    - original_wavelengths: 1D numpy array of original wavelengths.
    - target_wavelengths: 1D numpy array of target wavelengths.
    
    Returns:
    - resampled_cube: 3D numpy array resampled to target wavelengths.
    """
    logger.info("Resampling spectral cube using cubic spline interpolation.")
    
    # Ensure wavelengths are sorted for interpolation
    sort_idx = np.argsort(original_wavelengths)
    orig_wav_sorted = original_wavelengths[sort_idx]
    cube_sorted = spectral_cube[sort_idx, :, :]
    
    num_bands, lines, samples = cube_sorted.shape
    num_target_bands = len(target_wavelengths)
    
    # Initialize output array
    resampled_cube = np.zeros((num_target_bands, lines, samples), dtype=cube_sorted.dtype)
    
    # Flatten spatial dimensions for easier vectorized interpolation along the spectral axis
    # Reshape to (bands, lines * samples)
    flat_cube = cube_sorted.reshape(num_bands, -1)
    
    # We apply interpolation along axis 0
    # Note: CubicSpline requires the independent variable (wavelengths) to be strictly increasing.
    try:
        cs = CubicSpline(orig_wav_sorted, flat_cube, axis=0, extrapolate=False)
        flat_resampled = cs(target_wavelengths)
        
        # Replace NaN values (from extrapolation bounds) with 0 or a nodata value if necessary
        flat_resampled = np.nan_to_num(flat_resampled, nan=0.0)
        
        # Reshape back to (target_bands, lines, samples)
        resampled_cube = flat_resampled.reshape(num_target_bands, lines, samples)
        logger.info(f"Resampling complete. Original shape: {spectral_cube.shape}, New shape: {resampled_cube.shape}")
        
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        return None
        
    return resampled_cube

if __name__ == "__main__":
    # Mock test
    mock_cube = np.random.rand(100, 50, 50)
    mock_wavs = np.linspace(0.3, 3.5, 100)
    res = resample_cube(mock_cube, mock_wavs)
    print("Resampled shape:", res.shape)
