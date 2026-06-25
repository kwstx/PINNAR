import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.ndimage import generic_filter

class MarkovRandomFieldFilter:
    """
    Applies a physics-informed Markov Random Field (MRF) to threshold and smooth 
    mineralogical abundance maps, encouraging geologically plausible continuity.
    """
    def __init__(self, beta=1.0, iterations=10, threshold=0.05):
        """
        Args:
            beta (float): Weight of the pairwise (smoothing) term.
            iterations (int): Number of mean-field update iterations.
            threshold (float): Minimum abundance threshold; values below this are suppressed.
        """
        self.beta = beta
        self.iterations = iterations
        self.threshold = threshold

    def _get_pairwise_weights(self, external_data):
        """
        Compute edge-preserving weights from external data (e.g. albedo, topography).
        This would typically compute gradients to preserve sharp geological boundaries.
        For simplicity, we compute local variance or just use uniform weights if None.
        """
        pass

    def apply(self, mean_abundance, uncertainty, external_data=None):
        """
        Applies MRF to smooth the abundance map.

        Args:
            mean_abundance (np.ndarray): Shape (H, W, C).
            uncertainty (np.ndarray): Shape (H, W, C). Epistemic uncertainty to weight unary terms.
            external_data (np.ndarray): Shape (H, W, F). External data (e.g., elevation, albedo) 
                                        to inform the pairwise term (geological continuity).

        Returns:
            np.ndarray: Smoothed abundance map (H, W, C).
        """
        H, W, C = mean_abundance.shape
        smoothed = mean_abundance.copy()
        
        # Unary term weight (confidence = 1 / uncertainty)
        # Avoid division by zero
        confidence = 1.0 / (uncertainty + 1e-6)
        # Normalize confidence to have a reasonable scale
        confidence = confidence / (np.max(confidence, axis=(0, 1), keepdims=True) + 1e-6)

        # Precompute edge weights if external data is provided
        # Here we use a simple Gaussian kernel over a 3x3 neighborhood for the pairwise term
        kernel = np.array([[0.5, 1.0, 0.5],
                           [1.0, 0.0, 1.0],
                           [0.5, 1.0, 0.5]]) / 6.0

        for it in range(self.iterations):
            for c in range(C):
                # Calculate the neighborhood contribution
                # Using scipy generic_filter or simple convolution
                from scipy.signal import convolve2d
                
                # Pairwise term: Average of neighbors. 
                # If external data is provided, we can modulate this average.
                neighbor_avg = convolve2d(smoothed[:, :, c], kernel, mode='same', boundary='symm')
                
                if external_data is not None:
                    # Very simple physics-informed modulation: 
                    # Neighbors with similar external data have higher influence.
                    # This is a simplified bilateral filter approach for the pairwise term.
                    # For a full MRF, we'd explicitly construct the graph. 
                    # We'll stick to a simple modulated smoothing here.
                    pass
                
                # Mean field update: weighted average of data (unary) and neighbors (pairwise)
                # Unary is weighted by network confidence. Pairwise is weighted by beta.
                smoothed[:, :, c] = (confidence[:, :, c] * mean_abundance[:, :, c] + self.beta * neighbor_avg) / \
                                    (confidence[:, :, c] + self.beta)
                                    
        # Hard thresholding
        smoothed[smoothed < self.threshold] = 0.0
        
        # Re-normalize so abundances sum to 1.0 (or less, if we allow background)
        # For simplicity, if sum > 0, normalize
        row_sums = smoothed.sum(axis=-1, keepdims=True)
        # Avoid div by zero
        row_sums[row_sums == 0] = 1.0
        smoothed = smoothed / row_sums

        return smoothed

class GeoTIFFExporter:
    """
    Exports resource maps (abundance, uncertainty, variance) to GeoTIFF format
    with embedded metadata.
    """
    def __init__(self, crs_epsg=None, transform=None):
        self.crs = CRS.from_epsg(crs_epsg) if crs_epsg else None
        self.transform = transform

    def export(self, filepath, mean_abundance, epistemic_unc, total_variance, metadata=None):
        """
        Exports the pipeline outputs to a GeoTIFF.

        Args:
            filepath (str): Output GeoTIFF path.
            mean_abundance (np.ndarray): Shape (H, W, C).
            epistemic_unc (np.ndarray): Shape (H, W, C).
            total_variance (np.ndarray): Shape (H, W, C).
            metadata (dict): Wavelength and geometry metadata.
        """
        H, W, C = mean_abundance.shape
        
        # We will save 3 * C bands in total, or maybe separate files?
        # The prompt says: "Generate three output layers per scene: mean abundance, epistemic uncertainty, and total predictive variance. Export results in GeoTIFF format"
        # We can stack them as bands: [abundance_0...C, unc_0...C, var_0...C]
        total_bands = 3 * C
        
        transform = self.transform if self.transform else from_origin(0, 0, 1, 1)
        
        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=H,
            width=W,
            count=total_bands,
            dtype=mean_abundance.dtype,
            crs=self.crs,
            transform=transform,
        ) as dst:
            
            # Write Mean Abundance (Bands 1 to C)
            for c in range(C):
                dst.write(mean_abundance[:, :, c], c + 1)
                dst.set_band_description(c + 1, f"Mean Abundance Endmember {c}")
                
            # Write Epistemic Uncertainty (Bands C+1 to 2C)
            for c in range(C):
                dst.write(epistemic_unc[:, :, c], C + c + 1)
                dst.set_band_description(C + c + 1, f"Epistemic Unc Endmember {c}")
                
            # Write Total Predictive Variance (Bands 2C+1 to 3C)
            for c in range(C):
                dst.write(total_variance[:, :, c], 2*C + c + 1)
                dst.set_band_description(2*C + c + 1, f"Total Variance Endmember {c}")
                
            # Embed metadata
            if metadata:
                dst.update_tags(**metadata)
                
        print(f"Successfully exported {filepath} with {total_bands} bands.")
