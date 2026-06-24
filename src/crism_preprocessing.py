import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VolcanoScanCorrection(nn.Module):
    """
    Applies atmospheric correction using the Volcano-Scan method.
    Typically, this uses an atmospheric transmission spectrum derived from a scan over Olympus Mons.
    """
    def __init__(self, atmospheric_transmission=None):
        super().__init__()
        # atmospheric_transmission: 1D tensor of shape (Bands,)
        if atmospheric_transmission is not None:
            self.register_buffer('transmission', atmospheric_transmission)
        else:
            self.transmission = None

    def forward(self, x):
        # x shape: (Bands, Lines, Samples)
        if self.transmission is not None:
            # Broadcast transmission to match x
            t = self.transmission.view(-1, 1, 1)
            # Divide by transmission to correct atmospheric absorption
            return x / (t + 1e-8)
        return x

class SpatialDestriper(nn.Module):
    """
    Performs spatial destriping and bad-pixel interpolation using median filtering
    across adjacent lines (spatial dimensions).
    """
    def __init__(self, window_size=3):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        # x shape: (Bands, Lines, Samples)
        b, l, s = x.shape
        
        # Add batch dimension for unfold: (1, Bands, Lines, Samples)
        x_unfold = x.unsqueeze(0) 
        
        pad = self.window_size // 2
        # Padding format for F.pad is (left, right, top, bottom)
        x_padded = F.pad(x_unfold, (pad, pad, pad, pad), mode='reflect')
        
        # Unfold extracts sliding local blocks
        patches = F.unfold(x_padded, kernel_size=self.window_size)
        # patches shape: (1, Bands * window_size * window_size, L * S)
        
        patches = patches.view(1, b, self.window_size * self.window_size, l * s)
        # Median across the patch dimension (the spatial neighborhood)
        x_destriped, _ = patches.median(dim=2) # (1, b, l * s)
        
        x_destriped = x_destriped.view(b, l, s)
        return x_destriped

class ContinuumRemoval(nn.Module):
    """
    Isolates absorption features (like 1.5 µm and 3.0 µm water ice and hydroxyl)
    by fitting an upper convex hull to each spectrum and dividing it.
    """
    def __init__(self, wavelengths):
        super().__init__()
        self.wavelengths = wavelengths # numpy array or tensor of shape (Bands,)

    def _get_upper_convex_hull_continuum(self, spectrum, wavs):
        """
        Computes the upper convex hull for a 1D spectrum using a monotonic chain algorithm.
        """
        points = np.column_stack((wavs, spectrum))
        # Sort by wavelength (x)
        points = points[np.argsort(points[:, 0])]
        
        upper = []
        for p in points:
            while len(upper) >= 2:
                p1, p2 = upper[-2], upper[-1]
                # Cross product to check for right turn (convex upper boundary)
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross >= 0: # Not a right turn, meaning it breaks the upper convexity
                    upper.pop()
                else:
                    break
            upper.append(p)
        
        upper = np.array(upper)
        # Interpolate the upper hull to all wavelengths to form the continuum
        continuum = np.interp(wavs, upper[:, 0], upper[:, 1])
        return continuum

    def forward(self, x):
        # x shape: (Bands, Lines, Samples)
        b, l, s = x.shape
        x_np = x.detach().cpu().numpy()
        wavs_np = self.wavelengths.detach().cpu().numpy() if torch.is_tensor(self.wavelengths) else self.wavelengths
        
        # Flatten spatial dims to iterate over spectra
        x_flat = x_np.reshape(b, -1)
        
        continuum_flat = np.zeros_like(x_flat)
        for i in range(x_flat.shape[1]):
            spectrum = x_flat[:, i]
            continuum = self._get_upper_convex_hull_continuum(spectrum, wavs_np)
            # Continuum removal: divide spectrum by continuum
            continuum_flat[:, i] = spectrum / (continuum + 1e-8)
            
        cr_np = continuum_flat.reshape(b, l, s)
        cr_tensor = torch.from_numpy(cr_np).to(x.device).type(x.dtype)
        
        return cr_tensor

class AlbedoNormalizer(nn.Module):
    """
    Normalizes spectra to a common albedo scale to make models invariant to overall brightness.
    """
    def __init__(self, method='max'):
        super().__init__()
        self.method = method
        
    def forward(self, x):
        # x shape: (Bands, Lines, Samples)
        if self.method == 'max':
            norm_factors, _ = x.max(dim=0, keepdim=True)
        elif self.method == 'mean':
            norm_factors = x.mean(dim=0, keepdim=True)
        else:
            norm_factors = 1.0
            
        return x / (norm_factors + 1e-8)

class PhysicsAwarePCA(nn.Module):
    """
    Dimensionality reduction via PCA, with heuristic weighting applied to bands
    corresponding to important diagnostic features (e.g., 1.5 µm and 3.0 µm).
    """
    def __init__(self, n_components=20, wavelengths=None):
        super().__init__()
        self.n_components = n_components
        if torch.is_tensor(wavelengths):
            self.register_buffer('wavelengths', wavelengths)
        else:
            self.register_buffer('wavelengths', torch.tensor(wavelengths, dtype=torch.float32))
            
        self.transform_matrix = None
        self.mean_vector = None
        
    def _compute_heuristic_weights(self):
        wavs = self.wavelengths
        weights = torch.ones_like(wavs)
        
        # Emphasize the 1.5 µm and 3.0 µm regions for water ice and hydroxyl
        w1_mask = (wavs >= 1.4) & (wavs <= 1.6)
        w2_mask = (wavs >= 2.8) & (wavs <= 3.2)
        
        weights[w1_mask] *= 2.0
        weights[w2_mask] *= 3.0
        
        return weights
        
    def forward(self, x):
        # x shape: (Bands, Lines, Samples)
        b, l, s = x.shape
        x_flat = x.view(b, -1).t() # (L*S, Bands)
        
        weights = self._compute_heuristic_weights().to(x.device)
        # Apply physics-aware weights before PCA
        x_weighted = x_flat * weights
            
        # Center the data
        mean_vector = x_weighted.mean(dim=0, keepdim=True)
        x_centered = x_weighted - mean_vector
        
        # Compute low-rank PCA
        U, S, V = torch.pca_lowrank(x_centered, q=self.n_components)
        
        # Store for reconstruction / inference later
        self.transform_matrix = V # (Bands, n_components)
        self.mean_vector = mean_vector
        
        # Projected data
        x_pca = torch.matmul(x_centered, V) # (L*S, n_components)
        
        # Reshape back to spatial layout: (n_components, Lines, Samples)
        x_pca = x_pca.t().view(self.n_components, l, s)
        
        return x_pca

class CRISMPipeline(nn.Module):
    """
    Master pipeline unifying all CRISM preprocessing steps.
    """
    def __init__(self, 
                 wavelengths, 
                 atmospheric_transmission=None, 
                 spatial_context_size=3,
                 pca_components=20):
        super().__init__()
        self.spatial_context_size = spatial_context_size
        
        # Pipeline modules
        self.atm_corr = VolcanoScanCorrection(atmospheric_transmission)
        self.destriper = SpatialDestriper(window_size=3) # 3x3 median for destriping
        self.continuum = ContinuumRemoval(wavelengths)
        self.normalizer = AlbedoNormalizer(method='max')
        self.pca = PhysicsAwarePCA(n_components=pca_components, wavelengths=wavelengths)
        
    def extract_spatial_context(self, x, window_size):
        """
        Extracts spatial context neighborhoods around each pixel.
        Output shape: (N_pixels, N_bands_reduced, spatial_context)
        where spatial_context = window_size * window_size.
        """
        b, l, s = x.shape
        x_unfold = x.unsqueeze(0) # (1, Bands, Lines, Samples)
        
        pad = window_size // 2
        x_padded = F.pad(x_unfold, (pad, pad, pad, pad), mode='reflect')
        
        # F.unfold shape: (1, Bands * window_size^2, Lines * Samples)
        patches = F.unfold(x_padded, kernel_size=window_size)
        
        # Reshape to easily permute dimensions
        patches = patches.view(1, b, window_size**2, l * s)
        patches = patches.squeeze(0).permute(2, 0, 1) # (Lines * Samples, b, window_size^2)
        
        return patches
        
    def forward(self, x):
        """
        Expects x shape: (Bands, Lines, Samples)
        Returns: (N_pixels, N_bands_reduced, spatial_context)
        """
        # 1. Atmospheric Correction
        x = self.atm_corr(x)
        
        # 2. Spatial Destriping
        x = self.destriper(x)
        
        # 3. Continuum Removal
        x = self.continuum(x)
        
        # 4. Albedo Normalization
        x = self.normalizer(x)
        
        # 5. Physics-Aware PCA
        x_reduced = self.pca(x) 
        
        # 6. Extract Spatial Context
        out = self.extract_spatial_context(x_reduced, self.spatial_context_size)
        
        return out
