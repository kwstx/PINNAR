import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.colors as mcolors

class InteractiveVisualizer:
    """
    An interactive visualization module for PINNAR resource maps.
    Overlays abundance maps on basemap imagery, displays uncertainty as
    semi-transparent overlays, and allows spectral extraction at user-selected points.
    """
    def __init__(self, hyperspectral_cube, wavelengths, mean_abundance, uncertainty, class_names=None):
        """
        Args:
            hyperspectral_cube (np.ndarray): Shape (H, W, B). The raw or processed data cube.
            wavelengths (np.ndarray): Shape (B,). Wavelengths corresponding to the cube bands.
            mean_abundance (np.ndarray): Shape (H, W, C). Predicted mean abundances.
            uncertainty (np.ndarray): Shape (H, W, C). Predictive uncertainty.
            class_names (list): List of string names for the C endmembers.
        """
        self.cube = hyperspectral_cube
        self.wavelengths = wavelengths
        self.abundance = mean_abundance
        self.uncertainty = uncertainty
        
        self.H, self.W, self.C = self.abundance.shape
        self.class_names = class_names if class_names else [f"Endmember {i}" for i in range(self.C)]
        
        # Extract basemap from the middle band of the cube
        mid_band_idx = len(wavelengths) // 2
        self.basemap = self.cube[:, :, mid_band_idx]
        
        self.fig = None
        self.ax_map = None
        self.ax_spec = None
        self.current_endmember_idx = 0

    def _normalize_uncertainty_alpha(self, unc_map):
        """
        Normalizes uncertainty to [0, 1] to be used as an alpha channel (opacity).
        Lower uncertainty -> higher opacity (more visible abundance).
        """
        u_min, u_max = unc_map.min(), unc_map.max()
        if u_max == u_min:
            return np.ones_like(unc_map)
        
        # Invert so high uncertainty means low alpha (more transparent)
        alpha = 1.0 - (unc_map - u_min) / (u_max - u_min + 1e-6)
        # Clip to ensure it's not completely invisible
        return np.clip(alpha, 0.2, 1.0)

    def _draw_map(self):
        """Draws the basemap and the overlay for the current endmember."""
        self.ax_map.clear()
        
        # Plot basemap in grayscale
        self.ax_map.imshow(self.basemap, cmap='gray')
        
        # Get current endmember maps
        abund_map = self.abundance[:, :, self.current_endmember_idx]
        unc_map = self.uncertainty[:, :, self.current_endmember_idx]
        
        # Prepare an RGBA image for the abundance overlay
        # We use a colormap (e.g., inferno or viridis) and set alpha based on uncertainty
        cmap = plt.get_cmap('inferno')
        norm = mcolors.Normalize(vmin=0, vmax=1.0)
        
        rgba_img = cmap(norm(abund_map))
        # Modulate alpha channel by uncertainty
        rgba_img[:, :, 3] = self._normalize_uncertainty_alpha(unc_map) * (abund_map > 0.05).astype(float)
        
        # Overlay the abundance map
        self.ax_map.imshow(rgba_img)
        self.ax_map.set_title(f"Abundance Overlay: {self.class_names[self.current_endmember_idx]}\n(Opacity modulated by Certainty)")
        self.ax_map.axis('off')
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handles click events on the map to extract and plot spectra."""
        if event.inaxes != self.ax_map:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # Ensure bounds
        if 0 <= x < self.W and 0 <= y < self.H:
            spectrum = self.cube[y, x, :]
            
            self.ax_spec.clear()
            self.ax_spec.plot(self.wavelengths, spectrum, 'b-', label='Extracted Spectrum')
            self.ax_spec.set_title(f"Spectral Signature at ({x}, {y})")
            self.ax_spec.set_xlabel("Wavelength")
            self.ax_spec.set_ylabel("Reflectance")
            self.ax_spec.grid(True)
            self.fig.canvas.draw_idle()

    def _next_endmember(self, event):
        """Switches to the next endmember overlay."""
        self.current_endmember_idx = (self.current_endmember_idx + 1) % self.C
        self._draw_map()

    def show(self):
        """Displays the interactive visualization."""
        self.fig, (self.ax_map, self.ax_spec) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})
        
        # Setup next button
        ax_next = plt.axes([0.45, 0.01, 0.1, 0.05])
        self.btn_next = Button(ax_next, 'Next Mineral')
        self.btn_next.on_clicked(self._next_endmember)
        
        # Initial draw
        self._draw_map()
        
        # Initial empty spectrum
        self.ax_spec.set_title("Click on the map to extract spectrum")
        self.ax_spec.set_xlabel("Wavelength")
        self.ax_spec.set_ylabel("Reflectance")
        self.ax_spec.grid(True)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        plt.tight_layout(rect=[0, 0.08, 1, 1]) # Leave room for button
        plt.show()
