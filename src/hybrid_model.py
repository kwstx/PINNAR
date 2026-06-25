import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels, num_domains=3):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_domains)
        )
        
    def forward(self, x):
        x = self.grl(x)
        return self.net(x)

class SuperResolutionBranch(nn.Module):
    def __init__(self, in_channels, num_endmembers, num_bands, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        self.hr_head = PhysicsInformedHead(in_channels, num_endmembers, num_bands)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        hr_abundances, hr_ssa, hr_log_var, hr_abundance_log_var = self.hr_head(x)
        return hr_abundances, hr_ssa, hr_log_var, hr_abundance_log_var

class FourierFeatureMapping(nn.Module):
    """
    Applies sinusoidal positional encodings at multiple frequencies 
    to capture high-frequency spectral absorption features.
    """
    def __init__(self, input_dim, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Log-spaced frequencies for capturing features at different scales
        self.frequencies = 2.0 ** torch.arange(num_frequencies) * torch.pi
        
    def forward(self, x):
        # x shape: (B, H, W, C)
        freqs = self.frequencies.to(x.device)
        
        # Prepare for broadcasting: x -> (B, H, W, C, 1), freqs -> (num_frequencies)
        arg = x.unsqueeze(-1) * freqs
        arg = arg.reshape(*x.shape[:-1], -1)  # Flatten last two dims to (..., C * num_freqs)
        
        # Compute sine and cosine encodings
        fourier_features = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        
        # Concatenate original input with fourier features
        return torch.cat([x, fourier_features], dim=-1)


class SpectralEncoder(nn.Module):
    """
    Spectral branch consisting of fully connected layers acting on 
    Fourier-mapped spectral signatures pixel-by-pixel.
    """
    def __init__(self, input_bands, latent_dim=64, num_frequencies=4):
        super().__init__()
        self.fourier = FourierFeatureMapping(input_bands, num_frequencies=num_frequencies)
        
        # Original input + sine features + cosine features
        fourier_dim = input_bands + 2 * input_bands * num_frequencies
        
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim)
        )
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        # Permute to (B, H, W, C) for linear layer pixel-wise application
        x = x.permute(0, 2, 3, 1) 
        feat = self.fourier(x)
        latent = self.mlp(feat)
        
        # Permute back to (B, latent_dim, H, W)
        latent = latent.permute(0, 3, 1, 2)
        return latent


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net-style spatial branch operating on 32x32 patches
    to enforce spatial smoothness of mineral distributions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x is (B, C, 32, 32)
        x1 = self.inc(x)              # 32x32
        x2 = self.down1(x1)           # 16x16
        x3 = self.down2(x2)           # 8x8
        
        x = self.up1(x3)              # 16x16
        x = torch.cat([x, x2], dim=1) # 16x16
        x = self.conv1(x)             # 16x16
        
        x = self.up2(x)               # 32x32
        x = torch.cat([x, x1], dim=1) # 32x32
        x = self.conv2(x)             # 32x32
        
        return self.outc(x)           # 32x32


class PhysicsInformedHead(nn.Module):
    """
    Output head that directly predicts abundance fractions and effective 
    single-scattering albedo (SSA).
    """
    def __init__(self, in_channels, num_endmembers, num_bands):
        super().__init__()
        self.abundance_conv = nn.Conv2d(in_channels, num_endmembers, kernel_size=1)
        self.ssa_conv = nn.Conv2d(in_channels, num_bands, kernel_size=1)
        # Output log-variance for predictive uncertainty estimation of reflectance
        self.log_var_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Output log-variance for aleatoric uncertainty of abundances
        self.abundance_log_var_conv = nn.Conv2d(in_channels, num_endmembers, kernel_size=1)
        
    def forward(self, x):
        abundances = self.abundance_conv(x)
        # Softmax constraint on abundances to guarantee physical summation to unity
        abundances = F.softmax(abundances, dim=1)
        
        # Predict SSA (bounded between 0 and 1)
        ssa = torch.sigmoid(self.ssa_conv(x))
        
        # Predict log variance
        log_var = self.log_var_conv(x)
        abundance_log_var = self.abundance_log_var_conv(x)
        
        return abundances, ssa, log_var, abundance_log_var


class HybridSpectralSpatialModel(nn.Module):
    """
    Hybrid neural architecture fusing spectral physics with spatial context.
    
    - Spectral Encoder: Fully connected layers with Fourier feature mapping
    - Spatial Branch: Lightweight U-Net-style on 32x32 patches
    - Output: Predicts abundance fractions (softmax constrained), effective SSA, and predictive log-variance
    - Domain Adaptation: Uses GRL to align feature distributions across instruments.
    - Super Resolution: Sub-pixel convolutional branch for spatial upscaling.
    """
    def __init__(self, num_bands, num_endmembers, spec_latent_dim=64, spat_latent_dim=32, fourier_freqs=4, num_domains=3, upscale_factor=2):
        super().__init__()
        
        self.spectral_encoder = SpectralEncoder(num_bands, spec_latent_dim, fourier_freqs)
        self.spatial_branch = LightweightUNet(num_bands, spat_latent_dim)
        
        fusion_dim = spec_latent_dim + spat_latent_dim
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.output_head = PhysicsInformedHead(128, num_endmembers, num_bands)
        self.domain_discriminator = DomainDiscriminator(128, num_domains)
        self.super_res_branch = SuperResolutionBranch(128, num_endmembers, num_bands, upscale_factor)
        
    def forward(self, x):
        """
        x: Input hyperspectral image patches of shape (B, num_bands, 32, 32)
        Returns:
            abundances: (B, num_endmembers, 32, 32)
            ssa: (B, num_bands, 32, 32) - effective single-scattering albedo
            log_var: (B, 1, 32, 32) - predictive log-variance for reflectance uncertainty
            abundance_log_var: (B, num_endmembers, 32, 32) - aleatoric uncertainty for abundances
            domain_logits: (B, num_domains) - logits for domain classification
            hr_abundances: (B, num_endmembers, 32*upscale, 32*upscale)
            hr_ssa: (B, num_bands, 32*upscale, 32*upscale)
        """
        # Process through Spectral Branch (pixel-wise processing of Fourier features)
        spec_features = self.spectral_encoder(x)
        
        # Process through Spatial Branch (U-Net extracting structural/spatial context)
        spat_features = self.spatial_branch(x)
        
        # Fuse branches
        fused = torch.cat([spec_features, spat_features], dim=1)
        fused = self.fusion_conv(fused)
        
        # Physics-informed output
        abundances, ssa, log_var, abundance_log_var = self.output_head(fused)
        
        # Domain adaptation output
        domain_logits = self.domain_discriminator(fused)
        
        # Super-resolution output
        hr_abundances, hr_ssa, _, _ = self.super_res_branch(fused)
        
        return abundances, ssa, log_var, abundance_log_var, domain_logits, hr_abundances, hr_ssa
