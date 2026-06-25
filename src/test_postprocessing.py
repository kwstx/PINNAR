import pytest
import numpy as np
import os
import rasterio

from src.postprocessing import MarkovRandomFieldFilter, GeoTIFFExporter

def test_mrf_filter():
    mrf = MarkovRandomFieldFilter(beta=1.0, iterations=5, threshold=0.1)
    
    # Create synthetic abundance (10, 10, 2)
    mean_abundance = np.ones((10, 10, 2)) * 0.5
    # Add a strong signal in the center
    mean_abundance[4:6, 4:6, 0] = 0.9
    mean_abundance[4:6, 4:6, 1] = 0.1
    
    # Uniform low uncertainty
    uncertainty = np.ones((10, 10, 2)) * 0.01
    
    smoothed = mrf.apply(mean_abundance, uncertainty)
    
    assert smoothed.shape == mean_abundance.shape
    # Check normalization
    sums = smoothed.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    
    # Check that thresholding works (values below 0.1 should be 0)
    assert np.all((smoothed >= 0.1) | (smoothed == 0.0))

def test_geotiff_exporter(tmp_path):
    exporter = GeoTIFFExporter(crs_epsg=4326)
    
    out_path = tmp_path / "test_export.tif"
    
    H, W, C = 10, 10, 3
    mean_abundance = np.random.rand(H, W, C)
    epistemic_unc = np.random.rand(H, W, C)
    total_variance = np.random.rand(H, W, C)
    
    metadata = {'TEST_TAG': '12345'}
    
    exporter.export(str(out_path), mean_abundance, epistemic_unc, total_variance, metadata)
    
    assert out_path.exists()
    
    with rasterio.open(str(out_path)) as src:
        # Should have 3 * C bands
        assert src.count == 3 * C
        assert src.width == W
        assert src.height == H
        assert src.crs.to_epsg() == 4326
        
        # Check metadata
        tags = src.tags()
        assert 'TEST_TAG' in tags
        assert tags['TEST_TAG'] == '12345'
