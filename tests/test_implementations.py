"""
Unit tests for MPPCA implementations.

This module contains pytest-based unit tests to verify the correctness 
of all MPPCA implementations in the package.
"""
import numpy as np
import pytest
from mppca_implementations import (
    MP,
    MPnonlocal,
    MP_PCA_controlled,
    custom_mppca,
    improved_mppca,
    fixed_mppca
)

def generate_test_data(shape=(16, 16, 16, 4), noise_level=0.1):
    """Generate synthetic 4D MRI-like data with noise."""
    # Create signal
    x, y, z, n = shape
    
    # Create a synthetic MRI-like signal with a simple pattern
    x_grid, y_grid, z_grid = np.meshgrid(
        np.linspace(-1, 1, x),
        np.linspace(-1, 1, y),
        np.linspace(-1, 1, z),
        indexing='ij'
    )
    
    # Create a spherical object
    radius = 0.6
    sphere = (x_grid**2 + y_grid**2 + z_grid**2) < radius**2
    
    # Create data with different contrasts
    data = np.zeros(shape)
    for i in range(n):
        data[:, :, :, i] = sphere * (1.0 - 0.2 * i)  # Different contrasts
    
    # Add noise
    noise = np.random.normal(0, noise_level, shape)
    noisy_data = data + noise
    
    return noisy_data


class TestBasicImplementations:
    """Test basic MP-PCA implementations."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data fixture."""
        return generate_test_data()
    
    @pytest.fixture
    def test_data_2d(self, test_data):
        """Create 2D version of test data fixture."""
        return test_data.reshape(-1, test_data.shape[-1])
    
    def test_basic_mp(self, test_data_2d):
        """Test basic MP-PCA implementation."""
        result = MP(test_data_2d)
        assert result is not None
        # Check if result is a tuple (multiple outputs) or a single array
        if isinstance(result, tuple):
            assert result[0].shape == test_data_2d.shape
        else:
            assert result.shape == test_data_2d.shape
    
    def test_mp_pca_controlled(self, test_data_2d):
        """Test MP-PCA with automatic component selection."""
        result = MP_PCA_controlled(test_data_2d)
        assert result is not None
        # Check if result is a tuple (multiple outputs) or a single array
        if isinstance(result, tuple):
            assert result[0].shape == test_data_2d.shape
        else:
            assert result.shape == test_data_2d.shape
    
    def test_mp_pca_controlled_fixed(self, test_data_2d):
        """Test MP-PCA with fixed number of components."""
        result = MP_PCA_controlled(test_data_2d, n_components=2)
        assert result is not None
        # Check if result is a tuple (multiple outputs) or a single array
        if isinstance(result, tuple):
            assert result[0].shape == test_data_2d.shape
        else:
            assert result.shape == test_data_2d.shape


class TestAdvancedImplementations:
    """Test advanced MP-PCA implementations."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data fixture."""
        return generate_test_data()
    
    def test_custom_mppca(self, test_data):
        """Test patch-based controlled MP-PCA for 4D data."""
        result = custom_mppca(test_data, patch_radius=1, n_components=2)
        assert result is not None
        # Check if result is a tuple (multiple outputs) or a single array
        if isinstance(result, tuple):
            assert result[0].shape == test_data.shape
        else:
            assert result.shape == test_data.shape
    
    def test_fixed_mppca(self, test_data):
        """Test fixed threshold patch-based MP-PCA."""
        result = fixed_mppca(test_data, patch_radius=1)
        assert result is not None
        # Check if result is a tuple (multiple outputs) or a single array
        if isinstance(result, tuple):
            assert result[0].shape == test_data.shape
        else:
            assert result.shape == test_data.shape
    
    def test_improved_mppca(self, test_data):
        """Test improved patch-based MP-PCA."""
        result = improved_mppca(test_data, patch_radius=1)
        assert result is not None
        # Check if result is a tuple (multiple outputs) or a single array
        if isinstance(result, tuple):
            assert result[0].shape == test_data.shape
        else:
            assert result.shape == test_data.shape
    
    def test_mpnonlocal(self, test_data):
        """Test non-local variant with box-type patches."""
        result, sigma, npars, sigma_after = MPnonlocal(test_data, kernel=[3, 3, 3], patchtype='box')
        assert result is not None
        assert result.shape == test_data.shape
        assert sigma is not None
        assert npars is not None
        assert sigma_after is not None 