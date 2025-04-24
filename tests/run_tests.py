#!/usr/bin/env python3
"""
Run all tests for the Na-MRI denoising MP-PCA implementations.

This script allows for quick verification of all implementations without using pytest.
"""
import numpy as np
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

def main():
    """Run tests for all implementations."""
    print("Generating test data...")
    test_data_4d = generate_test_data()
    test_data_2d = test_data_4d.reshape(-1, test_data_4d.shape[-1])
    
    # Test all implementations
    implementations = [
        {
            "name": "Basic MP",
            "function": lambda data: MP(test_data_2d),
            "data": test_data_2d,
            "description": "Basic 2D implementation"
        },
        {
            "name": "MP_PCA_controlled",
            "function": lambda data: MP_PCA_controlled(test_data_2d),
            "data": test_data_2d,
            "description": "MP-PCA with automatic component selection"
        },
        {
            "name": "MP_PCA_controlled (fixed components)",
            "function": lambda data: MP_PCA_controlled(test_data_2d, n_components=2),
            "data": test_data_2d,
            "description": "MP-PCA with fixed number of components"
        },
        {
            "name": "custom_mppca",
            "function": lambda data: custom_mppca(test_data_4d, patch_radius=1, n_components=2),
            "data": test_data_4d,
            "description": "Patch-based controlled MP-PCA for 4D data"
        },
        {
            "name": "fixed_mppca",
            "function": lambda data: fixed_mppca(test_data_4d, patch_radius=1),
            "data": test_data_4d,
            "description": "Fixed threshold patch-based MP-PCA"
        },
        {
            "name": "improved_mppca",
            "function": lambda data: improved_mppca(test_data_4d, patch_radius=1),
            "data": test_data_4d,
            "description": "Improved patch-based MP-PCA (recommended)"
        },
        {
            "name": "MPnonlocal",
            "function": lambda data: MPnonlocal(test_data_4d, kernel=[3, 3, 3], patchtype='box'),
            "data": test_data_4d,
            "description": "Non-local variant with box-type patches"
        }
    ]
    
    print("\nTesting all implementations:")
    print("="*50)
    
    for impl in implementations:
        print(f"\nTesting: {impl['name']} - {impl['description']}")
        try:
            result = impl["function"](impl["data"])
            print(f"✅ SUCCESS: {impl['name']} works properly")
            if hasattr(result, '__len__') and len(result) > 1:
                print(f"   Output shape: {result[0].shape}, Additional outputs: {len(result)-1}")
            else:
                print(f"   Output shape: {result.shape}")
        except Exception as e:
            print(f"❌ FAILED: {impl['name']} raised an error:")
            print(f"   {str(e)}")
    
    print("\n" + "="*50)
    print("Test completed.")

if __name__ == "__main__":
    main() 