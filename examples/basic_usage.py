import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mppca_implementations import (
    MP, 
    MPnonlocal, 
    MP_PCA_controlled, 
    custom_mppca, 
    improved_mppca, 
    fixed_mppca
)

def generate_test_data(shape=(64, 64, 64, 4), noise_level=0.05):
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
    
    return data, noisy_data

def visualize_denoising_results(original, noisy, denoised, slice_idx=32, direction='z', title=None):
    """Visualize original, noisy, and denoised data."""
    plt.figure(figsize=(15, 4))
    
    # Select slice based on direction
    if direction == 'x':
        orig_slice = original[slice_idx, :, :, 0]
        noisy_slice = noisy[slice_idx, :, :, 0]
        denoised_slice = denoised[slice_idx, :, :, 0]
    elif direction == 'y':
        orig_slice = original[:, slice_idx, :, 0]
        noisy_slice = noisy[:, slice_idx, :, 0]
        denoised_slice = denoised[:, slice_idx, :, 0]
    else:  # 'z'
        orig_slice = original[:, :, slice_idx, 0]
        noisy_slice = noisy[:, :, slice_idx, 0]
        denoised_slice = denoised[:, :, slice_idx, 0]
    
    plt.subplot(131)
    plt.imshow(orig_slice, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(noisy_slice, cmap='gray')
    plt.title('Noisy')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(denoised_slice, cmap='gray')
    plt.title('Denoised' if title is None else title)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate test data
    print("Generating test data...")
    original_data, noisy_data = generate_test_data(noise_level=0.1)
    
    print("\n==== RECOMMENDED IMPLEMENTATION ====")
    
    # Demonstrate improved MP-PCA (RECOMMENDED)
    print("\nPerforming improved MP-PCA denoising (RECOMMENDED)...")
    denoised_improved, sigma_improved, npars_improved = improved_mppca(noisy_data, patch_radius=1)
    
    print(f"Improved MP-PCA results: sigma={sigma_improved:.4f}, npars={npars_improved}")
    visualize_denoising_results(original_data, noisy_data, denoised_improved, 
                               title="Denoised (Improved MP-PCA)")
    
    print("\n==== ALTERNATIVE IMPLEMENTATIONS ====")
    
    # Demonstrate basic MP denoising
    print("\nPerforming basic MP denoising (alternative)...")
    # Reshape for MP algorithm (which expects 2D input)
    x, y, z, n = noisy_data.shape
    reshaped_data = noisy_data.reshape(x*y*z, n)
    denoised_reshaped, sigma, npars = MP(reshaped_data)
    denoised_basic = denoised_reshaped.reshape(x, y, z, n)
    
    print(f"Basic MP results: sigma={sigma:.4f}, npars={npars}")
    visualize_denoising_results(original_data, noisy_data, denoised_basic,
                              title="Denoised (Basic MP)")
    
    # Demonstrate controlled MP-PCA denoising
    print("\nPerforming controlled MP-PCA denoising (alternative)...")
    denoised_controlled = custom_mppca(noisy_data, patch_radius=1, n_components=2)
    
    print("Controlled MP-PCA completed")
    visualize_denoising_results(original_data, noisy_data, denoised_controlled,
                              title="Denoised (Controlled MP-PCA)")
    
    # Demonstrate fixed MP-PCA
    print("\nPerforming fixed MP-PCA denoising (alternative)...")
    denoised_fixed = fixed_mppca(noisy_data, patch_radius=1)
    
    print("Fixed MP-PCA completed")
    visualize_denoising_results(original_data, noisy_data, denoised_fixed,
                              title="Denoised (Fixed MP-PCA)")
    
    # Demonstrate non-local MP-PCA
    print("\nPerforming non-local MP-PCA denoising (alternative)...")
    denoised_nonlocal, sigma_nonlocal, npars_nonlocal, _ = MPnonlocal(noisy_data, kernel=[3, 3, 3], patchtype='nonlocal')
    
    print(f"Non-local MP-PCA results: sigma={sigma_nonlocal.mean():.4f}")
    visualize_denoising_results(original_data, noisy_data, denoised_nonlocal,
                              title="Denoised (Non-local MP-PCA)")

if __name__ == "__main__":
    main() 