import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mppca_implementations.MP import MP
from mppca_implementations.mppca_controlled import custom_mppca
from utils.phase_correction import phase_correction, create_circular_mask

def generate_complex_test_data(shape=(64, 64, 64, 4), noise_level=0.05, phase_offset=np.pi/4):
    """Generate synthetic 4D complex MRI-like data with noise."""
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
    data = np.zeros(shape, dtype=complex)
    for i in range(n):
        # Create magnitude
        magnitude = sphere * (1.0 - 0.2 * i)
        
        # Create phase with a gradient
        phase = phase_offset * (x_grid + y_grid + z_grid)
        
        # Combine magnitude and phase
        data[:, :, :, i] = magnitude * np.exp(1j * phase)
    
    # Add complex noise
    noise_real = np.random.normal(0, noise_level, shape)
    noise_imag = np.random.normal(0, noise_level, shape)
    noise = noise_real + 1j * noise_imag
    
    noisy_data = data + noise
    
    return data, noisy_data

def visualize_complex_results(original, noisy, denoised, slice_idx=32):
    """Visualize original, noisy, and denoised complex data."""
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    
    # Get slices
    orig_slice = original[:, :, slice_idx, 0]
    noisy_slice = noisy[:, :, slice_idx, 0]
    denoised_slice = denoised[:, :, slice_idx, 0]
    
    # Display magnitude
    axs[0, 0].imshow(np.abs(orig_slice), cmap='gray')
    axs[0, 0].set_title('Original Magnitude')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(np.abs(noisy_slice), cmap='gray')
    axs[0, 1].set_title('Noisy Magnitude')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(np.abs(denoised_slice), cmap='gray')
    axs[0, 2].set_title('Denoised Magnitude')
    axs[0, 2].axis('off')
    
    # Display phase
    axs[1, 0].imshow(np.angle(orig_slice), cmap='hsv')
    axs[1, 0].set_title('Original Phase')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(np.angle(noisy_slice), cmap='hsv')
    axs[1, 1].set_title('Noisy Phase')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(np.angle(denoised_slice), cmap='hsv')
    axs[1, 2].set_title('Denoised Phase')
    axs[1, 2].axis('off')
    
    # Display real part
    axs[2, 0].imshow(np.real(orig_slice), cmap='gray')
    axs[2, 0].set_title('Original Real')
    axs[2, 0].axis('off')
    
    axs[2, 1].imshow(np.real(noisy_slice), cmap='gray')
    axs[2, 1].set_title('Noisy Real')
    axs[2, 1].axis('off')
    
    axs[2, 2].imshow(np.real(denoised_slice), cmap='gray')
    axs[2, 2].set_title('Denoised Real')
    axs[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate complex test data
    print("Generating complex test data...")
    original_data, noisy_data = generate_complex_test_data(noise_level=0.1)
    
    # Apply phase correction
    print("\nApplying phase correction...")
    corrected_data = np.zeros_like(noisy_data)
    for i in range(noisy_data.shape[3]):
        corrected_data[:, :, :, i] = phase_correction(noisy_data[:, :, :, i])
    
    # Denoise real part using controlled MP-PCA
    print("\nPerforming MP-PCA denoising on real part...")
    denoised_real = custom_mppca(np.real(corrected_data), patch_radius=1, n_components=2)
    
    # Denoise imaginary part using controlled MP-PCA
    print("\nPerforming MP-PCA denoising on imaginary part...")
    denoised_imag = custom_mppca(np.imag(corrected_data), patch_radius=1, n_components=2)
    
    # Combine real and imaginary parts
    denoised_complex = denoised_real + 1j * denoised_imag
    
    # Visualize results
    visualize_complex_results(original_data, noisy_data, denoised_complex)
    
    # Calculate metrics
    orig_mag = np.abs(original_data)
    denoised_mag = np.abs(denoised_complex)
    
    mse = np.mean((orig_mag - denoised_mag) ** 2)
    psnr = 20 * np.log10(orig_mag.max() / np.sqrt(mse))
    
    print(f"Denoising metrics: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

if __name__ == "__main__":
    main() 