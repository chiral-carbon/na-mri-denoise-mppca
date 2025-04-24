# Na-MRI Denoising using MP-PCA

A Python implementation of various Marchenko-Pastur Principal Component Analysis (MP-PCA) denoising methods for Sodium MRI (Na-MRI) data, with a primary focus on the improved MP-PCA implementation.

## Overview

This repository contains multiple implementations of the MP-PCA denoising algorithm, specifically tailored for Na-MRI data. The implementations include:

1. **Improved MP-PCA**: The recommended implementation with overlapping patches and automatic component selection, providing superior denoising quality.
2. **Fixed MP-PCA**: A faster variant with fixed component selection strategy.
3. **Basic MP-PCA**: The original MP-PCA algorithm based on [Veraart et al. (2016)](https://doi.org/10.1002/mrm.26059) [(github.com/NYU-DiffusionMRI/mppca_denoise)](https://github.com/NYU-DiffusionMRI/mppca_denoise)
4. **MP-PCA with controlled components**: An enhanced version allowing manual control of the number of components.
5. **Non-local MP-PCA**: Implementation of the non-local variant for improved performance based on [Veraart et al. (2016)](https://doi.org/10.1002/mrm.26059) [(github.com/NYU-DiffusionMRI/mppca_denoise)](https://github.com/NYU-DiffusionMRI/mppca_denoise)

The improved MP-PCA implementation is the primary focus and recommended approach for most applications, as it provides better noise reduction while preserving important signal features.

These implementations are designed to work with both magnitude and complex MRI data, with special considerations for phase correction in complex data.

## Installation

Ensure Python 3.10 or higher is installed.

### Setting up a virtual environment (recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python3.10 -m venv mppca-env

# Activate the virtual environment
# On Windows:
mppca-env\Scripts\activate
# On macOS/Linux:
source mppca-env/bin/activate
```

### Option 1: Install from repository

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/chiral-carbon/na-mri-denoise-mppca.git
cd na-mri-denoise-mppca

# Install all dependencies including pytest for testing
pip install -r requirements.txt
```

### Option 2: Install as a package (development mode)

For development purposes, you can install the package in development mode:

```bash
git clone https://github.com/chiral-carbon/na-mri-denoise-mppca.git
cd na-mri-denoise-mppca
pip install -e .
```

## Directory Structure and Usage

The repository is organized as follows:

```
na-mri-denoise-mppca/
├── mppca_implementations/   # Core algorithm implementations
│   ├── MP.py                # Basic MP-PCA implementation
│   ├── mp_nonlocal.py       # Non-local MP-PCA implementation
│   ├── mppca_controlled.py  # MP-PCA with controlled components
│   ├── mppca_helper.py      # Helper functions including improved_mppca
│   └── __init__.py          # Package exports
├── examples/                # Example scripts for getting started
│   ├── basic_usage.py       # Basic usage examples for all implementations
│   ├── complex_data_example.py  # Complex data handling with phase correction
│   └── __init__.py
├── tests/                   # Testing framework
│   ├── test_implementations.py  # Pytest-based unit tests
│   ├── run_tests.py         # Simple test runner (no pytest required)
│   ├── conftest.py          # Pytest configuration
│   └── __init__.py
├── utils/                   # Utility functions
│   ├── phase_correction.py  # Phase correction for complex data
│   ├── readNaImage.py       # Function to read Na-MRI data
│   └── __init__.py
├── notebooks/               # Jupyter notebooks with detailed examples
│   ├── complex_mppca_auto.ipynb     # Automated component selection
│   └── complex_mppca_manual.ipynb   # Manual component selection
├── requirements.txt         # Package dependencies
├── setup.py                 # Package setup script
├── LICENSE                  # License information
└── README.md                # This file
```

### mppca_implementations/

This directory contains the core algorithm implementations:

```bash
# View available implementations
ls -la mppca_implementations/

# Check function signatures and documentation
python -c "from mppca_implementations import improved_mppca; help(improved_mppca)"
```

### examples/

Contains runnable example scripts demonstrating different use cases:

```bash
# List example scripts
ls -la examples/

# Run the basic usage example (shows all implementations)
python examples/basic_usage.py

# Run the complex data example (shows phase correction)
python examples/complex_data_example.py
```

### notebooks/

Jupyter notebooks with detailed examples and visualizations:

```bash
# Install jupyter if needed
pip install jupyter

# Start Jupyter notebook server to explore the notebooks
jupyter notebook notebooks/

# Or use Jupyter Lab
jupyter lab notebooks/
```

### utils/

Utility functions for data handling and preprocessing:

```bash
# View available utilities
ls -la utils/

# Check phase correction documentation
python -c "from utils.phase_correction import phase_correction; help(phase_correction)"
```

### tests/

Contains testing framework for validating implementations:

```bash
# List test files
ls -la tests/

# Run quick tests without pytest
python tests/run_tests.py

# Run comprehensive pytest-based tests (using the exact path to avoid environment issues)
mppca-env/bin/pytest tests/test_implementations.py -v
```

## Fixed vs Improved MP-PCA

The package includes two primary patch-based implementations: `fixed_mppca` and `improved_mppca`. 

### Key Differences

1. **Noise Threshold Method**:
   - **fixed_mppca**: Uses a simple robust noise estimation with `noise_threshold = np.median(S) / 0.6745`
   - **improved_mppca**: Uses the Marchenko-Pastur distribution theory to calculate the optimal threshold through the `mp_threshold()` function

2. **Component Selection**:
   - **fixed_mppca**: Applies soft thresholding (subtracts the threshold and sets negative values to zero)
   - **improved_mppca**: Performs hard thresholding (zeros out components beyond the automatically determined threshold)

3. **Return Values**:
   - **fixed_mppca**: Returns only the denoised data
   - **improved_mppca**: Returns the denoised data, estimated noise level (sigma), and number of significant components

4. **Theoretical Foundation**:
   - **fixed_mppca**: Uses a simpler approach without leveraging the full Marchenko-Pastur theory
   - **improved_mppca**: More theoretically grounded in random matrix theory

5. **Performance**:
   - **fixed_mppca**: Generally faster but may not preserve fine details
   - **improved_mppca**: More computationally intensive but provides better signal preservation and noise removal

### Why improved_mppca is Recommended

The `improved_mppca` implementation is recommended for most use cases because:

1. **Superior noise removal**: Better separates noise from signal in low SNR regions
2. **Better detail preservation**: Preserves fine anatomical details that may be critical for Na-MRI analysis
3. **Adaptive component selection**: Automatically determines the optimal number of components based on local properties
4. **More robust estimation**: Provides more reliable noise estimates across different tissue types
5. **Additional metrics**: Returns useful diagnostic information (noise level and number of components)

In our benchmarks with sodium MRI data, `improved_mppca` consistently outperformed other methods, showing higher PSNR and better preservation of anatomical structures. This implementation was used for all results reported in our ISMRM 2025 abstract. While other implementations are provided for comparison and specific use cases, `improved_mppca` should be considered the default choice.

### Implementation Selection Guide

| Implementation | Use Case |
|----------------|----------|
| `improved_mppca` | Recommended for most Na-MRI denoising tasks |
| `fixed_mppca` | When computational speed is a priority |
| `custom_mppca` | When manual control over component selection is needed |
| `MPnonlocal` | For cases where non-local patch information improves results (Adapted from [NYU-DiffusionMRI/mppca_denoise](https://github.com/NYU-DiffusionMRI/mppca_denoise)) |
| `MP` | For simple 2D data or as a baseline comparison (Adapted from [NYU-DiffusionMRI/mppca_denoise](https://github.com/NYU-DiffusionMRI/mppca_denoise)) |

## Code Examples

### Recommended: Improved MP-PCA

The improved MP-PCA implementation is recommended for most applications:

```python
from mppca_implementations import improved_mppca
import numpy as np

# Create or load your 4D data
data_4d = np.random.rand(64, 64, 64, 4)  # [x, y, z, measurements]

# Apply improved MP-PCA denoising
denoised, sigma, npars = improved_mppca(data_4d, patch_radius=2)

print(f"Estimated noise level: {sigma}")
print(f"Number of signal components: {npars}")
```

### Alternative: Fixed MP-PCA

For faster processing when computational speed is a priority:

```python
from mppca_implementations import fixed_mppca
import numpy as np

# Create or load your 4D data
data_4d = np.random.rand(64, 64, 64, 4)  # [x, y, z, measurements]

# Apply fixed MP-PCA denoising
denoised = fixed_mppca(data_4d, patch_radius=1)

# Process the denoised data
print(f"Denoising completed with fixed component selection")
```

Key differences of `fixed_mppca`:
- Uses a fixed component selection strategy
- Generally faster than improved_mppca
- Does not return noise estimation (sigma) or component count (npars)
- Works well for consistent datasets where optimal component count is known

### Alternative: Basic MP-PCA

For simpler cases or 2D data:

```python
from mppca_implementations import MP
import numpy as np

# Create or load your data
data = np.random.rand(100, 20)  # Example data: 100 observations, 20 variables

# Apply MP-PCA denoising
denoised_data, sigma, npars = MP(data)
```

**Note:** The Basic MP-PCA (`MP`) implementation is adapted from the original algorithm by Veraart et al. at [NYU-DiffusionMRI/mppca_denoise](https://github.com/NYU-DiffusionMRI/mppca_denoise).

### Alternative: MP-PCA with Controlled Components

When you need to manually specify the number of components:

```python
from mppca_implementations import MP_PCA_controlled, custom_mppca
import numpy as np

# For 2D data
data_2d = np.random.rand(100, 20)
denoised_2d, sigma, npars, n_components = MP_PCA_controlled(data_2d, n_components=5)

# For 4D data (e.g., MRI volumes)
data_4d = np.random.rand(64, 64, 64, 4)  # [x, y, z, measurements]
denoised_4d = custom_mppca(data_4d, patch_radius=2, n_components=2)
```

### Alternative: Non-local MP-PCA

For cases where non-local patch information improves results:

```python
from mppca_implementations import MPnonlocal
import numpy as np

# Create or load your 4D data
data_4d = np.random.rand(64, 64, 64, 4)  # [x, y, z, measurements]

# Apply non-local MP-PCA denoising
denoised, sigma, npars, sigma_after = MPnonlocal(
    data_4d, 
    kernel=[5, 5, 5],          # 5x5x5 kernel
    patchtype='nonlocal',      # Use non-local patches
    patchsize=100,             # Number of similar patches to include
    shrink='threshold'         # Thresholding method
)
```

**Note:** The Non-local MP-PCA (`MPnonlocal`) implementation is adapted from the original MATLAB code by Veraart et al. at [NYU-DiffusionMRI/mppca_denoise](https://github.com/NYU-DiffusionMRI/mppca_denoise). This implementation provides advanced functionality for cases where spatial context beyond immediate neighbors can improve denoising results.

### Complex Data and Phase Correction

When working with complex data:

```python
from mppca_implementations import improved_mppca
from utils.phase_correction import phase_correction
import numpy as np

# Load complex data
complex_data = np.random.rand(64, 64, 64, 4) + 1j * np.random.rand(64, 64, 64, 4)

# Apply phase correction
corrected_data = np.zeros_like(complex_data)
for i in range(complex_data.shape[3]):
    corrected_data[:, :, :, i] = phase_correction(complex_data[:, :, :, i])

# Denoise real and imaginary parts separately
denoised_real = improved_mppca(np.real(corrected_data), patch_radius=2)[0]
denoised_imag = improved_mppca(np.imag(corrected_data), patch_radius=2)[0]

# Combine real and imaginary parts
denoised_complex = denoised_real + 1j * denoised_imag
```

## Running the Examples

The examples directory contains scripts that demonstrate how to use the library with various types of data. These are meant as starting points for your own analysis.

### Basic Usage Example

This script demonstrates all the MP-PCA implementations on synthetic data:

```bash
# Make sure you've activated your virtual environment
# and installed the package (see Installation section)

# Run the basic usage example
python examples/basic_usage.py
```

The script will:
1. Generate synthetic 3D+measurements data with noise
2. Apply the improved MP-PCA algorithm (recommended implementation)
3. Apply other implementations for comparison
4. Visualize the results side-by-side

### Complex Data Example

This script demonstrates how to work with complex data, including phase correction:

```bash
# Run the complex data example
python examples/complex_data_example.py
```

The script will:
1. Generate synthetic complex data
2. Apply phase correction to handle phase inconsistencies
3. Denoise the real and imaginary parts separately
4. Recombine the components and visualize the results

## Testing

The package includes a comprehensive testing framework to ensure all implementations work correctly. There are two ways to run tests:

### 1. Simple Test Runner (No pytest required)

Use the built-in test script that verifies all implementations:

```bash
# Make sure you're in the repository root directory
# and the virtual environment is activated

# Run the automated test script
python tests/run_tests.py
```

This script will:
- Generate synthetic test data
- Test each implementation with appropriate inputs
- Report success/failure for each implementation
- Show output shapes and additional outputs

Example output:
```
Generating test data...

Testing all implementations:
==================================================

Testing: Basic MP - Basic 2D implementation
✅ SUCCESS: Basic MP works properly
   Output shape: (4096, 4), Additional outputs: 2

...
```

### 2. Pytest-based Unit Tests

For more comprehensive testing:

```bash
# Make sure pytest is installed (it should be if you installed requirements.txt)
# Use the full path to pytest in the virtual environment to avoid PATH issues
./mppca-env/bin/pytest tests/test_implementations.py -v

# Run with coverage reporting (first install pytest-cov)
pip install pytest-cov
./mppca-env/bin/pytest --cov=mppca_implementations tests/
```

These tests verify that all implementations produce correct outputs with the expected dimensions and structure.

> **Note about pytest:** If you encounter the error "No module named pytest", use the full path to the pytest executable in your virtual environment as shown above. This ensures the correct pytest is used.

## Jupyter Notebooks

The notebooks directory contains more detailed examples with visualizations:

```bash
# Install jupyter if needed
pip install jupyter

# Start the notebook server
jupyter notebook notebooks/
```

Available notebooks:
- `complex_mppca_auto.ipynb`: Demonstrates automatic component selection with complex data
- `complex_mppca_manual.ipynb`: Shows manual component selection and parameter tuning

## Theory

### MP-PCA Denoising

The MP-PCA algorithm uses random matrix theory to separate signal from noise in the PCA domain. The key steps are:

1. **Patch extraction**: Extract local patches from the data
2. **PCA decomposition**: Perform SVD on the patch data
3. **Eigenvalue thresholding**: Determine the threshold between signal and noise eigenvalues using the Marchenko-Pastur distribution
4. **Component selection**: Keep only significant components
5. **Signal reconstruction**: Reconstruct the denoised signal

### Improvements in the Recommended Implementation

The improved MP-PCA implementation enhances the basic algorithm with:

1. **Overlapping patches**: Processes overlapping regions for smoother results
2. **Automatic component selection**: Determines the optimal number of components
3. **Patch-based processing**: Adapts to local signal properties
4. **Efficient implementation**: Optimized for performance with large datasets

## References

- Veraart, J., Novikov, D. S., Christiaens, D., Ades-Aron, B., Sijbers, J., & Fieremans, E. (2016). Denoising of diffusion MRI using random matrix theory. NeuroImage, 142, 394-406.
- Veraart, J., Fieremans, E., & Novikov, D. S. (2016). Diffusion MRI noise mapping using random matrix theory. Magnetic Resonance in Medicine, 76(5), 1582-1593.
- [NYU-DiffusionMRI/mppca_denoise](https://github.com/NYU-DiffusionMRI/mppca_denoise): Original MATLAB and Python implementations by Jelle Veraart and Benjamin Ades-Aron.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- NYU Langone Center for Advanced Imaging Innovation and Research
- Original MATLAB and Python implementations by Jelle Veraart and Benjamin Ades-Aron at the NYU Diffusion MRI lab
- This repository builds upon and extends the theoretical and practical foundations established in the [NYU-DiffusionMRI/mppca_denoise](https://github.com/NYU-DiffusionMRI/mppca_denoise) repository