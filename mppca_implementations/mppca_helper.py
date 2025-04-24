import numpy as np
from scipy import linalg
from tqdm import tqdm

def fixed_mppca(X, patch_radius=2, n_components=None):
    """
    Fixed implementation of MPPCA denoising.
    
    Parameters:
    -----------
    X : ndarray
        Input 4D array of shape [x, y, z, N]
    patch_radius : int
        Radius of patch for local PCA
    n_components : int
        Number of components to keep
    """
    sx, sy, sz, N = X.shape
    patch_size = 2 * patch_radius + 1
    
    # Initialize output
    Xdn = np.zeros_like(X)
    weights = np.zeros((sx, sy, sz))
    
    # Pad input array
    pad_width = ((patch_radius, patch_radius),
                 (patch_radius, patch_radius),
                 (patch_radius, patch_radius),
                 (0, 0))
    X_padded = np.pad(X, pad_width, mode='reflect')
    
    for x in range(patch_radius, sx + patch_radius):
        for y in range(patch_radius, sy + patch_radius):
            for z in range(patch_radius, sz + patch_radius):
                # Extract patch
                patch = X_padded[x-patch_radius:x+patch_radius+1,
                                y-patch_radius:y+patch_radius+1,
                                z-patch_radius:z+patch_radius+1]
                
                # Reshape patch to 2D matrix
                M = patch.reshape(-1, N)
                
                # Perform SVD
                U, S, Vh = linalg.svd(M, full_matrices=False)
                
                # Calculate noise threshold
                noise_threshold = np.median(S) / 0.6745  # Robust noise estimation
                
                if n_components is not None:
                    # Keep only specified number of components
                    S_filtered = np.zeros_like(S)
                    S_filtered[:n_components] = S[:n_components]
                    
                    # Apply soft thresholding to retained components
                    S_filtered = np.maximum(S_filtered - noise_threshold, 0)
                else:
                    # Automatic thresholding
                    S_filtered = np.maximum(S - noise_threshold, 0)
                
                # Reconstruct
                M_denoised = U @ np.diag(S_filtered) @ Vh
                
                # Reshape back to patch
                patch_denoised = M_denoised.reshape(patch_size, patch_size, patch_size, N)
                
                # Store central voxel
                Xdn[x-patch_radius, y-patch_radius, z-patch_radius] = \
                    patch_denoised[patch_radius, patch_radius, patch_radius]
                weights[x-patch_radius, y-patch_radius, z-patch_radius] += 1
    
    # Average overlapping patches
    Xdn = Xdn / weights[..., np.newaxis]
    
    return Xdn

def improved_mppca(X, patch_radius=2, n_components=None):
    """
    Marchenko-Pastur PCA denoising with automatic component selection.
    
    THIS IS THE RECOMMENDED IMPLEMENTATION for most denoising applications.
    It provides superior denoising quality by using overlapping patches and
    automatic optimal component selection.
    
    Parameters:
    -----------
    X : ndarray
        Input 4D array of shape [x, y, z, N] where N is number of measurements
    patch_radius : int
        Radius of patch for local PCA (default: 2)
    n_components : int, optional
        Force number of components. If None, automatically determined (recommended).
        
    Returns:
    --------
    Xdn : ndarray
        Denoised array of same shape as input
    sigma : float
        Estimated noise level
    n_components_used : int
        Number of components used for denoising
    """
    # Extract dimensions
    sx, sy, sz, N = X.shape
    patch_size = 2 * patch_radius + 1
    
    # Initialize output
    Xdn = np.zeros_like(X)
    weights = np.zeros((sx, sy, sz))
    
    # Pad input array
    pad_width = ((patch_radius, patch_radius),
                 (patch_radius, patch_radius),
                 (patch_radius, patch_radius),
                 (0, 0))
    X_padded = np.pad(X, pad_width, mode='reflect')
    
    # Process each voxel
    for x in tqdm(range(patch_radius, sx + patch_radius)):
        for y in range(patch_radius, sy + patch_radius):
            for z in range(patch_radius, sz + patch_radius):
                # Extract patch
                patch = X_padded[x-patch_radius:x+patch_radius+1,
                               y-patch_radius:y+patch_radius+1,
                               z-patch_radius:z+patch_radius+1]
                
                # Reshape patch to 2D matrix
                M = patch.reshape(-1, N)
                
                if n_components is None:
                    # Perform SVD
                    U, S, Vh = linalg.svd(M, full_matrices=False)
                    
                    # Marchenko-Pastur threshold
                    sigma, npars = mp_threshold(S, M.shape[0], N)
                    
                    # Zero out components beyond threshold
                    S[npars:] = 0
                    
                    # Reconstruct
                    M_denoised = U @ np.diag(S) @ Vh
                else:
                    # Use fixed number of components
                    U, S, Vh = linalg.svd(M, full_matrices=False)
                    S[n_components:] = 0
                    M_denoised = U @ np.diag(S) @ Vh
                    
                # Reshape back to patch
                patch_denoised = M_denoised.reshape(patch_size, patch_size, patch_size, N)
                
                # Store central voxel
                Xdn[x-patch_radius, y-patch_radius, z-patch_radius] = \
                    patch_denoised[patch_radius, patch_radius, patch_radius]
                weights[x-patch_radius, y-patch_radius, z-patch_radius] += 1
                
    # Average overlapping patches
    Xdn = Xdn / weights[..., np.newaxis]
    
    return Xdn, sigma if n_components is None else None, \
           npars if n_components is None else n_components

def mp_threshold(S, M, N):
    """
    Calculate Marchenko-Pastur threshold for eigenvalues.
    
    Parameters:
    -----------
    S : ndarray
        Singular values
    M : int
        Number of observations
    N : int
        Number of variables
        
    Returns:
    --------
    sigma : float
        Estimated noise level
    npars : int
        Number of significant components
    """
    # Convert to eigenvalues
    vals = S**2 / N
    
    # Get scaling factor
    scaling = np.ones(len(vals))
    if M > N:
        scaling = (M - np.arange(len(vals))) / N
        scaling[scaling < 1] = 1
    
    # Calculate sigmas
    csum = np.cumsum(vals[::-1])[::-1]
    cmean = csum / np.arange(len(vals), 0, -1)
    sigmasq_1 = cmean / scaling
    
    gamma = (M - np.arange(len(vals))) / N
    rangeMP = 4 * np.sqrt(gamma)
    rangeData = vals - vals[-1]
    sigmasq_2 = rangeData / rangeMP
    
    # Find threshold
    t = np.argmax(sigmasq_2 < sigmasq_1)
    sigma = np.sqrt(sigmasq_1[t])
    
    return sigma, t

def analyze_eigenvalues(X, patch_radius=2):
    """
    Analyze eigenvalue distribution for a sample of patches.
    
    Parameters:
    -----------
    X : ndarray
        Input 4D array
    patch_radius : int
        Radius of patch
        
    Returns:
    --------
    eigenvalues : ndarray
        Array of eigenvalues from sampled patches
    cutoffs : ndarray
        Array of cutoff values for each patch
    """
    patch_size = 2 * patch_radius + 1
    sx, sy, sz, N = X.shape
    
    # Sample patches
    n_samples = 100
    eigenvalues = []
    cutoffs = []
    
    for _ in range(n_samples):
        # Random location
        x = np.random.randint(patch_radius, sx - patch_radius)
        y = np.random.randint(patch_radius, sy - patch_radius)
        z = np.random.randint(patch_radius, sz - patch_radius)
        
        # Extract patch
        patch = X[x-patch_radius:x+patch_radius+1,
                 y-patch_radius:y+patch_radius+1,
                 z-patch_radius:z+patch_radius+1]
        
        # Reshape and perform SVD
        M = patch.reshape(-1, N)
        _, S, _ = linalg.svd(M, full_matrices=False)
        
        # Get threshold
        sigma, npars = mp_threshold(S, M.shape[0], N)
        
        eigenvalues.append(S**2 / N)
        cutoffs.append(npars)
        
    return np.array(eigenvalues), np.array(cutoffs) 