import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import numpy as np

def MP_PCA_controlled(X, n_components=None, centering=False):
    M, N = X.shape
    
    flip = False
    if M < N:
        X = X.T
        M, N = N, M
        flip = True
    
    if centering:
        colmean = np.mean(X, axis=0)
        X = X - np.tile(colmean, (M, 1))
    
    R = min(M, N)
    
    u, vals, v = np.linalg.svd(X, full_matrices=False)
    vals = (vals**2) / N
    
    csum = np.cumsum(vals[R-centering-1::-1])
    cmean = csum[R-centering-1::-1] / np.arange(R-centering, 0, -1)
    sigmasq_1 = cmean
    
    gamma = (M - np.arange(R - centering)) / N
    rangeMP = 4 * np.sqrt(gamma)
    rangeData = vals[:R - centering] - vals[R - centering - 1]
    sigmasq_2 = rangeData / rangeMP
    
    t = np.argmax(sigmasq_2 < sigmasq_1)
    sigma = np.sqrt(sigmasq_1[t])
    
    npars = t
    
    # if n_components is None:
    #     n_components = t
    # else:
    #     n_components = min(n_components, M)
    
    # vals_reconstructed = np.zeros_like(vals)
    # vals_reconstructed[:n_components] = vals[:n_components]
    # Xdn = np.dot(u, np.dot(np.diag(np.sqrt(N * vals_reconstructed)), v))

    if n_components is None:
        vals[t:] = 0
    else:
        n_components = min(n_components, R)
        vals[n_components:] = 0
    
    Xdn = np.dot(u, np.dot(np.diag(np.sqrt(N * vals)), v))
    
    if flip:
        Xdn = Xdn.T
    
    if centering:
        Xdn = Xdn + np.tile(colmean, (M, 1))
    
    return Xdn, sigma, npars, n_components


def custom_mppca(data, patch_radius=2, n_components=None):
    from itertools import product
    
    shape = data.shape[:-1]
    N = data.shape[-1]
    
    print(f"Input shape: {data.shape}")
    
    pad_width = [(patch_radius, patch_radius)] * 3 + [(0, 0)]
    padded_data = np.pad(data, pad_width, mode='reflect')
    
    print(f"Padded shape: {padded_data.shape}")
    
    denoised_data = np.zeros_like(data)
    
    for x, y, z in product(range(shape[0]), range(shape[1]), range(shape[2])):
        patch = padded_data[
            x:x+2*patch_radius+1,
            y:y+2*patch_radius+1,
            z:z+2*patch_radius+1,
            :
        ]
        
        X = patch.reshape(-1, N)
        
        print(f"Patch shape: {patch.shape}, X shape: {X.shape}")
        
        Xdn, _, _, n_components = MP_PCA_controlled(X, n_components)
        
        print(f"Xdn shape: {Xdn.shape}")
        print(f"n_components: {n_components}")
        
        denoised_data[x, y, z, :] = Xdn.reshape(patch.shape)[patch_radius, patch_radius, patch_radius, :]
    
    return denoised_data

def psnr_metric(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(original)
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def iterative_denoising(data, patch_radius=2):
    results = []
    max_components = min(data.shape[-1], (2*patch_radius+1)**3)
    
    for n_components in range(1, max_components + 1):
        denoised = custom_mppca(data, patch_radius, n_components)
        psnr = psnr_metric(data, denoised)
        results.append((n_components, psnr))
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate some noisy data
    np.random.seed(0)
    X = np.random.randn(100, 20)
    X[:, :5] += np.random.randn(100, 5) * 5  # Add more variance to first 5 components

    results = iterative_denoising(X, psnr_metric)

    # Plot results
    components, metrics = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(components, metrics)
    plt.xlabel('Number of components')
    plt.ylabel('PSNR')
    plt.title('Denoising performance vs. number of components')
    plt.grid(True)
    plt.show()