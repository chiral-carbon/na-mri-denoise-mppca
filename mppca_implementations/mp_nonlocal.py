import numpy as np
from scipy.linalg import svd
# from scipy.spatial.distance import cdist
# from scipy.spatial import ConvexHull


def refine_patch(data, kernel, M, pos_img, coil):
    """
    Refine patch by selecting the most similar patches.
    
    Parameters:
    -----------
    data : array
        The normalized data array
    kernel : tuple
        Kernel dimensions
    M : int
        Number of patches to select
    pos_img : array
        Position image for weighting
    coil : bool
        Whether the data includes coil dimension
    
    Returns:
    --------
    min_idx : array
        Indices of the most similar patches
    """
    # Get the center index
    center_idx = np.prod(kernel) // 2
    
    # Check the shape of data to handle it properly
    if coil:
        # For data with coil dimension (multi-coil data)
        if data.ndim == 3:  # (kernel_size, coils, volumes)
            refval = data[center_idx, :, :]
            refval = np.tile(refval, (np.prod(kernel), 1, 1))
            int_img = (1 / (data.shape[1] * data.shape[2])) * np.sum((data - refval) ** 2, axis=(1, 2))
        else:
            # Handle error case
            raise ValueError(f"Unexpected data shape for coil=True: {data.shape}")
    else:
        # For data without coil dimension (single-coil data)
        if data.ndim == 2:  # (kernel_size, volumes)
            refval = data[center_idx, :]
            refval = np.tile(refval, (np.prod(kernel), 1))
            int_img = (1 / data.shape[1]) * np.sum((data - refval) ** 2, axis=1)
        else:
            # Handle error case
            raise ValueError(f"Unexpected data shape for coil=False: {data.shape}")

    wdists = pos_img * int_img
    min_idx = np.argpartition(wdists, M)[:M]

    return min_idx

def normalize(data):
    data_norm = np.zeros(data.shape)
    for i in range(data.shape[-1]):
        data_ = data[..., i]
        data_norm[..., i] = np.abs(data_ / np.max(data_))
    return data_norm

def get_voxel_coords(sx, sy, sz, kx, ky, kz):
    mask = np.ones((sx, sy, sz), dtype=bool)
    mask[0:kx, :, :] = False
    mask[:, 0:ky, :] = False
    mask[:, :, 0:kz] = False
    mask[sx - kx:sx, :, :] = False
    mask[:, sy - ky:sy, :] = False
    mask[:, :, sz - kz:sz] = False
    maskinds = np.where(mask)
    x, y, z = maskinds
    return x, y, z

def unpad(data, kernel):
    k = (np.array(kernel) - 1) // 2
    data = data[k[0]:-k[0], k[1]:-k[1], k[2]:-k[2], ...]
    return data

def shrink(y, gamma):
    # Frobenius norm optimal shrinkage
    t = 1 + np.sqrt(gamma)
    s = np.zeros_like(y)
    x = y[y > t]
    s[y > t] = np.sqrt((x ** 2 - gamma - 1) ** 2 - 4 * gamma) / x
    return s

def denoise(X, nrm, exp, tn):
    N = X.shape[1]
    M = X.shape[0]
    Mp = min(M, N)
    Np = max(M, N)

    if M < N:
        X = X.T

    # Compute PCA eigenvalues
    u, vals, v = svd(X, full_matrices=False)
    vals = vals ** 2

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    u = u[:, order]
    v = v[:, order]

    ptn = np.arange(Mp - tn)
    p = np.arange(Mp)
    csum = np.cumsum(vals[::-1])[::-1]

    if exp == 1:  # veraart 2016
        sigmasq_1 = csum / ((Mp - p) * Np)
        rangeMP = 4 * np.sqrt((Mp - ptn) * (Np - tn))
    elif exp == 2:  # cordero-grande
        sigmasq_1 = csum / ((Mp - p) * (Np - p))
        rangeMP = 4 * np.sqrt((Mp - ptn) * (Np - ptn))
    elif exp == 3:  # jespersen
        sigmasq_1 = csum / ((Mp - p) * (Np - p))
        rangeMP = 4 * np.sqrt((Np - tn) * Mp)

    rangeData = vals[:Mp - tn] - vals[Mp - tn - 1]
    sigmasq_2 = rangeData / rangeMP

    t = np.argmax(sigmasq_2 < sigmasq_1[:Mp-tn])

    if np.isnan(t):
        sigma = np.nan
        npars = np.nan
        s = X
        sigma_after = np.nan
    else:
        sigma = np.sqrt(sigmasq_1[t])
        npars = t
        if nrm == 'threshold':
            vals[t:] = 0
            s = u @ np.diag(np.sqrt(vals)) @ v.T
        elif nrm == 'frob':
            vals_frob = np.sqrt(Mp) * sigma * shrink(np.sqrt(vals) / (np.sqrt(Mp) * sigma), Np / Mp)
            s = u @ np.diag(vals_frob) @ v.T

        s2_after = sigma ** 2 - csum[t] / (Mp * Np)
        # print("s2_after = ", s2_after)
        sigma_after = np.sqrt(s2_after)

    if M < N:
        s = s.T

    return s, sigma, npars, sigma_after

def MPnonlocal(data, *args, **kwargs):
    if np.iscomplexobj(data):
        data = np.complex64(data)
    else:
        data = np.float32(data)

    if data.ndim > 4:
        coil = True
    else:
        coil = False

    default_shrink = 'threshold'
    default_exp = 1

    nvols = data.shape[-1]
    p_ = np.arange(1, 2 * nvols, 2)
    pf_ = np.argmax(p_ ** 3 >= nvols)
    default_kernel = p_[pf_]
    default_patchtype = 'box'
    default_patchsize = default_kernel ** 3
    default_crop = 0

    # parse input arguments
    kernel = kwargs.get('kernel', default_kernel)
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    kernel = (np.array(kernel) + ((np.array(kernel) % 2) - 1)).astype(int)

    patchtype = kwargs.get('patchtype', default_patchtype)
    patchsize = kwargs.get('patchsize', default_patchsize)
    shrink = kwargs.get('shrink', default_shrink)
    exp = kwargs.get('exp', default_exp)
    cropdist = kwargs.get('crop', default_crop)

    psize = np.prod(kernel)
    non_local = False
    center_idx = int(np.prod(kernel) // 2)
    pos_img = None

    if patchtype == 'nonlocal':
        if patchsize >= np.prod(kernel):
            print('Selecting sane default nonlocal patch size')
            psize = int(np.floor(np.prod(kernel) - 0.2 * np.prod(kernel)))
            if psize <= nvols:
                psize = nvols + 1
        else:
            psize = patchsize
        non_local = True
        center_idx = 1
    elif patchtype != 'box':
        raise ValueError('patchtype options are "box" or "nonlocal"')

    nrm = shrink
    cropdist = cropdist

    print('Denoising data using parameters:')
    print(f'kernel     = {kernel}')
    print(f'patch type = {patchtype}')
    print(f'patch size = {psize}')
    print(f'shrinkage  = {shrink}')
    print(f'algorithm  = {exp}')
    print(f'cropdist   = {cropdist}')

    k = (np.array(kernel) - 1) // 2
    kx, ky, kz = k

    if coil:
        data = np.pad(data, ((kx, kx), (ky, ky), (kz, kz), (0, 0), (0, 0)), mode='wrap')
        sx, sy, sz, sc, N = data.shape
        M = psize * sc
    else:
        data = np.pad(data, ((kx, kx), (ky, ky), (kz, kz), (0, 0)), mode='wrap')
        sx, sy, sz, N = data.shape
        M = psize
        sc = 1

    mask = np.full(data.shape[:3], True, dtype=bool)
    mask[kx:-kx, ky:-ky, kz:-kz] = False
    x, y, z = get_voxel_coords(sx, sy, sz, kx, ky, kz)

    if non_local:
        patchcoords = np.argwhere(np.ones(kernel))
        pos_img = (1 / np.prod(kernel)) * np.sum((patchcoords - np.ceil(np.array(kernel) / 2)) ** 2, axis=1)

    sigma = np.zeros(x.shape[0], dtype=data.dtype)
    sigma_after = np.zeros(x.shape[0], dtype=data.dtype)
    npars = np.zeros(x.shape[0], dtype=data.dtype)
    Sigma = np.zeros((sx, sy, sz), dtype=data.dtype)
    Sigma_after = np.zeros((sx, sy, sz), dtype=data.dtype)
    Npars = np.zeros((sx, sy, sz), dtype=data.dtype)

    if coil:
        signal = np.zeros((sc, N, x.shape[0]), dtype=data.dtype)
        Signal = np.zeros((sx, sy, sz, sc, N), dtype=data.dtype)
    else:
        signal = np.zeros((1, N, x.shape[0]), dtype=data.dtype)
        Signal = np.zeros((sx, sy, sz, N), dtype=data.dtype)

    
    for nn in range(x.shape[0]):
        X = data[x[nn] - kx:x[nn] + kx + 1, y[nn] - ky:y[nn] + ky + 1, z[nn] - kz:z[nn] + kz + 1, ...]
        if coil:
            X = np.reshape(X, (np.prod(kernel), sc, N))
        else:
            X = np.reshape(X, (np.prod(kernel), N))
        if non_local:
            Xn = normalize(X)
            # Print data shapes for debugging
            # print(f"Xn shape: {Xn.shape}")
            # print(f"kernel: {kernel}, psize: {psize}")
            # print(f"pos_img shape: {pos_img.shape if pos_img is not None else None}")
            try:
                min_idx = refine_patch(Xn, kernel, psize, pos_img, coil)
                X = X[min_idx, ...]
            except Exception as e:
                print(f"Error in refine_patch: {e}")
                # Fallback: use the first psize elements
                min_idx = np.arange(psize)
                X = X[min_idx, ...]

        X = np.reshape(X, (M, N))
        s, sigma[nn], npars[nn], sigma_after[nn] = denoise(X, nrm, exp, cropdist)
        if coil:
            signal[:, :, nn] = s[center_idx::psize, :]
        else:
            signal[:, :, nn] = s[center_idx, :]


    for nn in range(x.shape[0]):
        Sigma[x[nn], y[nn], z[nn]] = sigma[nn]
        Sigma_after[x[nn], y[nn], z[nn]] = sigma_after[nn]
        Npars[x[nn], y[nn], z[nn]] = npars[nn]
        if coil:
            Signal[x[nn], y[nn], z[nn], :, :] = signal[:, :, nn]
        else:
            Signal[x[nn], y[nn], z[nn], :] = signal[0, :, nn]

    Sigma = unpad(Sigma, kernel)
    Sigma_after = unpad(Sigma_after, kernel)
    Npars = unpad(Npars, kernel)
    Signal = unpad(Signal, kernel)

    return Signal, Sigma, Npars, Sigma_after

# You can call the function like this:
# Signal, Sigma, Npars, Sigma_after = MPnonlocal(data, kernel=(5, 5, 5), patchtype='box', shrink='threshold')


