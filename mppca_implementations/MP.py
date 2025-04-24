import numpy as np
import matplotlib.pyplot as plt

def MP(X, nbins=0, centering=False):
    # MP: matrix denoising and noise estimation by exploiting data redundancy in the PCA domain
    # using universal properties of the eigenspectrum of random covariance matrices, i.e., Marchenko-Pastur distribution

    # Input:
    #   - X: [MxN] data matrix
    #   - nbins: number of histogram bins for visualization. If empty or not provided, no graphs will be shown.
    #   - centering: boolean flag indicating whether to center the data (default: False)

    # Output:
    #   - Xdn: [MxN] denoised data matrix
    #   - sigma: [1x1] noise level
    #   - npars: [1x1] number of significant components

    M, N = X.shape

    if centering:
        colmean = np.mean(X, axis=0)
        X = X - np.tile(colmean, (M, 1))

    R = min(M, N)
    scaling = np.ones(R - centering)
    if M > N:
        scaling = (M - np.arange(R - centering)) / N
        scaling[scaling < 1] = 1
        scaling = scaling.reshape(-1, 1)

    u, vals, v = np.linalg.svd(X, full_matrices=False)
    vals = vals**2 / N
    csum = np.cumsum(vals[R - centering - 1::-1])
    cmean = csum[R - centering - 1::-1] / np.arange(R - centering, 0, -1)
    sigmasq_1 = cmean / scaling.flatten()

    gamma = (M - np.arange(R - centering)) / N
    rangeMP = 4 * np.sqrt(gamma)
    rangeData = vals[:R - centering] - vals[R - centering - 1]
    sigmasq_2 = rangeData / rangeMP

    t = np.argmax(sigmasq_2 < sigmasq_1)
    sigma = np.sqrt(sigmasq_1[t])

    npars = t

    if nbins > 0:
        _, range_ = marchenko_pastur_distribution(np.random.rand(), sigma, M - npars, N)
        p, _ = marchenko_pastur_distribution(np.linspace(range_[0], range_[1], 100), sigma, M - npars, N)

        plt.figure()
        plt.gca().cla()
        range_ = [vals[R - centering - 1], vals[npars]]
        binwidth = np.diff(range_) / nbins
        scale = M * binwidth

        x = np.histogram(vals[:R - centering], bins=np.linspace(range_[0], range_[1], nbins))[0]

        plt.bar(np.linspace(range_[0], range_[1], nbins - 1), x / np.nansum(p))
        plt.plot(np.linspace(range_[0], range_[1], 100), np.real(p) * scale / np.nansum(p), 'r', linewidth=3)

        plt.xlabel('$\lambda$', fontname='Times', fontsize=20, interpreter='latex')
        plt.ylabel('$p(\lambda$)', fontname='Times', fontsize=20, interpreter='latex')
        plt.gca().set_fontsize(20)
        plt.gca().set_box(True)
        plt.gca().set_linewidth(2)
        plt.gca().set_fontsize(20)

        plt.title(f'sigma = {sigma:.2f} and npars = {npars}')

    vals[t:] = 0
    Xdn = u @ np.diag(np.sqrt(N * vals)) @ v

    if centering:
        Xdn = Xdn + np.tile(colmean, (M, 1))

    return Xdn, sigma, npars

def marchenko_pastur_distribution(lambda_, sigma, M, N):
    Q = M / N
    lambda_p = sigma**2 * (1 + np.sqrt(Q))**2
    lambda_m = sigma**2 * (1 - np.sqrt(Q))**2
    p = np.sqrt((lambda_p - lambda_) * (lambda_ - lambda_m)) / (2 * np.pi * Q * lambda_ * sigma**2)
    p[lambda_ < lambda_m] = 0
    p[lambda_ > lambda_p] = 0
    range_ = [lambda_m, lambda_p]
    return p, range_

# def marchenko_pastur_distribution(lambda_, sigma, M, N):
#     Q = M / N
    # lambda_p = sigma**2 * (1 + np.sqrt(Q))**2
    # lambda_m = sigma**2 * (1 - np.sqrt(Q))**2
    
    # # Check if lambda_ is within the valid range
    # if isinstance(lambda_, (int, float)):
    #     # lambda_ is a single value
    #     if lambda_m <= lambda_ <= lambda_p:
    #         p = np.sqrt((lambda_p - lambda_) * (lambda_ - lambda_m)) / (2 * np.pi * Q * lambda_ * sigma**2)
    #     else:
    #         p = 0.0
    # else:
    #     # lambda_ is an array
    #     valid_range = (lambda_ >= lambda_m) & (lambda_ <= lambda_p)
    #     p = np.zeros_like(lambda_)
    #     print(f'valid_range: {valid_range}')
    #     print(f'lambda_p: {lambda_p}, lambda_m: {lambda_m}, sigma: {sigma}')
    #     print(f'Q: {Q}, lambda_: {lambda_}, p: {p}')
    
    #     p[valid_range] = np.sqrt((lambda_p - lambda_[valid_range]) * (lambda_[valid_range] - lambda_m)) / (2 * np.pi * Q * lambda_[valid_range] * sigma**2)
    
    # range_ = [lambda_m, lambda_p]
    # return p, range_