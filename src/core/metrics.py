import numpy as np


def calculate_entropy(coeffs):
    """Обчислює ентропію Шеннона (бітів на піксель) для стиснутої матриці"""
    # Округлюємо коефіцієнти для підрахунку унікальних значень
    coeffs_int = np.round(coeffs).astype(int).flatten()
    _, counts = np.unique(coeffs_int, return_counts=True)
    probabilities = counts / counts.sum()
    # Формула Шеннона: H = - sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def calculate_ssim(img1, img2):
    """Глобальний індекс структурної подібності (SSIM)"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)

    mu1, mu2 = np.mean(i1), np.mean(i2)
    var1, var2 = np.var(i1), np.var(i2)
    covar = np.mean((i1 - mu1) * (i2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * covar + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2))
    return ssim


def calculate_metrics(orig, recon, pruned_coeffs, K, N):
    """Повертає PSNR, SSIM та теоретичний розмір (КБ)"""
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10((255 ** 2) / mse)

    ssim = calculate_ssim(orig, recon)
    entropy_bpp = calculate_entropy(pruned_coeffs)
    estimated_size_kb = (entropy_bpp * N) / (8 * 1024)  # Розмір у кілобайтах

    return psnr, ssim, estimated_size_kb