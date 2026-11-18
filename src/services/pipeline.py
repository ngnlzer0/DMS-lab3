import time
from src.core.image_utils import pad_image
from src.core.metrics import calculate_metrics
from src.core.compressor import prune_coefficients
from src.algorithms.dct import apply_dct, apply_idct
from src.algorithms.dwt import apply_dwt_2d_multilevel, apply_idwt_2d_multilevel


def run_compression_pipeline(img, keep_pct, dwt_levels):
    t0 = time.time()

    img_padded, orig_shape = pad_image(img, dwt_levels)
    N = img_padded.size

    # --- DCT Pipeline ---
    t_dct_start = time.time()
    dct_coeffs = apply_dct(img_padded)
    dct_pruned, dct_k = prune_coefficients(dct_coeffs, keep_pct)
    recon_dct = apply_idct(dct_pruned, orig_shape)
    t_dct = time.time() - t_dct_start

    # Отримуємо рівно 3 значення
    psnr_dct, ssim_dct, size_dct = calculate_metrics(img, recon_dct, dct_pruned, dct_k, N)

    # --- DWT Pipeline ---
    t_dwt_start = time.time()
    dwt_coeffs = apply_dwt_2d_multilevel(img_padded, dwt_levels)
    dwt_pruned, dwt_k = prune_coefficients(dwt_coeffs, keep_pct)
    recon_dwt = apply_idwt_2d_multilevel(dwt_pruned, dwt_levels, orig_shape)
    t_dwt = time.time() - t_dwt_start

    # Отримуємо рівно 3 значення
    psnr_dwt, ssim_dwt, size_dwt = calculate_metrics(img, recon_dwt, dwt_pruned, dwt_k, N)

    return {
        'Pct': keep_pct, 'N': N,
        'DCT_K': dct_k, 'DCT_PSNR': psnr_dct, 'DCT_SSIM': ssim_dct, 'DCT_SizeKB': size_dct, 'DCT_Time': t_dct,
        'Recon_DCT': recon_dct,
        'DWT_K': dwt_k, 'DWT_PSNR': psnr_dwt, 'DWT_SSIM': ssim_dwt, 'DWT_SizeKB': size_dwt, 'DWT_Time': t_dwt,
        'Recon_DWT': recon_dwt,
        'Total_Time': time.time() - t0
    }