import numpy as np


def generate_dct_matrix(N=8):
    """Генерує матрицю прямого DCT 8x8"""
    C = np.zeros((N, N))
    for u in range(N):
        for x in range(N):
            alpha = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
            C[u, x] = alpha * np.cos(((2 * x + 1) * u * np.pi) / (2 * N))
    return C


def apply_dct(img_padded):
    """Виконує пряме блокове DCT (8x8) над зображенням"""
    C = generate_dct_matrix(8)
    img_shifted = img_padded.astype(np.float64) - 128.0
    h, w = img_shifted.shape

    blocks = img_shifted.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2)
    return C @ blocks @ C.T


def apply_idct(dct_blocks, original_shape):
    """Виконує обернене блокове DCT та знімає padding/shift"""
    C = generate_dct_matrix(8)
    h, w = original_shape

    idct_blocks = C.T @ dct_blocks @ C
    img_recon = idct_blocks.swapaxes(1, 2).reshape((dct_blocks.shape[0] * 8, dct_blocks.shape[1] * 8))
    img_recon = img_recon[:h, :w] + 128.0

    return np.clip(img_recon, 0, 255).astype(np.uint8)