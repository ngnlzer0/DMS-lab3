import numpy as np


def apply_dwt_2d_multilevel(img_padded, levels):
    """Багаторівневе пряме DWT Хаара"""
    coeffs = img_padded.astype(np.float64).copy()
    h, w = coeffs.shape

    for _ in range(levels):
        avg_rows = (coeffs[:h, :w:2] + coeffs[:h, 1:w:2]) / np.sqrt(2)
        det_rows = (coeffs[:h, :w:2] - coeffs[:h, 1:w:2]) / np.sqrt(2)
        coeffs[:h, :w] = np.hstack((avg_rows, det_rows))

        avg_cols = (coeffs[:h:2, :w] + coeffs[1:h:2, :w]) / np.sqrt(2)
        det_cols = (coeffs[:h:2, :w] - coeffs[1:h:2, :w]) / np.sqrt(2)
        coeffs[:h, :w] = np.vstack((avg_cols, det_cols))

        h, w = h // 2, w // 2

    return coeffs


def apply_idwt_2d_multilevel(coeffs, levels, original_shape):
    """Багаторівневе обернене DWT Хаара"""
    img_recon = coeffs.copy()
    H, W = img_recon.shape
    h, w = H // (2 ** (levels - 1)), W // (2 ** (levels - 1))

    for _ in range(levels):
        temp = np.zeros((h, w))
        avg_cols, det_cols = img_recon[:h // 2, :w], img_recon[h // 2:h, :w]
        temp[0::2, :] = (avg_cols + det_cols) / np.sqrt(2)
        temp[1::2, :] = (avg_cols - det_cols) / np.sqrt(2)
        img_recon[:h, :w] = temp

        temp = np.zeros((h, w))
        avg_rows, det_rows = img_recon[:h, :w // 2], img_recon[:h, w // 2:w]
        temp[:, 0::2] = (avg_rows + det_rows) / np.sqrt(2)
        temp[:, 1::2] = (avg_rows - det_rows) / np.sqrt(2)
        img_recon[:h, :w] = temp

        h, w = h * 2, w * 2

    img_recon = img_recon[:original_shape[0], :original_shape[1]]
    return np.clip(img_recon, 0, 255).astype(np.uint8)