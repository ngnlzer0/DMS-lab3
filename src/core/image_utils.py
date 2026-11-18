import numpy as np


def pad_image(img, levels=1):
    """Доповнює зображення симетричним відображенням (Padding)"""
    h, w = img.shape
    modulus = max(8, 2 ** levels)
    pad_h = (modulus - h % modulus) % modulus
    pad_w = (modulus - w % modulus) % modulus

    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    return img_padded, (h, w)