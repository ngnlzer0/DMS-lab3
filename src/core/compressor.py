import numpy as np


def prune_coefficients(coeffs, percentage):
    """Відсікання найменших коефіцієнтів (залишаємо top-K %)"""
    if percentage >= 100:
        return coeffs, coeffs.size

    N = coeffs.size
    K = int(round(N * (percentage / 100.0)))
    if K == 0:
        K = 1

    # Лінійний алгоритм пошуку порогу
    abs_coeffs = np.abs(coeffs).flatten()
    threshold = np.partition(abs_coeffs, -K)[-K]

    mask = np.abs(coeffs) >= threshold
    pruned = np.where(mask, coeffs, 0)

    actual_k = np.count_nonzero(pruned)
    return pruned, actual_k