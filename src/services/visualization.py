import matplotlib

matplotlib.use('Agg')  # Headless режим для Linux серверів
import matplotlib.pyplot as plt


def save_comparison_plot(orig, res_dct, res_dwt, dct_psnr, dwt_psnr, pct, filepath):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"Збережено коефіцієнтів: {pct}%", fontsize=14)

    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Оригінал")
    axes[0].axis('off')

    axes[1].imshow(res_dct, cmap='gray')
    axes[1].set_title(f"DCT (PSNR: {dct_psnr:.2f} dB)")
    axes[1].axis('off')

    axes[2].imshow(res_dwt, cmap='gray')
    axes[2].set_title(f"DWT (PSNR: {dwt_psnr:.2f} dB)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()