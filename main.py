import argparse
import cv2
import os
import sys
import logging
from src.services.pipeline import run_compression_pipeline
from src.services.visualization import save_comparison_plot

# Налаштування логування (професійний вивід замість звичайних print)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def validate_inputs(args):
    """Валідація вхідних параметрів перед початком математичних обчислень"""
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Файл '{args.input}' не знайдено або шлях вказано невірно.")

    if not (0 < args.keep <= 100):
        raise ValueError(f"Відсоток коефіцієнтів має бути в діапазоні (0; 100]. Отримано: {args.keep}")

    if not (1 <= args.levels <= 8):
        raise ValueError(f"Кількість рівнів DWT має бути від 1 до 8. Отримано: {args.levels}")


def main():
    parser = argparse.ArgumentParser(description="Image Compression CLI (DCT vs DWT)")
    parser.add_argument("input", help="Шлях до вхідного зображення (наприклад, image.png)")
    parser.add_argument("-l", "--levels", type=int, default=3, help="Кількість рівнів DWT (1-8)")
    parser.add_argument("-k", "--keep", type=float, default=20.0, help="Відсоток коефіцієнтів для збереження")
    parser.add_argument("-o", "--outdir", default="experiments", help="Папка для результатів")

    args = parser.parse_args()

    try:
        # 1. Валідація аргументів
        validate_inputs(args)

        # 2. Безпечне читання зображення
        img_gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Не вдалося прочитати зображення '{args.input}'. Перевірте формат файлу.")

        # 3. Перевірка розміру (захист від нестачі пам'яті)
        h, w = img_gray.shape
        max_pixels = 2_000_000  # ~ 1920x1080
        if h * w > max_pixels:
            logging.warning(f"Зображення дуже велике ({w}x{h}). Обробка може зайняти багато пам'яті та часу.")

        # 4. Створення директорії з перевіркою прав доступу
        try:
            os.makedirs(args.outdir, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Немає прав на створення папки або запис у '{args.outdir}'.")

        img_name = os.path.splitext(os.path.basename(args.input))[0]

        logging.info(f"Початок обробки: '{args.input}' | Рівні DWT: {args.levels} | Збережено: {args.keep}%")

        # 5. Виклик бізнес-логіки
        res = run_compression_pipeline(img_gray, args.keep, args.levels)

        logging.info("Обробку завершено!")
        print("-" * 50)
        print(f"Час виконання : {res['Total_Time']:.3f} сек")
        print("-" * 50)
        print(f"DCT PSNR      : {res['DCT_PSNR']:.3f} dB | SSIM: {res['DCT_SSIM']:.4f}")
        print(f"DCT Розмір    : ~{res['DCT_SizeKB']:.1f} KB (оцінка за ентропією)")
        print("-" * 50)
        print(f"DWT PSNR      : {res['DWT_PSNR']:.3f} dB | SSIM: {res['DWT_SSIM']:.4f}")
        print(f"DWT Розмір    : ~{res['DWT_SizeKB']:.1f} KB (оцінка за ентропією)")
        print("-" * 50)

        # 6. Збереження результатів
        cv2.imwrite(os.path.join(args.outdir, f"{img_name}_dct_{args.keep}pct.png"), res['Recon_DCT'])
        cv2.imwrite(os.path.join(args.outdir, f"{img_name}_dwt_{args.keep}pct.png"), res['Recon_DWT'])

        plot_path = os.path.join(args.outdir, f"{img_name}_compare_{args.keep}pct.png")
        save_comparison_plot(
            img_gray, res['Recon_DCT'], res['Recon_DWT'],
            res['DCT_PSNR'], res['DWT_PSNR'], args.keep, plot_path
        )

        logging.info(f"Файли успішно збережено у директорію '{args.outdir}/'.")

    # --- Обробка всіх можливих помилок ---
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Помилка вхідних даних: {e}")
        sys.exit(1)
    except PermissionError as e:
        logging.error(f"Помилка доступу: {e}")
        sys.exit(1)
    except MemoryError:
        logging.error("Недостатньо оперативної пам'яті для обробки цього зображення. Спробуйте зменшити його розмір.")
        sys.exit(1)
    except Exception as e:
        # Відловлювання непередбачуваних критичних помилок (наприклад, збоїв всередині NumPy)
        logging.critical(f"Виникла непередбачувана помилка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()