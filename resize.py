import argparse
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import filedialog

from PIL import Image


TARGET_SIZE = (1920, 1920)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_images(input_dir: Path, output_dir: Path, rename_start: Optional[int] = None) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input harus berupa folder: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"Tidak ada file gambar yang didukung di folder: {input_dir}")
        return

    for index, image_path in enumerate(image_files):
        if rename_start is not None:
            output_name = f"{rename_start + index}.jpg"
        else:
            output_name = f"{image_path.stem}.jpg"

        output_path = output_dir / output_name
        try:
            with Image.open(image_path) as image:
                resized = image.resize(TARGET_SIZE, Image.Resampling.BICUBIC)
                if resized.mode != "RGB":
                    resized = resized.convert("RGB")
                resized.save(output_path, format="JPEG", quality=95)
            print(f"Berhasil: {image_path.name} -> {output_path}")
        except Exception as exc:
            print(f"Gagal memproses {image_path.name}: {exc}")


def pick_folder(title: str) -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(title=title)
    root.destroy()

    if not selected:
        raise ValueError(f"Pemilihan folder dibatalkan: {title}")

    return Path(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize semua gambar dalam folder menjadi 1920x1920 dengan bicubic."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        nargs="?",
        default=None,
        help="Folder sumber gambar. Jika tidak diisi, akan muncul dialog pilih folder.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        nargs="?",
        default=None,
        help="Folder hasil resize. Jika tidak diisi, akan muncul dialog pilih folder.",
    )
    parser.add_argument(
        "--rename-start",
        type=int,
        default=None,
        help="Nomor awal untuk rename berurutan, misalnya 1 akan menghasilkan 1.jpg, 2.jpg, dst.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_folder or pick_folder("Pilih folder input gambar")
    output_dir = args.output_folder or pick_folder("Pilih folder output hasil resize")
    resize_images(input_dir, output_dir, rename_start=args.rename_start)


if __name__ == "__main__":
    main()
