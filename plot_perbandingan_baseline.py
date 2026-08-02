from __future__ import annotations
"""
plot_perbandingan_baseline.py - Plot perbandingan grafik dari beberapa model baseline.

Menghasilkan 3 grafik perbandingan dalam satu image per metrik:
  1. Perbandingan Accuracy (Train & Validation) semua model
  2. Perbandingan mAP@0.50 (Train & Validation) semua model
  3. Perbandingan Loss (Train & Validation) semua model

Warna per model:
  - CNN       : Hijau tua (train) & Hijau muda (val)
  - ResNet    : Merah (train) & Kuning/Oranye (val)
  - VGG16     : Hitam (train) & Abu-abu (val)
  - Hybrid    : Biru tua dongker (train) & Biru muda (val)

Penggunaan:
  python plot_perbandingan_baseline.py

Output:
  Di folder outputs/perbandingan_baseline/:
    - perbandingan_accuracy.png
    - perbandingan_mAP50.png
    - perbandingan_loss.png
"""

import re
import matplotlib
matplotlib.use('Agg')  # Nonaktifkan GUI backend — simpan langsung ke file
import matplotlib.pyplot as plt
import os
import glob



def _apply_style():
    """Terapkan style darkgrid dengan fallback untuk semua versi matplotlib."""
    for s in ('seaborn-v0_8-darkgrid', 'seaborn-darkgrid'):
        try:
            plt.style.use(s)
            return
        except OSError:
            pass
    # Fallback manual jika seaborn tidak tersedia sama sekali
    plt.rcParams.update({
        'axes.facecolor': '#eaeaf2',
        'axes.edgecolor': 'white',
        'axes.grid': True,
        'grid.color': 'white',
        'grid.linewidth': 1.0,
    })


# ============================================================
# KONFIGURASI WARNA - Palet warna yang tersedia untuk model
# ============================================================
COLOR_PALETTES = [
    {
        'label': 'Hijau tua & Hijau muda',
        'color_train': '#006400',    # Hijau tua (Dark Green)
        'color_val':   '#90EE90',    # Hijau muda (Light Green)
    },
    {
        'label': 'Merah & Kuning',
        'color_train': '#CC0000',    # Merah
        'color_val':   '#FFD700',    # Kuning
    },
    {
        'label': 'Hitam & Abu-abu',
        'color_train': '#000000',    # Hitam
        'color_val':   '#A0A0A0',    # Abu-abu
    },
    {
        'label': 'Biru tua dongker & Biru muda',
        'color_train': '#00008B',    # Biru tua dongker (Dark Navy Blue)
        'color_val':   '#87CEEB',    # Biru muda (Light Blue / Sky Blue)
    },
    {
        'label': 'Ungu & Magenta',
        'color_train': '#4B0082',    # Ungu tua (Indigo)
        'color_val':   '#FF69B4',    # Magenta / Pink
    },
    {
        'label': 'Coklat & Oranye',
        'color_train': '#8B4513',    # Coklat tua (Saddle Brown)
        'color_val':   '#FFA500',    # Oranye
    },
]

OUTPUT_BASE = "outputs"
OUTPUT_FOLDER = "perbandingan_baseline"


# ============================================================
# FUNGSI UTILITAS
# ============================================================

def smooth_curve(points, factor=0.75):
    """Smoothing eksponensial untuk kurva."""
    smoothed = []
    for point in points:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append(prev * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed


def parse_logs(log_dir):
    """
    Parse semua file log di direktori.

    Returns:
        epoch_data: dict[epoch] -> {
            'train_loss', 'val_loss', 'val_acc', 'train_acc',
            'mAP50', 'train_mAP50'
        }
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))
    if not log_files:
        print(f"  [WARN] Tidak ada file log ditemukan di '{log_dir}'.")
        return None

    epoch_data = {}
    current_eval_epoch = None
    in_train_block = False

    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 1. Parsing epoch summary line (berisi loss, mAP, accuracy)
                match_epoch = re.search(
                    r'Epoch\s+(\d+)/\d+.*'
                    r'TrainLoss(?:Cls)?=([\d.]+).*'
                    r'ValLoss(?:Cls)?=([\d.]+).*'
                    r'mAP50=([\d.]+).*'
                    r'AvgAccMultiLabel=([\d.]+)',
                    line
                )
                if match_epoch:
                    epoch = int(match_epoch.group(1))
                    if epoch not in epoch_data:
                        epoch_data[epoch] = {}
                    epoch_data[epoch]['train_loss'] = float(match_epoch.group(2))
                    epoch_data[epoch]['val_loss'] = float(match_epoch.group(3))
                    epoch_data[epoch]['mAP50'] = float(match_epoch.group(4))
                    epoch_data[epoch]['val_acc'] = float(match_epoch.group(5))
                    continue

                # 2. Parsing Evaluation Block Epoch
                match_eval = re.search(r'METRIK MULTI-LABEL\s+\|\s+Epoch\s+(\d+)', line)
                if match_eval:
                    current_eval_epoch = int(match_eval.group(1))
                    in_train_block = False
                    continue

                # 3. Masuk ke blok TRAIN MULTI-LABEL
                if current_eval_epoch and 'TRAIN MULTI-LABEL' in line:
                    in_train_block = True
                    continue

                # 4. Ambil metrik dari dalam blok TRAIN MULTI-LABEL
                if current_eval_epoch and in_train_block:
                    # 4a. Train Accuracy
                    if '| Accuracy ' in line:
                        vals = re.findall(r'\|\s*([\d.]+)\s*', line)
                        if vals:
                            if current_eval_epoch not in epoch_data:
                                epoch_data[current_eval_epoch] = {}
                            epoch_data[current_eval_epoch]['train_acc'] = float(vals[3]) if len(vals) >= 4 else float(vals[-1])

                    # 4b. Train mAP@0.50
                    if '| mAP@0.50 ' in line:
                        vals = re.findall(r'\|\s*([\d.]+)\s*', line)
                        if vals:
                            if current_eval_epoch not in epoch_data:
                                epoch_data[current_eval_epoch] = {}
                            epoch_data[current_eval_epoch]['train_mAP50'] = float(vals[3]) if len(vals) >= 4 else float(vals[-1])
                            in_train_block = False
                            current_eval_epoch = None

    return epoch_data


# ============================================================
# FUNGSI PLOT PERBANDINGAN
# ============================================================

def plot_comparison_accuracy(all_model_data, output_dir):
    """Plot grafik perbandingan Accuracy semua model dalam satu gambar."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(18, 6))

    has_data = False

    for model_info, epoch_data in all_model_data:
        if epoch_data is None:
            continue

        name = model_info['name']
        color_train = model_info['color_train']
        color_val = model_info['color_val']
        ls_train = model_info['linestyle_train']
        ls_val = model_info['linestyle_val']

        # Val accuracy
        val_epochs = sorted([e for e in epoch_data if 'val_acc' in epoch_data[e]])
        val_accuracies = [epoch_data[e]['val_acc'] for e in val_epochs]


        # Train accuracy
        train_epochs = sorted([e for e in epoch_data if 'train_acc' in epoch_data[e]])
        train_accuracies = [epoch_data[e]['train_acc'] for e in train_epochs]

        if not val_epochs:
            print(f"  [SKIP] {name}: Data accuracy kosong.")
            continue

        has_data = True

        # Tambahkan titik awal untuk Train
        if train_epochs and train_epochs[0] > 1 and val_epochs:
            train_epochs.insert(0, val_epochs[0])
            train_accuracies.insert(0, val_accuracies[0])

        # Smoothing
        val_acc_smooth = smooth_curve(val_accuracies, factor=0.75)
        train_acc_smooth = smooth_curve(train_accuracies, factor=0.3) if train_accuracies else []

        # Plot
        ax.plot(val_epochs, val_acc_smooth,
                label=f'{name} - Val Accuracy',
                color=color_val, linewidth=2.0, linestyle=ls_val)
        if train_accuracies:
            ax.plot(train_epochs, train_acc_smooth,
                    label=f'{name} - Train Accuracy',
                    color=color_train, linewidth=2.0, linestyle=ls_train)

    if not has_data:
        print("  [SKIP] Tidak ada data accuracy untuk diplot.")
        plt.close()
        return

    ax.set_title('Perbandingan Training & Validation Accuracy - Semua Model Baseline',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(top=1.0)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=9, loc='lower right', frameon=True, shadow=True, ncol=2)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'perbandingan_accuracy.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik Perbandingan Accuracy disimpan: {output_path}")


def plot_comparison_mAP50(all_model_data, output_dir):
    """Plot grafik perbandingan mAP@0.50 semua model dalam satu gambar."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(18, 6))

    has_data = False

    for model_info, epoch_data in all_model_data:
        if epoch_data is None:
            continue

        name = model_info['name']
        color_train = model_info['color_train']
        color_val = model_info['color_val']
        ls_train = model_info['linestyle_train']
        ls_val = model_info['linestyle_val']

        # Val mAP
        val_epochs = sorted([e for e in epoch_data if 'mAP50' in epoch_data[e]])
        val_mAP = [epoch_data[e]['mAP50'] for e in val_epochs]

        # Train mAP
        train_epochs = sorted([e for e in epoch_data if 'train_mAP50' in epoch_data[e]])
        train_mAP = [epoch_data[e]['train_mAP50'] for e in train_epochs]

        if not val_epochs:
            print(f"  [SKIP] {name}: Data mAP@0.50 kosong.")
            continue

        has_data = True

        # Tambahkan titik awal untuk Train
        if train_epochs and train_epochs[0] > 1 and val_epochs:
            train_epochs.insert(0, val_epochs[0])
            train_mAP.insert(0, val_mAP[0])

        # Smoothing
        val_mAP_smooth = smooth_curve(val_mAP, factor=0.75)
        train_mAP_smooth = smooth_curve(train_mAP, factor=0.3) if train_mAP else []

        # Plot
        ax.plot(val_epochs, val_mAP_smooth,
                label=f'{name} - Val mAP@0.50',
                color=color_val, linewidth=2.0, linestyle=ls_val)
        if train_mAP:
            ax.plot(train_epochs, train_mAP_smooth,
                    label=f'{name} - Train mAP@0.50',
                    color=color_train, linewidth=2.0, linestyle=ls_train)

    if not has_data:
        print("  [SKIP] Tidak ada data mAP@0.50 untuk diplot.")
        plt.close()
        return

    ax.set_title('Perbandingan Training & Validation mAP@0.50 - Semua Model Baseline',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('mAP@0.50', fontsize=14, fontweight='bold')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=9, loc='lower right', frameon=True, shadow=True, ncol=2)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'perbandingan_mAP50.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik Perbandingan mAP@0.50 disimpan: {output_path}")


def plot_comparison_loss(all_model_data, output_dir):
    """Plot grafik perbandingan Loss semua model dalam satu gambar."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(18, 6))

    has_data = False

    for model_info, epoch_data in all_model_data:
        if epoch_data is None:
            continue

        name = model_info['name']
        color_train = model_info['color_train']
        color_val = model_info['color_val']
        ls_train = model_info['linestyle_train']
        ls_val = model_info['linestyle_val']

        # Loss epochs
        loss_epochs = sorted([e for e in epoch_data
                              if 'train_loss' in epoch_data[e] and 'val_loss' in epoch_data[e]])
        train_losses = [epoch_data[e]['train_loss'] for e in loss_epochs]
        val_losses = [epoch_data[e]['val_loss'] for e in loss_epochs]

        if not loss_epochs:
            print(f"  [SKIP] {name}: Data loss kosong.")
            continue

        has_data = True

        # Smoothing
        train_losses_smooth = smooth_curve(train_losses, factor=0.75)
        val_losses_smooth = smooth_curve(val_losses, factor=0.75)

        # Plot
        ax.plot(loss_epochs, train_losses_smooth,
                label=f'{name} - Train Loss',
                color=color_train, linewidth=2.0, linestyle=ls_train)
        ax.plot(loss_epochs, val_losses_smooth,
                label=f'{name} - Val Loss',
                color=color_val, linewidth=2.0, linestyle=ls_val)

    if not has_data:
        print("  [SKIP] Tidak ada data loss untuk diplot.")
        plt.close()
        return

    ax.set_title('Perbandingan Training & Validation Loss - Semua Model Baseline',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss Total', fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0.5, top=1.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=9, loc='upper right', frameon=True, shadow=True, ncol=2)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'perbandingan_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik Perbandingan Loss disimpan: {output_path}")


# ============================================================
# MAIN - INTERAKTIF
# ============================================================

def get_available_folders():
    """Dapatkan daftar folder output yang memiliki subfolder logs."""
    if not os.path.isdir(OUTPUT_BASE):
        return []
    return sorted([
        d for d in os.listdir(OUTPUT_BASE)
        if os.path.isdir(os.path.join(OUTPUT_BASE, d, "logs"))
    ])


def select_models_interactive():
    """Interaktif: pilih folder dan assign nama + warna untuk setiap model."""
    folders = get_available_folders()
    if not folders:
        print("[ERROR] Tidak ada folder output dengan subfolder logs.")
        return []

    print(f"\nFolder yang tersedia ({len(folders)}):")
    for i, f in enumerate(folders, 1):
        print(f"  {i:2d}. {f}")

    # Input jumlah model
    print()
    while True:
        try:
            n_models = int(input("Berapa model yang ingin dibandingkan? (2-6): ").strip())
            if 2 <= n_models <= 6:
                break
            print("  Masukkan angka antara 2-6.")
        except ValueError:
            print("  Input tidak valid, masukkan angka.")

    models = []

    for i in range(n_models):
        print(f"\n--- Model ke-{i+1} ---")

        # Pilih folder
        while True:
            try:
                idx = int(input(f"  Pilih nomor folder (1-{len(folders)}): ").strip())
                if 1 <= idx <= len(folders):
                    chosen_folder = folders[idx - 1]
                    break
                print(f"  Masukkan angka antara 1-{len(folders)}.")
            except ValueError:
                print("  Input tidak valid, masukkan angka.")

        print(f"  -> Folder dipilih: {chosen_folder}")

        # Input nama model
        default_name = chosen_folder.split('_', 3)[-1] if chosen_folder.count('_') >= 3 else chosen_folder
        name = input(f"  Nama model (default: {default_name}): ").strip()
        if not name:
            name = default_name

        # Pilih warna
        print(f"  Pilih warna:")
        for j, palette in enumerate(COLOR_PALETTES, 1):
            print(f"    {j}. {palette['label']}")

        while True:
            try:
                color_idx = int(input(f"  Pilih nomor warna (1-{len(COLOR_PALETTES)}): ").strip())
                if 1 <= color_idx <= len(COLOR_PALETTES):
                    chosen_palette = COLOR_PALETTES[color_idx - 1]
                    break
                print(f"  Masukkan angka antara 1-{len(COLOR_PALETTES)}.")
            except ValueError:
                print("  Input tidak valid, masukkan angka.")

        print(f"  -> Warna: {chosen_palette['label']}")

        models.append({
            'name': name,
            'folder': chosen_folder,
            'color_train': chosen_palette['color_train'],
            'color_val': chosen_palette['color_val'],
            'linestyle_train': '-',
            'linestyle_val': '--',
        })

    return models


def main():
    print("=" * 65)
    print("  PLOT PERBANDINGAN BASELINE - Accuracy, mAP@0.50, Loss")
    print("  (Interaktif: Pilih folder & warna untuk setiap model)")
    print("=" * 65)

    # Pilih model secara interaktif
    models = select_models_interactive()
    if not models:
        return

    # Buat folder output
    output_dir = os.path.join(OUTPUT_BASE, OUTPUT_FOLDER)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 65}")
    print(f"[INFO] Output akan disimpan di: {os.path.abspath(output_dir)}")
    print(f"[INFO] Jumlah model yang akan dibandingkan: {len(models)}")
    print("-" * 65)

    # Ringkasan model yang dipilih
    print(f"\n  Model yang akan dibandingkan:")
    for i, m in enumerate(models, 1):
        print(f"    {i}. {m['name']} -> {m['folder']}")

    # Parse log dari setiap model
    all_model_data = []

    for model in models:
        name = model['name']
        folder = model['folder']
        log_dir = os.path.join(OUTPUT_BASE, folder, "logs")

        print(f"\n[PARSE] {name} -> {folder}")

        if not os.path.isdir(log_dir):
            print(f"  [ERROR] Folder logs tidak ditemukan: {log_dir}")
            all_model_data.append((model, None))
            continue

        epoch_data = parse_logs(log_dir)
        if epoch_data:
            n_epochs = len(epoch_data)
            print(f"  [OK] Ditemukan data untuk {n_epochs} epoch")
        else:
            print(f"  [WARN] Tidak ada data yang dapat di-parse.")

        all_model_data.append((model, epoch_data))

    # Plot 3 grafik perbandingan
    print(f"\n{'=' * 65}")
    print(f"[PLOT] Membuat 3 grafik perbandingan (rasio 1:3)...")
    print("-" * 65)

    plot_comparison_accuracy(all_model_data, output_dir)
    plot_comparison_mAP50(all_model_data, output_dir)
    plot_comparison_loss(all_model_data, output_dir)

    print("-" * 65)
    print(f"\n[DONE] Selesai! 3 grafik perbandingan telah disimpan di folder:")
    print(f"  {os.path.abspath(output_dir)}")
    print(f"\n  File yang dihasilkan:")
    print(f"    1. perbandingan_accuracy.png")
    print(f"    2. perbandingan_mAP50.png")
    print(f"    3. perbandingan_loss.png")


if __name__ == "__main__":
    main()
