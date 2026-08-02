from __future__ import annotations
"""
plot_grafik.py - Plot 3 grafik sekaligus dari satu folder output.

Grafik yang dihasilkan:
  1. Training & Validation Average Accuracy
  2. Training & Validation mAP@0.50
  3. Training & Validation Total Loss

Penggunaan:
  python plot_grafik.py
  -> Masukkan nama folder output (contoh: run_20260609_043459_OPTIM_RMSProp)

Output:
  Di dalam folder yang sama:
    - accuracy_average.png
    - mAP50.png
    - loss_total.png
"""

import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import glob


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
        print(f"Error: Tidak ada file log ditemukan di '{log_dir}'.")
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

                # 2. Parsing Evaluation Block Epoch (biasanya setiap 10 epoch)
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
                            # Setelah mAP didapat, blok train selesai
                            in_train_block = False
                            current_eval_epoch = None

    return epoch_data


def plot_accuracy(epoch_data, output_dir):
    """Plot grafik Accuracy (Train & Validation)."""
    val_epochs = sorted([e for e in epoch_data if 'val_acc' in epoch_data[e]])
    val_accuracies = [epoch_data[e]['val_acc'] for e in val_epochs]

    train_epochs = sorted([e for e in epoch_data if 'train_acc' in epoch_data[e]])
    train_accuracies = [epoch_data[e]['train_acc'] for e in train_epochs]

    if not val_epochs:
        print("  [SKIP] Data validasi accuracy kosong.")
        return

    # Tambahkan titik awal untuk Train Accuracy
    if train_epochs and train_epochs[0] > 1 and val_epochs:
        train_epochs.insert(0, val_epochs[0])
        train_accuracies.insert(0, val_accuracies[0])

    # Best epoch
    best_idx = val_accuracies.index(max(val_accuracies))
    best_epoch = val_epochs[best_idx]
    best_acc = val_accuracies[best_idx]

    # Smoothing
    val_acc_smooth = smooth_curve(val_accuracies, factor=0.75)
    train_acc_smooth = smooth_curve(train_accuracies, factor=0.3) if train_accuracies else []

    # Plot
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 6))

    plt.plot(val_epochs, val_acc_smooth, label='Validation Avg Accuracy',
             color='#ff7f0e', linewidth=2.5)
    if train_accuracies:
        plt.plot(train_epochs, train_acc_smooth, label='Train Avg Accuracy',
                 color='#1f77b4', linewidth=2.5)

    plt.annotate(f'Best Epoch: {best_epoch}\nVal Acc: {best_acc:.4f}',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch, best_acc - 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold', color='red',
                 horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    plt.title('Training & Validation Average Accuracy', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(top=1.0)
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'accuracy_average.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik Accuracy disimpan: {output_path}")


def plot_mAP50(epoch_data, output_dir):
    """Plot grafik mAP@0.50 (Train & Validation)."""
    val_epochs = sorted([e for e in epoch_data if 'mAP50' in epoch_data[e]])
    val_mAP = [epoch_data[e]['mAP50'] for e in val_epochs]

    train_epochs = sorted([e for e in epoch_data if 'train_mAP50' in epoch_data[e]])
    train_mAP = [epoch_data[e]['train_mAP50'] for e in train_epochs]

    if not val_epochs:
        print("  [SKIP] Data mAP@0.50 kosong.")
        return

    # Tambahkan titik awal untuk Train mAP
    if train_epochs and train_epochs[0] > 1 and val_epochs:
        train_epochs.insert(0, val_epochs[0])
        train_mAP.insert(0, val_mAP[0])

    # Best epoch (berdasarkan Val mAP tertinggi)
    best_idx = val_mAP.index(max(val_mAP))
    best_epoch = val_epochs[best_idx]
    best_mAP = val_mAP[best_idx]

    # Smoothing
    val_mAP_smooth = smooth_curve(val_mAP, factor=0.75)
    train_mAP_smooth = smooth_curve(train_mAP, factor=0.3) if train_mAP else []

    # Plot
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 6))

    plt.plot(val_epochs, val_mAP_smooth, label='Validation mAP@0.50',
             color='#ff7f0e', linewidth=2.5)
    if train_mAP:
        plt.plot(train_epochs, train_mAP_smooth, label='Train mAP@0.50',
                 color='#1f77b4', linewidth=2.5)

    plt.annotate(f'Best Epoch: {best_epoch}\nmAP@0.50: {best_mAP:.4f}',
                 xy=(best_epoch, best_mAP),
                 xytext=(best_epoch, best_mAP + 0.20),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold', color='red',
                 horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    plt.title('Training & Validation mAP@0.50', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('mAP@0.50', fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.0)
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'mAP50.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik mAP@0.50 disimpan: {output_path}")


def plot_loss(epoch_data, output_dir):
    """Plot grafik Total Loss (Train & Validation)."""
    loss_epochs = sorted([e for e in epoch_data
                          if 'train_loss' in epoch_data[e] and 'val_loss' in epoch_data[e]])
    train_losses = [epoch_data[e]['train_loss'] for e in loss_epochs]
    val_losses = [epoch_data[e]['val_loss'] for e in loss_epochs]

    if not loss_epochs:
        print("  [SKIP] Data loss kosong.")
        return

    # Best epoch (val loss terendah)
    best_idx = val_losses.index(min(val_losses))
    best_epoch = loss_epochs[best_idx]
    best_val_loss = val_losses[best_idx]

    # Smoothing
    train_losses_smooth = smooth_curve(train_losses, factor=0.75)
    val_losses_smooth = smooth_curve(val_losses, factor=0.75)

    # Plot
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(18, 6))

    plt.plot(loss_epochs, train_losses_smooth, label='Train Loss',
             color='#1f77b4', linewidth=2.5)
    plt.plot(loss_epochs, val_losses_smooth, label='Validation Loss',
             color='#ff7f0e', linewidth=2.5)

    plt.annotate(f'Best Epoch: {best_epoch}\nVal Loss: {best_val_loss:.4f}',
                 xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch, best_val_loss + 0.3),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                 fontsize=12, fontweight='bold', color='red',
                 horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    plt.title('Training & Validation Loss', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Total', fontsize=14, fontweight='bold')
    plt.ylim(bottom=0.0)
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'loss_total.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik Loss disimpan: {output_path}")


def main():
    print("=" * 60)
    print("  PLOT GRAFIK - Accuracy, mAP@0.50, Loss")
    print("=" * 60)

    # Tampilkan daftar folder yang tersedia
    output_base = "outputs"
    if os.path.isdir(output_base):
        folders = sorted([
            d for d in os.listdir(output_base)
            if os.path.isdir(os.path.join(output_base, d, "logs"))
        ])
        if folders:
            print(f"\nFolder yang tersedia ({len(folders)}):")
            for i, f in enumerate(folders[-10:], 1):  # Tampilkan 10 terakhir
                print(f"  {i:2d}. {f}")
            if len(folders) > 10:
                print(f"  ... dan {len(folders) - 10} folder lainnya")

    # Input nama folder
    print()
    folder_name = input("Masukkan nama folder output: ").strip()

    # Cari folder
    run_dir = os.path.join(output_base, folder_name)
    if not os.path.isdir(run_dir):
        print(f"\nError: Folder '{run_dir}' tidak ditemukan!")
        return

    log_dir = os.path.join(run_dir, "logs")
    if not os.path.isdir(log_dir):
        print(f"\nError: Folder logs tidak ditemukan di '{log_dir}'!")
        return

    # Parse logs
    print(f"\n[PARSE] Membaca log dari: {log_dir}")
    epoch_data = parse_logs(log_dir)

    if not epoch_data:
        print("Tidak ada data yang dapat di-parse dari log.")
        return

    n_epochs = len(epoch_data)
    print(f"[INFO] Ditemukan data untuk {n_epochs} epoch")

    # Generate 3 grafik
    print(f"\n[PLOT] Membuat 3 grafik...")
    print(f"[OUTPUT] Disimpan di: {run_dir}")
    print("-" * 40)

    plot_accuracy(epoch_data, run_dir)
    plot_mAP50(epoch_data, run_dir)
    plot_loss(epoch_data, run_dir)

    print("-" * 40)
    print(f"\n[DONE] Selesai! 3 grafik telah disimpan di folder:")
    print(f"  {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
