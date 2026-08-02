from __future__ import annotations
"""
plot_grafik_wide.py - Plot 3 grafik sekaligus dari satu folder output.
Sama persis dengan plot_grafik.py versi asli, hanya rasio image diubah
menjadi 1:3 (lebih panjang horizontal).

Grafik yang dihasilkan:
  1. Training & Validation Average Accuracy
  2. Training & Validation mAP@0.50
  3. Training & Validation Total Loss

Sumber data (prioritas):
  1. Checkpoint file (latest_checkpoint.pth) — data train & val lengkap per epoch
  2. Log file (.log) — fallback jika checkpoint tidak tersedia

Penggunaan:
  python plot_grafik_wide.py
  -> Masukkan nama folder output (contoh: run_20260609_043459_OPTIM_RMSProp)

Output:
  Di dalam folder yang sama:
    - accuracy_average.png
    - mAP50.png
    - loss_total.png
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


# ═══════════════════════════════════════════════════════════════════
#  SUMBER DATA 1: Checkpoint (prioritas utama, data lengkap)
# ═══════════════════════════════════════════════════════════════════

def load_epoch_data_from_checkpoint(run_dir):
    """
    Muat data training dari checkpoint file.
    Checkpoint menyimpan train_state yang berisi histori lengkap
    setiap epoch (train loss, val loss, train mAP, val mAP, dll.).
    
    Returns:
        epoch_data dict atau None jika gagal.
    """
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    candidates = ['latest_checkpoint.pth', 'best_model.pth']

    for name in candidates:
        path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(path):
            continue

        try:
            import torch
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            ts = ckpt.get('train_state', {})
            del ckpt  # bebaskan memori (model weights)

            if not ts:
                continue

            epoch_data = _build_epoch_data_from_train_state(ts)
            if epoch_data:
                print(f"[INFO] Data dimuat dari checkpoint: {name}")
                return epoch_data

        except Exception as e:
            print(f"  [WARN] Gagal memuat {name}: {e}")
            continue

    return None


def _build_epoch_data_from_train_state(ts):
    """Bangun epoch_data dict dari train_state checkpoint."""
    h_tr_loss = ts.get('h_tr_loss', [])
    h_val_loss = ts.get('h_val_loss', [])
    h_map50 = ts.get('h_map50', [])
    h_multi_acc = ts.get('h_multi_acc', [])
    x_tr_metrics = ts.get('x_tr_metrics', [])
    h_tr_map50 = ts.get('h_tr_map50', [])
    h_tr_multi_acc = ts.get('h_tr_multi_acc', [])

    total_epochs = len(h_tr_loss)
    if total_epochs == 0:
        return None

    # Tentukan eval_frequency dari data
    n_val = len(h_val_loss)
    if n_val == 0:
        return None
    eval_freq = max(1, round(total_epochs / n_val))

    # Bangun daftar epoch validasi
    val_epochs = [i for i in range(1, total_epochs + 1) if i % eval_freq == 0][:n_val]
    # Jika eval_freq=1 dan jumlahnya pas, val_epochs = [1, 2, ..., total_epochs]

    epoch_data = {}

    # Isi train loss (setiap epoch)
    for i in range(total_epochs):
        epoch = i + 1
        epoch_data[epoch] = {'train_loss': float(h_tr_loss[i])}

    # Isi val loss, val mAP50, val acc (setiap eval epoch)
    for i, epoch in enumerate(val_epochs):
        if epoch not in epoch_data:
            epoch_data[epoch] = {}
        if i < len(h_val_loss):
            epoch_data[epoch]['val_loss'] = float(h_val_loss[i])
        if i < len(h_map50):
            epoch_data[epoch]['mAP50'] = float(h_map50[i])
        if i < len(h_multi_acc):
            epoch_data[epoch]['val_acc'] = float(h_multi_acc[i])

    # Isi train mAP50 & train acc (setiap epoch di x_tr_metrics)
    for i, epoch in enumerate(x_tr_metrics):
        if epoch not in epoch_data:
            epoch_data[epoch] = {}
        if i < len(h_tr_map50):
            epoch_data[epoch]['train_mAP50'] = float(h_tr_map50[i])
        if i < len(h_tr_multi_acc):
            epoch_data[epoch]['train_acc'] = float(h_tr_multi_acc[i])

    return epoch_data


# ═══════════════════════════════════════════════════════════════════
#  SUMBER DATA 2: Log file (fallback)
# ═══════════════════════════════════════════════════════════════════

def parse_logs(log_dir):
    """
    Parse semua file log di direktori.
    
    Sumber data:
      1. Baris epoch summary: Epoch XXX/YYY | ... TrainLoss=... ValLoss=... mAP50=... AvgAccMultiLabel=...
      2. Baris [TrainMetrics] (ditulis train.py setiap epoch, jika tersedia)
      3. Blok tabel TRAIN MULTI-LABEL (fallback, setiap TRAIN_EVAL_FREQUENCY)

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

                # 2. Parsing [TrainMetrics] compact line (setiap epoch)
                match_train_metrics = re.search(
                    r'\[TrainMetrics\]\s+E(\d+)\s*\|'
                    r'.*TrainmAP50=([\d.]+)'
                    r'.*TrainAvgAcc=([\d.]+)',
                    line
                )
                if match_train_metrics:
                    epoch = int(match_train_metrics.group(1))
                    if epoch not in epoch_data:
                        epoch_data[epoch] = {}
                    epoch_data[epoch]['train_mAP50'] = float(match_train_metrics.group(2))
                    epoch_data[epoch]['train_acc'] = float(match_train_metrics.group(3))
                    continue

                # 3. Parsing Evaluation Block Epoch (biasanya setiap 10 epoch)
                match_eval = re.search(r'METRIK MULTI-LABEL\s+\|\s+Epoch\s+(\d+)', line)
                if match_eval:
                    current_eval_epoch = int(match_eval.group(1))
                    in_train_block = False
                    continue

                # 4. Masuk ke blok TRAIN MULTI-LABEL
                if current_eval_epoch and 'TRAIN MULTI-LABEL' in line:
                    in_train_block = True
                    continue

                # 5. Ambil metrik dari dalam blok TRAIN MULTI-LABEL (fallback)
                if current_eval_epoch and in_train_block:
                    if '| Accuracy ' in line:
                        vals = re.findall(r'\|\s*([\d.]+)\s*', line)
                        if vals:
                            if current_eval_epoch not in epoch_data:
                                epoch_data[current_eval_epoch] = {}
                            if 'train_acc' not in epoch_data[current_eval_epoch]:
                                epoch_data[current_eval_epoch]['train_acc'] = float(vals[3]) if len(vals) >= 4 else float(vals[-1])

                    if '| mAP@0.50 ' in line:
                        vals = re.findall(r'\|\s*([\d.]+)\s*', line)
                        if vals:
                            if current_eval_epoch not in epoch_data:
                                epoch_data[current_eval_epoch] = {}
                            if 'train_mAP50' not in epoch_data[current_eval_epoch]:
                                epoch_data[current_eval_epoch]['train_mAP50'] = float(vals[3]) if len(vals) >= 4 else float(vals[-1])
                            in_train_block = False
                            current_eval_epoch = None

    return epoch_data


# ═══════════════════════════════════════════════════════════════════
#  FUNGSI PLOT
# ═══════════════════════════════════════════════════════════════════

def plot_accuracy(epoch_data, output_dir):
    """Plot grafik Accuracy (Train & Validation)."""
    val_epochs = sorted([e for e in epoch_data if 'val_acc' in epoch_data[e]])
    val_accuracies = [epoch_data[e]['val_acc'] for e in val_epochs]

    train_epochs = sorted([e for e in epoch_data if 'train_acc' in epoch_data[e]])
    train_accuracies = [epoch_data[e]['train_acc'] for e in train_epochs]

    if not val_epochs:
        print("  [SKIP] Data validasi accuracy kosong.")
        return

    # Best epoch
    best_idx = val_accuracies.index(max(val_accuracies))
    best_epoch = val_epochs[best_idx]
    best_acc = val_accuracies[best_idx]

    # Smoothing — adaptif berdasarkan jumlah data
    val_smooth_factor = 0.6 if len(val_accuracies) > 30 else 0.3
    train_smooth_factor = 0.6 if len(train_accuracies) > 30 else 0.3
    val_acc_smooth = smooth_curve(val_accuracies, factor=val_smooth_factor)
    train_acc_smooth = smooth_curve(train_accuracies, factor=train_smooth_factor) if train_accuracies else []

    # Plot
    _apply_style()
    plt.figure(figsize=(18, 6))

    plt.plot(val_epochs, val_acc_smooth, label='Validation Avg Accuracy',
             color='#ff7f0e', linewidth=2.5)
    if train_accuracies:
        plt.plot(train_epochs, train_acc_smooth, label='Train Avg Accuracy',
                 color='#1f77b4', linewidth=2.5)

    # Marker best epoch
    plt.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    plt.scatter([best_epoch], [best_acc], color='green', s=80, zorder=5,
                label=f'Best Val Acc={best_acc:.4f} (E{best_epoch})')

    plt.title('Training & Validation Average Accuracy', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(top=0.92)
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

    # Best epoch (berdasarkan Val mAP tertinggi)
    best_idx = val_mAP.index(max(val_mAP))
    best_epoch = val_epochs[best_idx]
    best_mAP = val_mAP[best_idx]

    # Smoothing
    val_smooth_factor = 0.6 if len(val_mAP) > 30 else 0.3
    train_smooth_factor = 0.6 if len(train_mAP) > 30 else 0.3
    val_mAP_smooth = smooth_curve(val_mAP, factor=val_smooth_factor)
    train_mAP_smooth = smooth_curve(train_mAP, factor=train_smooth_factor) if train_mAP else []

    # Plot
    _apply_style()
    plt.figure(figsize=(18, 6))

    plt.plot(val_epochs, val_mAP_smooth, label='Validation mAP@0.50',
             color='#ff7f0e', linewidth=2.5)
    if train_mAP:
        plt.plot(train_epochs, train_mAP_smooth, label='Train mAP@0.50',
                 color='#1f77b4', linewidth=2.5)

    # Marker best epoch
    plt.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    plt.scatter([best_epoch], [best_mAP], color='green', s=80, zorder=5,
                label=f'Best Val mAP={best_mAP:.4f} (E{best_epoch})')

    plt.title('Training & Validation mAP@0.50', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('mAP@0.50', fontsize=14, fontweight='bold')
    plt.ylim(0.0, 0.40)
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
    smooth_factor = 0.6 if len(train_losses) > 30 else 0.3
    train_losses_smooth = smooth_curve(train_losses, factor=smooth_factor)
    val_losses_smooth = smooth_curve(val_losses, factor=smooth_factor)

    # Plot
    _apply_style()
    plt.figure(figsize=(18, 6))

    plt.plot(loss_epochs, train_losses_smooth, label='Train Loss',
             color='#1f77b4', linewidth=2.5)
    plt.plot(loss_epochs, val_losses_smooth, label='Validation Loss',
             color='#ff7f0e', linewidth=2.5)

    # Marker best epoch
    plt.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    plt.scatter([best_epoch], [best_val_loss], color='green', s=80, zorder=5,
                label=f'Best Val Loss={best_val_loss:.4f} (E{best_epoch})')

    plt.title('Training & Validation Loss', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Total', fontsize=14, fontweight='bold')
    plt.ylim(bottom=0.8, top=1.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'loss_total.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Grafik Loss disimpan: {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  PLOT GRAFIK WIDE (1:3) - Accuracy, mAP@0.50, Loss")
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

    # ── Muat data: prioritas checkpoint, fallback log ──
    print(f"\n[PARSE] Mencoba memuat data dari checkpoint...")
    epoch_data = load_epoch_data_from_checkpoint(run_dir)

    if not epoch_data:
        print(f"[PARSE] Checkpoint tidak tersedia, parsing dari log: {log_dir}")
        epoch_data = parse_logs(log_dir)

    if not epoch_data:
        print("Tidak ada data yang dapat di-parse.")
        return

    n_epochs = len(epoch_data)
    n_train_acc = sum(1 for e in epoch_data if 'train_acc' in epoch_data[e])
    n_train_map = sum(1 for e in epoch_data if 'train_mAP50' in epoch_data[e])
    n_val_acc = sum(1 for e in epoch_data if 'val_acc' in epoch_data[e])
    n_val_map = sum(1 for e in epoch_data if 'mAP50' in epoch_data[e])
    print(f"[INFO] Total {n_epochs} epoch")
    print(f"[INFO] Train : {n_train_acc} titik accuracy, {n_train_map} titik mAP50")
    print(f"[INFO] Val   : {n_val_acc} titik accuracy, {n_val_map} titik mAP50")

    # Generate 3 grafik
    print(f"\n[PLOT] Membuat 3 grafik (rasio 1:3)...")
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
