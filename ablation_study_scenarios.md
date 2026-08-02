# Skenario Ablation Study: Modul Hybrid Transformer

Dokumen ini merinci skenario *ablation study* (studi ablasi) yang berfokus pada variasi jumlah layer CNN (`nn.Conv2d`) di dalam modul Transformer (CTE, CPSA, dan LFFN). Studi ini dirancang untuk membuktikan kepada penguji/dosen pengaruh jumlah layer CNN terhadap akurasi dan beban komputasi.

## 1. Kelompok Skenario Pengurangan Layer CNN (Downscaling)
Tujuan: Membuktikan apa yang terjadi jika elemen CNN dalam Transformer dihilangkan atau dilemahkan.

### Skenario A: Baseline (Model Anda Saat Ini)
- **Konfigurasi**: CTE (1 layer 3x3) + CPSA (1 layer 3x3, 1 layer 1x1) + LFFN (2 layer 3x3, 2 layer 1x1). Total awal = 7 Layer.
- **Rincian Layer**: 4 layer CNN 1x1 + 3 layer CNN 3x3 (mengasumsikan 1 LFFN dw_conv atau perhitungan spesifik arsitektur Anda).
- **Total Layer per Blok**: 7 Layer CNN.
- **Hipotesis**: Ini adalah titik keseimbangan (sweet spot) antara kemampuan menangkap detail lokal (gambar daun) dan pemahaman konteks global.

### Skenario B: Mengurangi CNN di LFFN
- **Ubah**: Hapus konvolusi spasial (3x3) di LFFN. Jadikan LFFN murni menggunakan CNN 1x1 (Pointwise). CTE dan CPSA tidak disentuh.
- **Rincian Layer**: 4 layer CNN 1x1 + 1 layer CNN 3x3.
- **Total Layer per Blok**: **5 Layer**.
- **Hipotesis**: Akurasi deteksi tepi (seperti bentuk ulat grayak) akan turun karena model kehilangan kemampuan spasial (Spatial Inductive Bias) di feed-forward.

### Skenario C: Mengurangi CNN di CPSA
- **Ubah**: Hapus konvolusi spasial (3x3) di dalam CPSA. Biarkan Attention bekerja menggunakan CNN 1x1 (Linear). CTE dan LFFN tidak disentuh.
- **Rincian Layer**: 4 layer CNN 1x1 + 2 layer CNN 3x3.
- **Total Layer per Blok**: **6 Layer**.
- **Hipotesis**: Beban memori komputasi akan melonjak drastis karena *Parameter-Sharing* yang dilakukan oleh CNN spasial dihilangkan, namun akurasi belum tentu naik.

### Skenario D: Mengurangi CNN di CPSA dan LFFN (Transformer Tanpa CNN Spasial)
- **Ubah**: Menggabungkan Skenario B dan C. Seluruh konvolusi spasial (3x3) di dalam modul CPSA dan LFFN dihapus dan diganti dengan 1x1. **CTE sama sekali tidak disentuh** (tetap dipertahankan sebagai jembatan fitur).
- **Rincian Layer**: 4 layer CNN 1x1 + 0 layer CNN 3x3 (di dalam blok CPSA & LFFN).
- **Total Layer per Blok**: **4 Layer**.
- **Hipotesis**: Modul Transformer akan kehilangan insting spasial lokalnya dan hanya mengandalkan fitur spasial tunggal dari CTE, yang mungkin menyulitkannya membedakan tepi hama yang kompleks.

#### Skenario E: "Transformer + Linear Head" (Hapus CNN di Prediction Head)
- **Ubah**: Melanjutkan Skenario D, kita hapus seluruh `nn.Conv2d(3x3)` di bagian *Detection Head* (Bagian penentu kelas dan kotak). Ganti menggunakan layer Linear/Dense/Pointwise murni. (CTE dibiarkan tetap 3x3).
- **Sisa CNN 3x3**: Hanya ada di CTE, Backbone (ResNet), dan Neck (FPN).
- **Hipotesis**: Akurasi tebakan kotak (Bounding Box) akan berantakan karena Head kehilangan kemampuan menyelaraskan bentuk dan posisi spasial yang biasanya dibantu oleh CNN.

### Skenario F: "Hapus Neck/FPN CNN" (Tanpa Pyramid Konvolusi)
- **Ubah**: Melanjutkan Skenario E, kita hapus jaringan konvolusi spasial di FPN/PANet. Fitur dari Backbone langsung diproses tanpa peleburan multi-skala berbasis CNN spasial.
- **Sisa CNN 3x3**: Hanya di CTE dan Backbone (ResNet).
- **Hipotesis**: Model akan sangat lemah mendeteksi hama berukuran sangat kecil (seperti telur/ulat kecil) karena fitur multi-skala hancur.

### Skenario G: "Hapus Backbone ResNet" (Tanpa Ekstraktor Awal)
- **Ubah**: Melanjutkan Skenario F, kita menghapus **seluruh Backbone ResNet**. Gambar diproses langsung menuju CTE (Atau Patch Projection awal). 
- **Sisa CNN 3x3**: Satu-satunya CNN spasial yang tersisa di seluruh arsitektur model kini **hanya CTE**.
- **Hipotesis**: Model akan sangat kesulitan mengenali bentuk daun dan hama dari nol tanpa bantuan ResNet, waktu training akan meningkat drastis.

### Skenario H: "Extreme Pure Transformer - 0 CNN" (Hapus CTE Spasial)
- **Ubah**: Melanjutkan Skenario G, ini adalah langkah pemusnahan terakhir. Konvolusi spasial (3x3) terakhir benteng pertahanan yang tersisa di dalam CTE akhirnya **dihapus** dan diganti dengan 1x1 (Linear Patch Projection).
- **Total Layer CNN 3x3 di seluruh Model**: **0 Layer** (Extreme Pure ViT).
- **Hipotesis**: Transformer benar-benar menjadi murni tanpa kemampuan spasial sedikitpun dari ujung ke ujung. Model ini kemungkinan besar akan *underfit* parah atau hancur karena ketiadaan pondasi ekstraksi fitur lokal (Inductive Bias) bawaan dari CNN sama sekali.

---

## 2. Kelompok Skenario Penambahan Layer CNN (Upscaling)
Tujuan: Menjawab tantangan "apakah 7 layer itu terlalu sedikit?", kita buat skenario eksperimen dengan memperdalam layer CNN hingga mendekati ~15 layer di dalam blok Transformer.

### Skenario I: "Deep CTE" (Penguatan Pengekstraksi Awal)
- **Ubah**: Daripada hanya 1 layer Conv di CTE, kita buat CTE memiliki 3 layer berurutan (Conv -> Conv -> Conv) seperti batang awal arsitektur modern (stem).
- **Rincian**: Deep CTE (3 layer) + CPSA (2 layer) + LFFN (4 layer).
- **Total Layer per Blok**: **9 Layer**.
- **Hipotesis**: Menambah parameter di awal sebelum masuk Transformer bisa memberikan fitur yang lebih matang, namun ada risiko *overfitting*.

### Skenario J: "Deep CPSA" (Attention Multi-Skala)
- **Ubah**: Di dalam modul CPSA, kita berikan 2 jalur `sr_conv` berbeda (misal 3x3 dan 5x5) yang di-*concat*, lalu disatukan dengan *Pointwise* tambahan.
- **Rincian**: CTE (1) + Deep CPSA (4 layer) + LFFN (4 layer).
- **Total Layer per Blok**: **9 Layer**.
- **Hipotesis**: Multi-skala konvolusi di dalam Attention membuat model bisa melihat penyakit kecil (moler) dan objek panjang (ulat) bersamaan.

### Skenario K: "Heavy Hybrid" (Mendekati ~15 Layer)
- **Ubah**: Kita maksimalkan jumlah konvolusi di seluruh blok. CTE diperdalam (3 layer). LFFN kita jadikan modul *Inverted Bottleneck* ganda (tiap LFFN punya 6 layer konvolusi). CPSA memakai multi-skala (4 layer). Dan tambah 2 layer CNN sebelum masuk ke Detection Head.
- **Rincian**: Deep CTE (3) + Deep CPSA (4) + Double LFFN (6) + Post-Norm CNN (2).
- **Total Layer per Blok**: **15 Layer CNN**.
- **Hipotesis**: Model ini akan sangat dalam dan *powerful*, namun hampir dipastikan jumlah parameter (parameter count) akan melonjak drastis, sehingga waktu inferensi (FPS) akan anjlok.

---

## Tabel Rangkuman Skenario Eksperimen

| Skenario | Keterangan | Target Penghapusan | Karakteristik |
| :--- | :--- | :--- | :--- |
| **A (Baseline)** | 7 Layer (4x 1x1 + 3x 3x3) | - | Ringan, Seimbang (*Sweet-Spot*) |
| **B (Kurang LFFN)**| 5 Layer (4x 1x1 + 1x 3x3) | LFFN (3x3 -> 1x1) | Kehilangan fitur spasial lokal FFN |
| **C (Kurang CPSA)**| 6 Layer (4x 1x1 + 2x 3x3) | CPSA (3x3 -> 1x1) | Boros Memori (O(N^2) Attention) |
| **D (D - LFFN & CPSA)**| 4 Layer (4x 1x1 + 0x 3x3 di CPSA/LFFN) | CPSA & LFFN (3x3 -> 1x1)| CTE masih jadi jembatan 3x3 |
| **E (Linear Head)**| Model + Head Linear | Detection Head (3x3 -> 1x1)| Bounding box meleset/kasar |
| **F (No FPN)** | E (Tanpa Pyramid Neck) | Neck/FPN (3x3 -> 1x1) | Kemampuan multi-skala hancur |
| **G (No ResNet)** | F (Tanpa Backbone) | Backbone ResNet Dihapus | CTE adalah CNN terakhir yang tersisa |
| **H (0 CNN)** | Extreme Pure ViT | CTE (3x3 -> 1x1) | 0 CNN Spasial di Seluruh Arsitektur |
| **I (Deep CTE)** | 9 Layer (Upscaling) | - | Fitur masuk lebih matang |
| **J (Deep CPSA)**| 9 Layer (Upscaling) | - | Kuat mendeteksi objek beragam ukuran |
| **K (Heavy Hybrid)**| 15 Layer (Upscaling)| - | Sangat berat, lambat, risiko *overfit* |

## Cara Menggunakan Ini Untuk Skripsi/Tesis
Jika dosen pembimbing bertanya, Anda dapat mengajukan Skenario **B**, **D**, dan **I** sebagai *ablation study* di bab Metodologi. Anda bisa berkata: 
*"Untuk membuktikan bahwa 7 layer adalah desain yang paling optimal, saya melakukan ablasi dengan menghapus layer (Skenario B dan D) dan menambah layer (Skenario I). Hasilnya akan menunjukkan apakah 7 layer itu kurang, cukup, atau berlebihan."*
