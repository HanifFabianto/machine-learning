# Optimasi Persediaan dengan Binarisasi Demand, Machine Learning, dan Dynamic Reorder Point

Notebook `binarization_ROPdynamis.ipynb` membangun alur optimasi persediaan menggunakan dataset **DataCo Smart Supply Chain**. Proyek ini mengelompokkan produk/SKU menjadi *Low Demand* dan *High Demand*, melatih model machine learning untuk memprediksi permintaan harian, lalu menghitung *Dynamic Safety Stock* dan *Dynamic Reorder Point* (ROP).

## Tujuan Proyek

- Mengubah data transaksi menjadi deret waktu permintaan harian per SKU.
- Membuat fitur historis dan kalender untuk peramalan permintaan.
- Melakukan binarisasi SKU berdasarkan rata-rata permintaan harian.
- Membandingkan tujuh algoritma regresi pada setiap kelompok permintaan.
- Menggunakan hasil peramalan dan variasi *lead time* untuk menghitung stok pengaman dan titik pemesanan ulang yang dinamis.
- Mengekspor hasil utama ke file CSV.

## Dataset

Notebook mengunduh dataset melalui KaggleHub:

- **Dataset:** [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- **File yang digunakan:** `DataCoSupplyChainDataset.csv`

Notebook memilih kolom yang berkaitan dengan tanggal pesanan, produk, jumlah unit, penjualan, manfaat per pesanan, metode pengiriman, dan durasi pengiriman aktual.

## Alur Kerja

### Tahap 1 — Data Preparation & Preprocessing

1. Memilih kolom transaksi yang relevan.
2. Mengubah tanggal pesanan menjadi format tanggal.
3. Mengisi nilai numerik yang kosong menggunakan median per SKU, dengan median keseluruhan sebagai cadangan.
4. Mengisi nilai `Shipping Mode` yang kosong menggunakan modus.
5. Mengagregasikan transaksi menjadi permintaan harian per SKU.
6. Melengkapi tanggal tanpa transaksi dengan permintaan `0`.
7. Membuat fitur historis:
   - `lag_7` dan `lag_30`
   - `roll_mean_7` dan `roll_std_7`
   - `roll_mean_30` dan `roll_std_30`
8. Membuat fitur kalender: hari dalam minggu, tanggal dalam bulan, bulan, kuartal, dan indikator akhir pekan.
9. Membagi data secara kronologis: 80% awal untuk *train* dan 20% akhir untuk *test*.
10. Melakukan `StandardScaler` yang di-*fit* pada data train.

Fitur historis dibuat hanya dari data sebelumnya untuk mengurangi risiko *temporal leakage*. Baris yang belum memiliki fitur historis lengkap, termasuk lag 30 hari, dikeluarkan dari matriks model.

### Tahap 2 — Binarisasi Demand

Setiap SKU dikelompokkan berdasarkan rata-rata permintaan hariannya pada periode train.

| Label | Nilai | Aturan |
|---|---:|---|
| Low Demand | `0` | Rata-rata permintaan harian SKU ≤ ambang |
| High Demand | `1` | Rata-rata permintaan harian SKU > ambang |

Ambang binarisasi adalah rata-rata kolom `Demand` pada data train. Penghitungan ambang dan rata-rata permintaan per SKU menggunakan periode train agar data test tidak memengaruhi penentuan kelompok.

Notebook juga menghitung karakteristik SKU seperti `Sales_Revenue`, `ADI`, `CV2`, `Lead_Time`, dan `Criticality`, serta memetakan pola permintaan ke kuadran berikut menggunakan batas `ADI = 1.32` dan `CV² = 0.49`:

- **Smooth:** permintaan sering dan relatif stabil.
- **Erratic:** permintaan sering, tetapi jumlahnya berubah-ubah.
- **Intermittent:** permintaan jarang, tetapi jumlahnya relatif stabil.
- **Lumpy:** permintaan jarang dan jumlahnya berubah-ubah.

### Tahap 3 — Demand Forecasting dengan Machine Learning

Tujuh algoritma regresi dibandingkan secara terpisah pada kelompok *Low Demand* dan *High Demand*:

1. Lasso
2. Ridge
3. Support Vector Regression (SVR)
4. Decision Tree Regressor
5. Random Forest Regressor
6. XGBoost Regressor
7. LightGBM Regressor

Prediksi permintaan negatif dipotong menjadi nol. Karena SVR dapat memerlukan waktu komputasi tinggi, notebook membatasi sampel data train SVR hingga 20.000 baris saat diperlukan.

### Tahap 4 — Evaluasi Model dan Validasi Temporal

Performa model dievaluasi menggunakan:

- **MAE (Mean Absolute Error):** rata-rata kesalahan absolut.
- **RMSE (Root Mean Squared Error):** akar rata-rata kuadrat kesalahan.
- **MAPE (Mean Absolute Percentage Error):** persentase kesalahan absolut; perhitungan hanya menggunakan observasi dengan permintaan aktual lebih besar dari nol.

Notebook juga memeriksa urutan waktu train/test dan menyajikan perbandingan performa model per kelompok.

### Tahap 5 — Rolling-Origin dan Pemilihan Model

Notebook menjalankan validasi *rolling-origin* dengan tiga fold pada data train untuk melihat performa model pada beberapa blok waktu. Pada setiap fold, rentang data latih diperluas seiring waktu, dan scaler di-*fit* pada data latih fold tersebut.

Perlu diperhatikan bahwa **pemilihan model terbaik dalam kode dilakukan berdasarkan RMSE terkecil pada data test** untuk masing-masing kelompok. Hasil rolling-origin digabungkan ke tabel evaluasi, tetapi bukan menjadi kriteria pemilihan final pada fungsi pemilihan model yang digunakan.

### Tahap 6 — Dynamic Safety Stock dan Dynamic Reorder Point

Untuk setiap SKU, notebook merangkum rata-rata prediksi permintaan harian (`d`), RMSE peramalan (`RMSE_forecast`), rata-rata *lead time* (`L`), dan simpangan baku *lead time* (`sigma_L`).

Rumus yang digunakan:

\[
SS_{\text{dynamic}} =
Z \times \sqrt{L \times RMSE_{\text{forecast}}^2 + d^2 \times \sigma_L^2}
\]

\[
ROP_{\text{dynamic}} = d \times L + SS_{\text{dynamic}}
\]

Keterangan:

| Simbol/variabel | Arti |
|---|---|
| `SS_dynamic` | *Dynamic Safety Stock* atau stok pengaman dinamis |
| `ROP_dynamic` | *Dynamic Reorder Point* atau titik pemesanan ulang dinamis |
| `d` | Rata-rata prediksi permintaan harian dari model terbaik |
| `RMSE_forecast` | Akar rata-rata kuadrat error peramalan pada data test per SKU |
| `L` | Rata-rata durasi pengiriman aktual (*lead time*) |
| `sigma_L` | Simpangan baku durasi pengiriman aktual |
| `Z` | Faktor layanan yang ditetapkan menjadi `1.65` dalam notebook (ditujukan untuk sekitar 95%) |

Notebook juga membandingkan stok pengaman dinamis dengan perhitungan pembanding yang memakai simpangan baku permintaan historis.

## Teknologi dan Pustaka

Notebook menggunakan Python dan pustaka berikut:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `kagglehub`
- `scikit-learn`
- `xgboost`
- `lightgbm`

## Cara Menjalankan

1. Unduh atau kloning repositori yang berisi notebook ini.
2. Siapkan Python dan Jupyter Notebook/JupyterLab, atau gunakan Google Colab.
3. Instal pustaka yang diperlukan:

   ```bash
   pip install numpy pandas matplotlib seaborn kagglehub scikit-learn xgboost lightgbm
   ```

4. Buka `binarization_ROPdynamis.ipynb`.
5. Jalankan sel dari atas ke bawah. Notebook akan mengunduh dataset menggunakan KaggleHub; pastikan lingkungan memiliki akses internet dan izin yang diperlukan.
6. Periksa hasil tabel dan visualisasi, serta file CSV yang dihasilkan.

## File Output

Pada tahap akhir, notebook menyimpan tiga file CSV di direktori kerja aktif:

| File | Isi |
|---|---|
| `hasil_dynamic_ss_rop_per_sku.csv` | Hasil binarisasi kelompok, model terbaik, prediksi permintaan, RMSE forecast, Safety Stock, dan Reorder Point per SKU |
| `hasil_evaluasi_model_per_klaster.csv` | Metrik evaluasi model untuk setiap kelompok |
| `hasil_binarisasi_per_sku.csv` | Informasi dan karakteristik SKU untuk proses binarisasi |

## Catatan

- Hasil bergantung pada dataset, lingkungan eksekusi, dan versi pustaka yang digunakan.
- Ambang binarisasi dihitung dari rata-rata permintaan harian pada data train, bukan dari data keseluruhan.
- MAPE hanya dihitung pada observasi dengan permintaan aktual > 0 karena pembagian dengan nol tidak terdefinisi.
- Nilai `Z = 1.65` dipakai sesuai implementasi notebook; interpretasi service level merupakan asumsi model dan perlu disesuaikan dengan kebijakan operasional.
- Evaluasi model terbaik menggunakan RMSE pada data test, sementara rolling-origin disediakan sebagai evaluasi temporal tambahan.
- Sebelum digunakan untuk keputusan operasional, validasikan hasil, satuan waktu, definisi permintaan, dan asumsi *lead time* dengan kebutuhan bisnis.

## Struktur Repositori

Contoh struktur sederhana:

```text
.
├── README.md
└── binarization_ROPdynamis.ipynb
```

File CSV akan muncul di direktori kerja setelah notebook dijalankan.

## Referensi Dataset

- DataCo Smart Supply Chain for Big Data Analysis — Kaggle: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

