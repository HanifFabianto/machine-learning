# Stacking ML untuk Analisis Preskriptif Inventori

Proyek ini menggunakan machine learning untuk memodelkan variabilitas permintaan harian (*`sigma_daily_demand`*), membandingkan performa beberapa model regresi, memilih model berdasarkan kombinasi metrik evaluasi menggunakan **Entropy–TOPSIS**, lalu menghasilkan rekomendasi pengelolaan inventori.

Notebook sumber: `StackingML_PrescriptiveInventory.ipynb`

## Tujuan

- Mengeksplorasi dataset optimasi inventori dan *stockout*.
- Melatih serta membandingkan beberapa model regresi.
- Membuat model ensemble dengan metode **Stacking Regressor**.
- Memberi peringkat model dengan metode **Entropy–TOPSIS**.
- Menghitung rekomendasi inventori berupa *Economic Order Quantity* (EOQ), *safety stock*, dan *reorder point* (ROP).
- Menandai SKU yang mungkin membutuhkan pemesanan ulang dan memperkirakan biaya inventori.
- (Opsional) Mengirim hasil analisis ke Google Sheets melalui Google Colab.

## Dataset

Notebook mengunduh dataset melalui KaggleHub:

- **Dataset:** [Inventory and Stockout Optimization](https://www.kaggle.com/datasets/sergionefedov/inventory-and-stockout-optimization)
- **File yang digunakan:** `inventory_optimization.csv`
- **Kolom target model:** `sigma_daily_demand`

Dataset memuat atribut terkait SKU, kategori produk, permintaan harian, *lead time*, biaya, tingkat layanan, dan indikator *stockout*. Kolom yang tersedia secara persis dapat dilihat pada dataset dan notebook.

## Alur analisis

1. **Persiapan lingkungan dan data** — mengimpor pustaka, mengunduh CSV, dan membaca data menggunakan pandas.
2. **Eksplorasi data** — memeriksa ukuran data dan nilai kosong, melihat sampel data, serta memvisualisasikan korelasi dan variabilitas permintaan berdasarkan kategori.
3. **Pra-pemrosesan** — memisahkan target `sigma_daily_demand`, menghapus `sku_id` dari fitur, melakukan *one-hot encoding* untuk `category`, membagi data menjadi data latih dan uji (70:30), lalu menerapkan `MinMaxScaler`.
4. **Pemodelan regresi** — melatih dan mengevaluasi model-model berikut:
   - Ridge Regression
   - Lasso Regression
   - LightGBM Regressor
   - Random Forest Regressor
   - XGBoost Regressor
   - Decision Tree Regressor
   - Linear Regression
   - Stacking Regressor
5. **Evaluasi dan pemeringkatan** — membandingkan MAE, MSE, RMSE, MAPE, R², dan Adjusted R². Notebook menggunakan MAE, RMSE, MAPE (semakin kecil semakin baik) serta Adjusted R² (semakin besar semakin baik) dalam pemeringkatan Entropy–TOPSIS.
6. **Analisis preskriptif inventori** — menggunakan RMSE dari model dengan peringkat teratas sebagai estimasi ketidakpastian permintaan untuk menghitung rekomendasi inventori.
7. **Ekspor hasil (opsional)** — notebook menyediakan bagian autentikasi Google Colab untuk menulis hasil ke Google Sheets.

## Rekomendasi yang dihitung

Implementasi analisis preskriptif pada notebook menghitung:

- **EOQ (`prescriptive_eoq`)** — kuantitas pemesanan ekonomis berdasarkan permintaan tahunan, biaya pemesanan, dan biaya penyimpanan.
- **Safety stock (`prescriptive_safety_stock`)** — stok pengaman untuk tingkat layanan yang ditentukan.
- **ROP (`prescriptive_rop`)** — titik pemesanan ulang berdasarkan ekspektasi permintaan selama *lead time* dan *safety stock*.
- **Tindakan pemesanan (`reorder_action`)** — membandingkan *safety stock* preskriptif dengan kolom `safety_stock` jika tersedia.
- **Estimasi biaya** — termasuk biaya pemesanan tahunan, biaya penyimpanan tahunan, dan total biaya preskriptif.

Tingkat layanan pada pemanggilan fungsi di notebook ditetapkan sebesar **95%**. Hasilnya merupakan rekomendasi berdasarkan asumsi dan kolom dataset, bukan pengganti validasi operasional bisnis.

## Teknologi yang digunakan

- Python
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- LightGBM
- XGBoost
- TensorFlow/Keras dan pustaka lain yang diimpor notebook
- KaggleHub
- SciPy
- gspread dan Google Colab (untuk ekspor Google Sheets opsional)

## Cara menjalankan

### Opsi 1 — Google Colab

1. Unggah `StackingML_PrescriptiveInventory.ipynb` ke Google Colab.
2. Jalankan sel secara berurutan.
3. Pastikan akses internet tersedia agar dataset dapat diunduh melalui KaggleHub.
4. Jika ingin mengekspor hasil ke Google Sheets, jalankan sel ekspor terakhir di Google Colab dan selesaikan autentikasi akun Google saat diminta.

### Opsi 2 — Lingkungan Python lokal

1. Siapkan lingkungan Python dan instal pustaka yang digunakan notebook, misalnya:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn kagglehub lightgbm xgboost scipy statsmodels tensorflow keras gspread
   ```

2. Jalankan notebook melalui Jupyter Notebook atau JupyterLab.
3. Untuk bagian ekspor Google Sheets, gunakan Google Colab atau sesuaikan proses autentikasi karena kode ekspor memakai `google.colab`.

> Catatan: versi pustaka dan konfigurasi lingkungan dapat memengaruhi kompatibilitas. Tidak semua pustaka yang diimpor pada notebook harus digunakan di setiap tahap.

## Struktur berkas

```text
.
├── README.md
├── StackingML_PrescriptiveInventory.ipynb
└── inventory_optimization.csv  # diunduh saat notebook dijalankan
```

## Catatan dan batasan

- Target prediksi yang digunakan dalam notebook adalah `sigma_daily_demand`, bukan permintaan harian rata-rata.
- Nilai RMSE model terpilih digunakan sebagai estimasi ketidakpastian permintaan dalam perhitungan *safety stock*.
- Sebelum digunakan untuk keputusan operasional, validasi kembali asumsi perhitungan, kualitas data, hasil evaluasi model, dan rekomendasi biaya.
- Sel ekspor Google Sheets memerlukan autentikasi dan izin Google yang sesuai.

## Referensi

- [Dataset Inventory and Stockout Optimization di Kaggle](https://www.kaggle.com/datasets/sergionefedov/inventory-and-stockout-optimization)
- [Dokumentasi scikit-learn: StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
- [Dokumentasi scikit-learn: MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

