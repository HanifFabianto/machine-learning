# ABC-XYZ-FSN dan Dynamic Reorder Point (ROP)

## Deskripsi Proyek

Proyek ini menggabungkan klasifikasi inventori **ABC**, **XYZ**, dan **FSN** dengan pemodelan prediksi permintaan harian serta perhitungan **Dynamic Safety Stock** dan **Reorder Point (ROP)**. Analisis dilakukan pada tingkat produk untuk membantu memahami kontribusi kuantitas penjualan, kestabilan permintaan, dan frekuensi pergerakan produk.

Notebook utama: `ABC_XYZ_FSN_ROPdinamis.ipynb`

## Tujuan

- Mengeksplorasi dataset rantai pasok dan penjualan.
- Membentuk data permintaan harian per produk (*daily demand*).
- Mengelompokkan produk berdasarkan ABC, XYZ, dan FSN.
- Membandingkan beberapa model regresi untuk memprediksi kuantitas permintaan harian pada kategori FSN.
- Menggunakan hasil peramalan dan kesalahan prediksi sebagai masukan perhitungan Safety Stock dan ROP dinamis.

## Dataset

Notebook mengunduh dataset melalui `kagglehub`:

- **Dataset Kaggle:** [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- File yang digunakan dalam notebook: `DataCoSupplyChainDataset.csv`
- Kolom yang digunakan antara lain tanggal pemesanan, nama produk, kuantitas item, harga produk, dan diskon item.

Notebook mengambil file CSV menggunakan daftar file hasil unduhan dan memilih `csv_files[1]`. Jika urutan file CSV berbeda di lingkungan Anda, sesuaikan indeks tersebut agar menunjuk ke `DataCoSupplyChainDataset.csv`.

## Alur Analisis

### 1. Memuat dan mengeksplorasi data

Notebook memuat pustaka analisis data, visualisasi, serta machine learning; mengunduh dataset; kemudian memeriksa dimensi data, nilai kosong, korelasi fitur numerik, dan jumlah transaksi per kategori produk.

### 2. Feature engineering permintaan harian

Data tanggal diubah ke format datetime dan diagregasikan berdasarkan tanggal serta `Product Name`. Variabel yang dihitung meliputi:

- `Daily_Order_Quantity`: total kuantitas pesanan per produk per hari (target prediksi).
- `Avg_Price`: rata-rata harga produk pada agregasi harian.
- `Total_Discount`: total diskon pada agregasi harian.
- Fitur kalender: hari dalam minggu, bulan, kuartal, tahun, dan indikator akhir pekan.

Nama produk diubah menggunakan one-hot encoding. Notebook juga menerapkan `MinMaxScaler` dan membagi data menjadi train dan test dengan `test_size=0.3` serta `random_state=42`.

### 3. Klasifikasi FSN

FSN didasarkan pada rasio hari aktif, yaitu jumlah hari dengan transaksi produk dibandingkan total rentang hari dataset.

| Kategori | Aturan pada notebook | Interpretasi |
|---|---:|---|
| Fast-moving (F) | Rasio hari aktif ≥ 30% | Produk sering mengalami transaksi |
| Slow-moving (S) | Rasio hari aktif ≥ 5% dan < 30% | Produk mengalami transaksi dengan frekuensi lebih rendah |
| Non-moving (N) | Rasio hari aktif < 5% | Produk sangat jarang mengalami transaksi |

### 4. Klasifikasi ABC

Produk diurutkan berdasarkan total kuantitas penjualan, lalu dihitung persentase kumulatifnya.

| Kategori | Ambang persentase kumulatif pada notebook |
|---|---:|
| A | ≤ 70% |
| B | > 70% sampai dengan 90% |
| C | > 90% |

### 5. Klasifikasi XYZ

XYZ menggunakan *coefficient of variation* (CV), yaitu standar deviasi permintaan harian dibagi rata-rata permintaan harian.

| Kategori | Aturan CV | Interpretasi |
|---|---:|---|
| X | CV ≤ 0,5 | Permintaan relatif stabil |
| Y | 0,5 < CV ≤ 1,0 | Permintaan berfluktuasi sedang |
| Z | CV > 1,0 | Permintaan sangat bervariasi |

Hasil ABC, XYZ, dan FSN kemudian digabung berdasarkan `Product Name` untuk membentuk segmen gabungan, misalnya `A-X-Fast-moving`.

### 6. Pemodelan dan evaluasi prediksi

Notebook menjalankan fungsi regresi secara terpisah untuk kategori FSN (Fast-moving, Slow-moving, dan Non-moving). Model yang dibandingkan adalah:

- Lasso Regression
- LightGBM Regressor
- Random Forest Regressor
- XGBoost Regressor
- Decision Tree Regressor
- Linear Regression
- Support Vector Regression (SVR)

Kinerja model dievaluasi menggunakan **MAE**, **RMSE**, dan **MAPE**. Notebook mencari model dengan error terkecil untuk setiap metrik dan membuat peringkat rata-rata dari ketiga metrik untuk menentukan model terbaik secara keseluruhan.

### 7. Simulasi forecast dan inventori dinamis

Notebook membuat simulasi prediksi untuk produk kategori Fast-moving menggunakan `RandomForestRegressor` dengan `n_estimators=100` dan `max_depth=10`. Selisih nilai aktual dan hasil prediksi digunakan sebagai error peramalan.

Fungsi inventori dinamis menghitung Safety Stock (SS) dan Reorder Point (ROP) dengan mempertimbangkan variabilitas permintaan, variabilitas lead time, tingkat layanan, dan faktor risiko model:

- **Safety Stock:** `Z × √(Lead Time × σ_d² + Demand² × σ_l²) × (1 + risk_ml)`
- **Reorder Point:** `(Demand × Lead Time) + Safety Stock`

Parameter yang ditetapkan dalam notebook:

| Parameter | Nilai | Keterangan |
|---|---:|---|
| `SERVICE_LEVEL_Z` | 1,65 | Z-score yang diasosiasikan notebook dengan service level sekitar 95% |
| `AVG_LEAD_TIME` | 3 hari | Rata-rata lead time asumsi |
| `SIGMA_L` | 0,5 hari | Standar deviasi lead time asumsi |
| `risk_ml` | MAPE model terbaik atau 0,29 | Faktor penyesuaian risiko; notebook menggunakan 0,29 jika `df_comparison` tidak tersedia |

Hasil akhir disimpan pada `df_master_inventory`, yang memuat permintaan aktual, hasil forecast, error, `Safety_Stock`, dan `Reorder_Point`.

> **Catatan implementasi:** parameter lead time dan faktor risiko di atas merupakan asumsi/aturan yang ditulis di notebook, bukan nilai yang otomatis diestimasi dari data pemasok. Sebelum digunakan untuk keputusan operasional, validasikan definisi satuan, metode evaluasi, serta parameter dengan data inventori aktual.

## Teknologi dan Pustaka

Notebook menggunakan Python dengan pustaka berikut:

- `pandas` dan `numpy` — manipulasi dan perhitungan data
- `matplotlib` dan `seaborn` — visualisasi
- `kagglehub` — pengunduhan dataset Kaggle
- `scikit-learn` — preprocessing, pembagian data, model regresi, dan metrik evaluasi
- `xgboost` — model XGBoost
- `lightgbm` — model LightGBM
- `optuna` — diimpor pada notebook, tetapi tidak tampak digunakan pada alur utama yang ditampilkan

## Cara Menjalankan

1. Pastikan Python dan Jupyter Notebook/JupyterLab tersedia.
2. Instal pustaka yang diperlukan. Contoh:

   ```bash
   pip install pandas numpy matplotlib seaborn kagglehub scikit-learn xgboost lightgbm optuna
   ```

3. Buka `ABC_XYZ_FSN_ROPdinamis.ipynb` di Jupyter Notebook atau Google Colab.
4. Jalankan sel secara berurutan dari atas ke bawah.
5. Pastikan akses dan persyaratan unduhan dataset Kaggle tersedia.
6. Jika pemilihan file CSV gagal, periksa daftar `csv_files` dan ubah `csv_files[1]` agar mengarah ke `DataCoSupplyChainDataset.csv`.

## Output Utama

Beberapa objek hasil yang dibuat notebook:

- `daily_demand` — data permintaan harian per produk.
- `fsn_stats` — statistik dan kategori FSN.
- `abc_stats` — statistik dan kategori ABC.
- `xyz_stats` — statistik dan kategori XYZ.
- `product_analysis` — gabungan klasifikasi ABC-XYZ-FSN.
- `results_fsn` / `Regression_comparision` — hasil evaluasi model regresi per kategori FSN.
- `df_simulasi_forecast` — data aktual, forecast, dan error untuk simulasi kategori Fast-moving.
- `df_master_inventory` — hasil akhir simulasi Safety Stock dan Reorder Point dinamis.

## Batasan dan Hal yang Perlu Diperhatikan

- Notebook berfokus pada kuantitas pesanan harian per produk dan klasifikasi berbasis ambang yang ditetapkan di kode.
- Kategori ABC pada implementasi ini dihitung dari **kuantitas**, bukan nilai pendapatan atau biaya produk.
- Perhitungan FSN menggunakan jumlah hari aktif dibandingkan keseluruhan rentang waktu dataset.
- Simulasi inventori menggunakan asumsi lead time dan variabilitas lead time yang ditentukan manual.
- Pada sel simulasi, model Random Forest dilatih pada data test yang telah diskalakan, lalu memprediksi data yang sama. Untuk evaluasi yang tidak bias, model sebaiknya dilatih pada data train dan dievaluasi pada data test yang terpisah.
- Pastikan definisi MAPE, satuan demand, dan faktor risiko konsisten sebelum menggunakan SS/ROP untuk operasional.

## Struktur File

```text
.
├── README.md
└── ABC_XYZ_FSN_ROPdinamis.ipynb
```

Dataset diunduh melalui `kagglehub` ketika notebook dijalankan, sehingga tidak harus disimpan di repositori kecuali memang diperlukan.

## Lisensi dan Sumber Data

Silakan merujuk ke halaman dataset Kaggle untuk informasi sumber, atribusi, dan ketentuan penggunaan dataset: [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).

