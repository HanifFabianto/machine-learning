# Optimasi Persediaan dengan K-Means, Demand Forecasting, dan Dynamic ROP

Notebook ini mengembangkan pipeline optimasi persediaan berbasis data untuk produk/SKU. Alurnya mencakup preprocessing transaksi menjadi permintaan harian, segmentasi SKU dengan K-Means, pemodelan peramalan permintaan untuk setiap klaster, pemilihan model terbaik berbasis error dan stabilitas temporal, serta perhitungan **Dynamic Safety Stock** dan **Dynamic Reorder Point (ROP)**.

## Tujuan

- Mengubah data transaksi menjadi deret waktu permintaan harian per SKU.
- Mengelompokkan SKU berdasarkan karakteristik nilai penjualan, pola permintaan, lead time, dan proksi criticality.
- Membandingkan tujuh algoritma regresi untuk setiap klaster.
- Memilih model terbaik per klaster dengan mempertimbangkan performa dan stabilitas dari waktu ke waktu.
- Menghitung safety stock dan reorder point dinamis berdasarkan ketidakpastian peramalan dan lead time.

## Dataset

Notebook mengunduh dataset melalui `kagglehub`:

- **Dataset:** [DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- **File yang digunakan:** `DataCoSupplyChainDataset.csv`
- **Kolom utama:** tanggal order, ID dan nama produk, sales, kuantitas item, waktu pengiriman aktual, benefit per order, dan shipping mode.

Notebook menyebut data transaksi berjumlah lebih dari 180.000 baris. Data kemudian diubah menjadi panel permintaan harian per SKU.

## Alur Pipeline

1. **Data preparation** — memilih kolom, mengubah tanggal, menangani nilai kosong, dan memeriksa data.
2. **Agregasi demand** — menjumlahkan kuantitas terjual per SKU per hari.
3. **Pembentukan panel harian** — melengkapi hari tanpa transaksi dengan demand 0.
4. **Feature engineering** — membuat fitur lag, rolling statistics, dan fitur kalender.
5. **Pembagian temporal** — 80% periode awal untuk train dan 20% periode akhir untuk test.
6. **Segmentasi K-Means** — mengelompokkan SKU dengan lima kriteria inventori.
7. **Demand forecasting** — melatih tujuh model regresi secara terpisah di setiap klaster.
8. **Evaluasi dan pemilihan model** — membandingkan MAE, RMSE, MAPE, serta hasil rolling-origin.
9. **Dynamic inventory** — menghitung Safety Stock dan Reorder Point untuk setiap SKU.
10. **Ekspor hasil** — menyimpan hasil inventori, evaluasi model, dan segmentasi ke CSV.

## 1. Preprocessing dan Feature Engineering

Kolom transaksi yang digunakan meliputi:

- `Product Card Id` sebagai identitas SKU.
- `order date (DateOrders)` sebagai tanggal.
- `Order Item Quantity` sebagai kuantitas permintaan.
- `Sales` sebagai nilai penjualan.
- `Days for shipping (real)` sebagai lead time aktual.
- `Benefit per order` dan `Shipping Mode` sebagai informasi tambahan.

Nilai kosong pada kolom numerik diisi menggunakan median per SKU, dengan median global sebagai fallback. Nilai kosong pada `Shipping Mode` diisi dengan modus.

Transaksi diagregasi menjadi `Demand` harian per SKU. Hari tanpa transaksi di antara tanggal pertama dan terakhir produk tercatat dilengkapi dengan demand 0.

Fitur peramalan yang digunakan:

| Fitur | Deskripsi |
|---|---|
| `lag_7` | Demand tujuh hari sebelumnya |
| `lag_30` | Demand 30 hari sebelumnya |
| `roll_mean_7` | Rata-rata demand tujuh hari sebelumnya |
| `roll_std_7` | Standar deviasi demand tujuh hari sebelumnya |
| `roll_mean_30` | Rata-rata demand 30 hari sebelumnya |
| `roll_std_30` | Standar deviasi demand 30 hari sebelumnya |
| `DayOfWeek` | Hari dalam minggu |
| `DayOfMonth` | Tanggal dalam bulan |
| `Month` | Bulan |
| `Quarter` | Kuartal |
| `Is_Weekend` | Indikator akhir pekan |

Rolling features menggunakan demand yang digeser satu hari agar tidak memakai nilai demand pada hari yang sedang diprediksi. Baris yang belum memiliki seluruh fitur, terutama karena kebutuhan `lag_30`, dihapus. `StandardScaler` di-fit pada data train.

## 2. Segmentasi SKU dengan K-Means

Setiap SKU dikelompokkan menggunakan lima kriteria:

| Kriteria | Makna |
|---|---|
| `Sales_Revenue` | Total nilai penjualan pada periode train |
| `CV2` | Kuadrat koefisien variasi demand pada hari dengan permintaan |
| `ADI` | Rasio jumlah hari pengamatan terhadap jumlah hari dengan permintaan |
| `Lead_Time` | Rata-rata waktu pengiriman aktual |
| `Criticality` | Proksi urgensi berdasarkan `Shipping Mode` |

Dataset tidak menyediakan kolom `Criticality`. Notebook membentuk proksi ordinal dari `Shipping Mode`:

- Standard Class = 1
- Second Class = 2
- First Class = 3
- Same Day = 4

Nilai tersebut kemudian dirata-ratakan per SKU. Seluruh kriteria dinormalisasi menggunakan `MinMaxScaler`. Jumlah klaster dicari dengan **Silhouette Coefficient** untuk kandidat \(k=2\) sampai \(k=8\); nilai tertinggi dipilih, kecuali `K_OVERRIDE` ditetapkan secara manual.

Notebook juga memetakan pola demand menggunakan ADI dan CV² dengan ambang ADI = 1,32 dan CV² = 0,49:

- **Smooth:** demand sering dan relatif stabil.
- **Erratic:** demand sering tetapi bervariasi.
- **Intermittent:** demand jarang tetapi relatif stabil.
- **Lumpy:** demand jarang dan bervariasi.

Label klaster awal dibuat sebagai `Klaster 0`, `Klaster 1`, dan seterusnya. Label ini bukan interpretasi bisnis otomatis.

## 3. Model Demand Forecasting

Tujuh algoritma regresi dilatih secara terpisah untuk setiap klaster:

| Model | Konfigurasi utama di notebook |
|---|---|
| Lasso | `alpha=0.01`, `max_iter=10000` |
| Ridge | `alpha=1.0` |
| SVR | RBF kernel, `C=10`, `epsilon=0.1` |
| Decision Tree Regressor | `max_depth=10` |
| Random Forest Regressor | 100 estimator, `max_depth=10` |
| XGBoost Regressor | 200 estimator, `learning_rate=0.1`, `max_depth=6` |
| LightGBM Regressor | 200 estimator, `learning_rate=0.05` |

Prediksi demand negatif dipotong menjadi nol. Karena SVR dapat memerlukan waktu komputasi besar, jumlah sampel train untuk fitting SVR dibatasi hingga 20.000 baris apabila jumlah data lebih besar.

### Metrik evaluasi

- **MAE (Mean Absolute Error):** rata-rata nilai absolut kesalahan.
- **RMSE (Root Mean Squared Error):** memberi penalti lebih besar pada kesalahan yang besar.
- **MAPE:** dihitung hanya pada observasi dengan demand aktual lebih dari nol, karena pembagian dengan nol tidak terdefinisi.

Nilai metrik yang lebih kecil menunjukkan error yang lebih rendah. Hasil dapat berbeda sesuai lingkungan dan versi library; README ini tidak menetapkan angka performa karena angka tersebut perlu diambil dari hasil eksekusi notebook.

## 4. Validasi Temporal dan Pemilihan Model

Notebook memeriksa bahwa periode test berada setelah periode train dan memverifikasi bahwa `lag_7` sesuai dengan demand tujuh hari sebelumnya.

Selain evaluasi pada test set, notebook menerapkan **rolling-origin validation** dengan tiga fold pada periode train. Pada tiap fold, model dilatih menggunakan periode yang lebih awal dan diuji pada blok waktu sesudahnya.

Pemilihan model terbaik dilakukan **per klaster**, berdasarkan rata-rata peringkat dari lima ukuran berikut:

1. MAE pada test set.
2. RMSE pada test set.
3. MAPE pada test set.
4. Rata-rata RMSE rolling-origin.
5. Standar deviasi RMSE rolling-origin.

Peringkat 1 adalah nilai error terkecil. Model dengan `Avg_Error_Rank` terkecil dipilih sebagai model terbaik keseluruhan untuk klaster tersebut. Notebook juga menampilkan heatmap RMSE dan ringkasan model terbaik.

## 5. Dynamic Safety Stock dan Reorder Point

Perhitungan inventori menggunakan rata-rata prediksi demand, error peramalan, dan statistik lead time untuk setiap SKU.

Rumus yang digunakan:

\[
SS_{\text{dynamic}} =
Z \sqrt{L \times RMSE_{\text{forecast}}^2 + d^2 \times \sigma_L^2}
\]

\[
ROP_{\text{dynamic}} = d \times L + SS_{\text{dynamic}}
\]

Keterangan:

| Simbol | Arti |
|---|---|
| \(d\) | Rata-rata prediksi demand harian dari model terbaik klaster |
| \(RMSE_{\text{forecast}}\) | RMSE forecast SKU pada periode test |
| \(L\) | Rata-rata lead time SKU |
| \(\sigma_L\) | Standar deviasi lead time SKU |
| \(Z\) | Faktor service level; notebook menetapkan 1,65 (sekitar 95%) |
| \(SS_{\text{dynamic}}\) | Safety stock dinamis |
| \(ROP_{\text{dynamic}}\) | Titik pemesanan ulang dinamis |

Notebook juga menghitung pembanding `SS_Statis` menggunakan standar deviasi demand historis dari periode train, lalu membandingkan rata-rata safety stock statis dan dinamis per klaster.

> Hasil SS dan ROP merupakan estimasi berbasis asumsi dan performa model. Sebelum dipakai untuk keputusan pembelian, validasikan definisi service level, satuan demand, distribusi error, lead time, dan kebijakan stok perusahaan.

## Output

Notebook mengekspor tiga file CSV:

| File | Isi |
|---|---|
| `hasil_dynamic_ss_rop_per_sku.csv` | Estimasi inventori per SKU, termasuk safety stock, reorder point, klaster, dan model terbaik |
| `hasil_evaluasi_model_per_klaster.csv` | Metrik evaluasi model untuk setiap klaster |
| `hasil_segmentasi_kmeans_per_sku.csv` | Kriteria segmentasi, nomor klaster, label klaster, dan kuadran demand per SKU |

Notebook juga menghasilkan visualisasi seperti heatmap korelasi, permintaan per kategori, evaluasi silhouette, peta ADI–CV², heatmap RMSE, serta perbandingan safety stock statis dan dinamis.

## Teknologi dan Dependensi

- Python 3
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- LightGBM
- kagglehub

Instal dependensi dengan:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm kagglehub
```

## Cara Menjalankan

1. Buka notebook `K_means_ROPdinamis.ipynb` di Google Colab atau Jupyter.
2. Pasang dependensi jika belum tersedia.
3. Pastikan lingkungan memiliki akses internet dan akses untuk mengunduh dataset dari Kaggle.
4. Jalankan seluruh sel secara berurutan, mulai dari unduh dataset hingga ekspor hasil.
5. Periksa hasil tabel, visualisasi, dan tiga file CSV yang dihasilkan.

Notebook memilih `csv_files[1]` untuk memuat dataset. Jika urutan file hasil unduhan berbeda di lingkungan lain, periksa daftar file CSV yang dicetak dan sesuaikan indeks agar benar-benar menunjuk ke `DataCoSupplyChainDataset.csv`.

## Batasan dan Catatan

- Segmentasi K-Means sensitif terhadap skala dan pemilihan kriteria; hasilnya perlu ditinjau secara bisnis.
- `Criticality` hanyalah proksi ordinal dari shipping mode, bukan ukuran kritikalitas produk yang terverifikasi.
- MAPE tidak mencakup hari dengan demand aktual nol.
- Evaluasi test bersifat temporal, tetapi kinerja pada data historis belum menjamin performa di kondisi operasional mendatang.
- Formula inventori merupakan pendekatan estimasi; asumsi dan parameter perlu diuji pada data serta kebijakan perusahaan.
- Label klaster default masih generik (`Klaster n`) dan sebaiknya diberi nama bisnis setelah profil tiap klaster ditinjau.

---

**Kategori proyek:** Supply Chain Analytics · Inventory Optimization · K-Means Clustering · Demand Forecasting · Dynamic Safety Stock · Reorder Point

