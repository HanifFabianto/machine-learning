# Predictive Maintenance: Prediksi Kegagalan Mesin dengan Machine Learning

Proyek ini membangun dan membandingkan dua model untuk memprediksi **kegagalan mesin industri** menggunakan AI4I 2020 Predictive Maintenance Dataset. Notebook mengevaluasi model **Random Forest Classifier** dan **MLP Neural Network** dengan metrik klasifikasi, validasi silang, visualisasi, serta estimasi biaya bisnis.

> **Catatan nama file:** Walaupun notebook bernama `Remaining_Usefull_Life_(RUL).ipynb`, implementasi yang ada saat ini memprediksi label biner `Machine failure` (gagal/tidak gagal), bukan mengestimasi Remaining Useful Life dalam bentuk nilai waktu atau jumlah siklus.

## Tujuan Proyek

- Mengolah dan mengeksplorasi data pemeliharaan mesin.
- Membentuk fitur tambahan yang relevan dengan kondisi termal dan mekanis.
- Melatih serta membandingkan Random Forest dan MLP Neural Network.
- Mengevaluasi hasil dengan F1-Score, ROC-AUC, Average Precision, confusion matrix, dan Stratified 5-Fold Cross-Validation.
- Menghubungkan hasil klasifikasi dengan ilustrasi biaya downtime dan inspeksi.

## Dataset

Notebook mengunduh dataset melalui `kagglehub`:

- **Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- **Sumber yang disebutkan dalam notebook:** UCI Machine Learning Repository
- **Jumlah data:** sekitar 10.000 baris
- **Target:** `Machine failure` (0 = tidak gagal, 1 = gagal)

Dataset berisi informasi tipe mesin, suhu udara dan proses, kecepatan rotasi, torsi, keausan alat, serta label kegagalan mesin.

## Alur Pengerjaan

1. **Pemuatan dan pemeriksaan data** — membaca CSV, memeriksa dimensi, tipe data, missing values, duplikasi, statistik deskriptif, dan distribusi kelas target.
2. **Exploratory Data Analysis (EDA)** — menganalisis distribusi kegagalan dan hubungan fitur dengan kondisi kegagalan melalui visualisasi.
3. **Feature engineering** — mengubah kolom `Type` menjadi numerik dan membuat fitur turunan.
4. **Preprocessing** — membagi data secara stratified menjadi 80% train dan 20% test; melakukan standardisasi untuk input MLP.
5. **Pelatihan model** — melatih Random Forest dan MLP.
6. **Evaluasi dan perbandingan** — mengukur metrik pada test set dan melakukan validasi silang 5-fold pada data train.
7. **Interpretasi industri** — mengaitkan false negative dan false positive dengan asumsi biaya operasional.

## Feature Engineering

Notebook menggunakan 10 fitur masukan:

| Fitur | Keterangan |
|---|---|
| `Type_enc` | Encoding numerik dari tipe mesin |
| `Air temperature [K]` | Suhu udara |
| `Process temperature [K]` | Suhu proses |
| `Rotational speed [rpm]` | Kecepatan rotasi |
| `Torque [Nm]` | Torsi |
| `Tool wear [min]` | Keausan alat |
| `Temp_diff` | Suhu proses dikurangi suhu udara |
| `Power` | Estimasi daya mekanis dari torsi dan kecepatan rotasi |
| `Wear_Torque` | Interaksi keausan alat dan torsi |
| `Speed_Torque` | Interaksi kecepatan rotasi dan torsi |

Target klasifikasi adalah `Machine failure`.

## Model

### 1. Random Forest Classifier

Konfigurasi utama:
- `n_estimators=300`
- `max_depth=12`
- `min_samples_split=5`
- `min_samples_leaf=2`
- `max_features="sqrt"`
- `class_weight="balanced"`
- `random_state=42`

Model dilatih menggunakan fitur yang telah direkayasa, tanpa standardisasi.

### 2. MLP Neural Network

Konfigurasi utama:
- Arsitektur: `128 → 64 → 32` neuron pada hidden layers
- Aktivasi: ReLU
- Optimizer: Adam
- `alpha=0.001` untuk regularisasi L2
- `batch_size=64`
- `max_iter=300`
- Early stopping dengan `validation_fraction=0.15`

MLP menggunakan fitur yang telah distandardisasi dengan `StandardScaler` yang di-fit hanya pada data train.

## Evaluasi

Notebook membandingkan model menggunakan:

- **F1-Score** — keseimbangan precision dan recall untuk kelas kegagalan.
- **ROC-AUC** — kemampuan model membedakan kelas berdasarkan skor probabilitas.
- **Average Precision (AP)** — ringkasan kurva precision–recall, berguna saat kelas tidak seimbang.
- **Confusion Matrix** dan classification report.
- **Stratified 5-Fold Cross-Validation** dengan scoring F1 pada data train.
- Waktu pelatihan.

Notebook memilih model dengan **F1-Score test yang lebih tinggi** sebagai model terbaik. Nilai hasil tidak dicantumkan di README ini karena perlu diperoleh dari menjalankan notebook pada lingkungan yang digunakan.

## Visualisasi

Notebook membuat dashboard EDA dan visualisasi evaluasi, termasuk:

- Distribusi kelas target.
- Confusion matrix untuk kedua model.
- Kurva ROC dan precision–recall.
- Feature importance Random Forest.
- Perbandingan F1 antar-fold.
- Learning curve Random Forest.
- Kurva loss MLP.
- Grafik perbandingan metrik kedua model.

Salah satu berkas visualisasi yang disimpan oleh notebook adalah `model_evaluation.png` di direktori output.

## Interpretasi Dampak Bisnis

Notebook menggunakan asumsi biaya berikut untuk membuat **proxy biaya**, bukan perhitungan biaya aktual:

- Kegagalan yang tidak terdeteksi (*false negative*): **Rp50.000.000 per kejadian**.
- Alarm palsu yang memicu inspeksi (*false positive*): **Rp500.000 per kejadian**.

Perkiraan biaya dihitung dari jumlah FN dan FP pada test set untuk masing-masing model. Angka ini adalah asumsi dalam notebook dan perlu divalidasi terhadap biaya operasional nyata sebelum digunakan untuk keputusan bisnis.

Notebook juga membahas pentingnya memantau torsi, keausan alat, dan perbedaan suhu. Rekomendasi ambang yang ditulis di notebook merupakan interpretasi eksploratif, sehingga sebaiknya diuji lagi menggunakan data produksi sebelum diterapkan pada mesin nyata.

## Teknologi dan Dependensi

- Python 3
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- kagglehub

## Cara Menjalankan

Notebook dirancang untuk lingkungan seperti Google Colab karena direktori output ditetapkan ke `/content/outputs`.

1. Buka notebook di Google Colab atau Jupyter yang sesuai.
2. Pastikan dependensi terpasang:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
   ```

3. Jalankan sel notebook secara berurutan.
4. Pastikan akses internet dan akses dataset Kaggle tersedia untuk proses unduh melalui `kagglehub`.
5. Periksa hasil metrik dan file visualisasi pada direktori output.

Jika dijalankan di luar Google Colab, sesuaikan `OUTPUT_DIR` dengan direktori lokal yang dapat ditulis.

## Batasan

- **Bukan model RUL saat ini:** target yang digunakan adalah klasifikasi biner `Machine failure`, bukan prediksi sisa umur mesin.
- Data AI4I merupakan dataset benchmark/sintetis; performa pada data ini belum tentu mencerminkan kondisi pabrik sesungguhnya.
- Kegagalan merupakan kelas minoritas, sehingga F1, recall, precision, dan confusion matrix perlu dilihat bersama; akurasi saja tidak cukup.
- Ambang klasifikasi default dan asumsi biaya perlu disesuaikan dengan toleransi risiko serta kondisi operasional.
- Sebelum integrasi ke SCADA/PLC atau penggunaan real-time, model perlu divalidasi pada data produksi, diuji dari sisi keselamatan, dan dipantau terhadap data drift.

## Pengembangan Selanjutnya

Jika tujuan proyek benar-benar **Remaining Useful Life (RUL)**, langkah berikutnya adalah menggunakan target RUL yang didefinisikan sebagai nilai kontinu atau jumlah siklus/waktu hingga kegagalan, lalu membangun pipeline regresi dan mengevaluasinya dengan metrik seperti MAE dan RMSE. Notebook saat ini belum mengimplementasikan tahapan tersebut.

---

**Jenis proyek:** Machine Learning · Predictive Maintenance · Binary Classification · Industrial AI

