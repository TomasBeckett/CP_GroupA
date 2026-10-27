# FCIP Indonesia — Dashboard AML + Random Forest

Paket ini menyesuaikan antarmuka dengan referensi FCIP: dashboard overview, alert queue, alert detail, case management, rule management, data operations, regulatory reporting, dan governance/access controls.

## Isi paket

- `dashboard-aml-fraud-integrated.html` — dashboard responsif satu halaman dengan navigasi internal dan interaksi demo.
- `app.py` — backend API Flask untuk memeriksa status pipeline dan menjalankan prediksi.
- `aml_random_forest_pipeline.joblib` — pipeline model tersimpan yang disediakan bersama materi awal.
- `requirements.txt` — dependensi Python yang dibutuhkan backend.

## Menjalankan di Windows

1. Ekstrak ZIP ke sebuah folder.
2. Gunakan Python 3.11 agar environment lebih konsisten dengan pipeline.
3. Buka Command Prompt atau Terminal pada folder hasil ekstrak.
4. Buat dan aktifkan virtual environment, lalu instal dependensi:

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Jalankan backend Flask:

   ```bat
   python app.py
   ```

6. Buka `http://127.0.0.1:5000` pada browser. Jangan membuka file HTML langsung dengan `file://` untuk pengujian prediksi, karena dashboard memanggil endpoint `/api/health` dan `/api/predict` di backend yang sama.

## Fungsi yang tersedia

- Navigasi UI: Dashboard, Alert Queue, Alert Detail, Case Management, Rule Management, Data Operations, Reporting, dan Governance.
- Filter alert dan kasus; pilih baris untuk aksi bulk; ubah status demo; tambahkan kasus, rule, akun demo, dan draft STR; ekspor daftar demo ke CSV.
- `GET /api/health`: memeriksa apakah pipeline dapat dimuat dan mengembalikan metadata model.
- `POST /api/predict`: menerima objek JSON transaksi dan `threshold` antara 0 dan 1; mengembalikan kelas prediksi serta skor probabilitas dari `predict_proba`.

## Batasan yang penting

- Alert, kasus, rule, laporan, akun, kalender, chart, dan metrik operasional merupakan data contoh untuk prototipe. Perubahan pada modul tersebut hanya disimpan sementara di memori browser, bukan database.
- Integrasi model yang benar-benar terhubung ke backend hanya berada di modul **Data Operations** melalui `/api/health` dan `/api/predict`.
- Pipeline yang disediakan dijalankan sebagai model tersimpan. Tidak ada retraining dari antarmuka karena dataset training dan kode training tidak termasuk dalam materi.
- Belum ada autentikasi, otorisasi server, kontrol akses RBAC yang ditegakkan backend, penyimpanan audit persisten, rate limiting, atau deployment production.
- API mengasumsikan label kelas `1` berarti `SUSPICIOUS`. Verifikasi pemetaan label ini terhadap data training.
- Sebelum penggunaan operasional, jalankan di environment yang kompatibel dan evaluasi precision/recall, false positive, calibration, serta threshold pada data validasi yang sesuai. Skor model bukan bukti tindak pidana dan keputusan memerlukan peninjauan manusia.
