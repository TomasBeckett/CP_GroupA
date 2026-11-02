# FCIP Indonesia — Dashboard AML + Random Forest

Paket ini menyesuaikan antarmuka dengan referensi FCIP: dashboard overview, alert queue, alert detail, case management, rule management, data operations, regulatory reporting, dan governance/access controls.

## Isi paket

- `dashboard/dashboard-aml-fraud-integrated.html` — dashboard responsif satu halaman dengan navigasi internal dan interaksi demo.
- `app.py` — backend API Flask untuk memeriksa status pipeline dan menjalankan prediksi.
- `models/aml_random_forest_pipeline.joblib` — pipeline model tersimpan yang disediakan bersama materi awal.
- `train_model.py` — skrip untuk melatih dan mengevaluasi Random Forest dari label transaksi.
- `models/aml_random_forest_pipeline_trained.joblib` — hasil training terbaru yang dipakai backend.
- `models/aml_training_report.json` — metrik dan rincian evaluasi holdout temporal.
- `database/database_schema.sql` — SQL schema untuk PostgreSQL.
- `.env.example` — template konfigurasi lokal.
- `requirements.txt` — dependensi Python yang dibutuhkan backend.
- `../render.yaml` — konfigurasi deploy backend di Render dengan secret diisi terpisah.

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
- `GET /api/datasets/status`: memeriksa status dataset PostgreSQL jika koneksi database dikonfigurasi, atau lima CSV sumber sebagai fallback lokal.
- `POST /api/datasets/predict`: membentuk fitur transaksi dari CSV dan mengembalikan prediksi satu halaman. Body JSON menerima `offset` (default `0`), `limit` (default `100`, maksimum `1000`), dan `threshold` antara `0` dan `1`.
- `GET /api/db/health`: memeriksa koneksi PostgreSQL, tabel master, dan skema tabel transaksi/prediksi.
- `POST /api/datasets/upload`: menerima CSV transaksi, membuat prediksi, lalu mengganti dataset aktif dalam satu transaksi PostgreSQL.
- `GET /api/dashboard/summary` dan `GET /api/alerts`: mengembalikan ringkasan risiko dan alert model yang sudah dipaginasi berdasarkan skor tertinggi.

Jika PostgreSQL belum dikonfigurasi, dashboard **Data Operations** menyediakan pemeriksaan CSV dan prediksi batch dengan offset/jumlah baris. Dataset fallback ini dibaca dari folder `data` di root repository secara default. Untuk menyimpan CSV di folder lain, set environment variable sebelum menjalankan backend:

   ```bat
   set AML_DATA_DIR=D:\data\aml
   python app.py
   ```

Folder tersebut harus berisi file dengan nama dan kolom sesuai dataset FCIP di atas. Backend menghubungkan transaksi ke akun, customer, device, dan merchant, lalu membentuk 46 fitur model. Hasil fitur disimpan di cache proses; perubahan ukuran atau waktu modifikasi file akan memicu pembacaan ulang. Prediksi hanya memproses baris yang diminta, tetapi pembentukan fitur memerlukan pemuatan dan pemrosesan seluruh riwayat transaksi.

Secara default, seluruh CSV referensi dan transaksi berada di folder `data` pada root repository.

Beberapa fitur tidak tersedia langsung pada CSV dan dihitung berdasarkan asumsi yang didokumentasikan oleh endpoint. Secara khusus, `hour`/`day_of_week` menggunakan waktu UTC, `log_amount_idr` dihitung sebagai `log1p(amount_idr)`, fitur historis dihitung per `account_id`, dan customer yang tidak dapat dihubungkan secara unik melalui akun/entity dibiarkan kosong. Pastikan asumsi ini sama dengan pipeline training sebelum memakai skor untuk keputusan. Endpoint batch mengembalikan maksimal 1.000 baris per permintaan agar hasil dapat dipaginasi.

## Melatih model Random Forest

Dataset transaksi memiliki label sintetis `scenario_label`: `NORMAL`, `SUSPICIOUS`, dan `EDGE_CASE`. Untuk membuat indikator potensi fraud dari data yang tersedia, training memetakan `NORMAL=0` dan `SUSPICIOUS=1`; `SUSPICIOUS` dipakai sebagai **proxy potensi fraud**, bukan label fraud terkonfirmasi. `EDGE_CASE` dikecualikan karena merupakan kelas ketiga yang maknanya berbeda. `synthetic_pattern` tidak digunakan sebagai fitur maupun target.

Setelah environment pada langkah di atas aktif, jalankan dari folder paket:

   ```bat
   python train_model.py
   ```

Training membentuk fitur dengan backend yang sama, membagi data secara kronologis (80% waktu awal untuk training, 20% waktu terbaru untuk pengujian), melaporkan precision/recall/F1 untuk kelas `POTENTIAL_FRAUD_PROXY`, balanced accuracy, ROC-AUC, average precision dan confusion matrix, lalu menyimpan:

- `models/aml_random_forest_pipeline_trained.joblib` — model hasil training, otomatis dipakai backend.
- `models/aml_training_report.json` — detail split, jumlah kelas, metrik evaluasi, dan batasan data.
- `database/database_schema.sql` — tabel transaksi aktif, prediksi, dan status upload untuk PostgreSQL.
- `.env.example` — contoh konfigurasi koneksi lokal; salin ke `.env` di folder root repository dan isi password hanya di file lokal tersebut.

File model awal `models/aml_random_forest_pipeline.joblib` tidak ditimpa. Backend menggunakan model hasil training; untuk memilih file model lain, set `AML_MODEL_PATH` ke path model tersebut sebelum menjalankan `app.py`. Untuk menyesuaikan jumlah pohon dan lokasi output, lihat `python train_model.py --help`.

**Peringatan evaluasi:** label dan transaksi yang disediakan bersifat sintetis. Metrik mengukur kecocokan dengan label skenario sintetis pada holdout waktu, bukan efektivitas deteksi fraud nyata. Skor menunjukkan kemiripan dengan pola `SUSPICIOUS` sintetis; itu bukan probabilitas fraud terkonfirmasi. Semua alert perlu ditinjau manusia. Fitur dan transformasinya juga harus diverifikasi terhadap spesifikasi training sebelum dipakai untuk keputusan operasional.

## Menghubungkan PostgreSQL dan upload transaksi

1. Pastikan tabel master `aml.customers`, `aml.accounts`, `aml.devices`, dan `aml.merchants` sudah terisi.
2. Di pgAdmin, buka Query Tool untuk database `aml_fcip`, buka `database\database_schema.sql` dari folder paket lalu jalankan isinya. Skrip membuat `aml.active_transactions`, `aml.model_predictions`, dan `aml.dataset_state`; tidak menghapus tabel master.
3. Salin `.env.example` ke file `.env` di folder root repository (`CP_GroupA\.env`), lalu masukkan password PostgreSQL lokal pada `AML_DB_PASSWORD`. File `.env` diabaikan Git; jangan unggah atau bagikan isinya.
4. Instal dependensi yang baru dan jalankan aplikasi dari PowerShell:

   ```powershell
   cd H:\tomas\Github\CP_GroupA\aml_dashboard_integration_package
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe app.py
   ```

   Jika virtual environment belum dibuat, ikuti perintah pembuatan `.venv` pada bagian menjalankan di atas.
5. Buka `http://127.0.0.1:5000`, pilih **Data Operations**, lalu cek status database. Setelah status **Ready**, pilih satu CSV transaksi dan klik **Upload & Predict**.

Backend menguji model dan tabel referensi sebelum menyimpan upload. Seluruh transaksi dan skor prediksi baru ditulis dalam satu transaksi PostgreSQL: upload yang tidak lolos validasi/prediksi tidak mengganti dataset aktif lama. Upload sukses menggantikan transaksi dan hasil prediksi aktif yang lama. `GET /api/db/health` memeriksa koneksi/skema, `POST /api/datasets/upload` menerima multipart `file` dan `threshold`, `GET /api/dashboard/summary` mengembalikan hitungan indikasi potensi fraud serta rata-rata skor, dan `GET /api/alerts` memberi pagination alert dari skor tertinggi. Untuk integrasi, API mempertahankan field `prediction` bernilai `SUSPICIOUS` demi kompatibilitas dan menambahkan `potential_fraud_class` (`POTENTIAL_FRAUD` atau `NOT_FLAGGED`).

Untuk pengembangan lokal, contoh `.env` memakai user `postgres`. Jangan gunakan superuser untuk deployment; buat role aplikasi dengan hak akses minimum, password kuat, dan konfigurasi rahasia di luar Git.

## Deploy untuk diuji teman (Supabase + Render)

Dengan setup ini, teman hanya membuka URL dashboard. PostgreSQL berjalan di Supabase dan Flask berjalan di Render; mereka tidak perlu memasang PostgreSQL atau menjalankan backend sendiri.

1. Buat project PostgreSQL di Supabase. Salin kredensial koneksi **Session pooler** dari menu Connect. Gunakan nilai host, port, dan user yang ditampilkan Supabase, nama database `postgres`, serta SSL.
2. Pada database Supabase, buat/impor tabel master `aml.customers`, `aml.accounts`, `aml.devices`, dan `aml.merchants` dengan nama kolom yang dipakai backend. Gunakan CSV referensi di root repository `data` (customer, account, device, merchant), lalu jalankan `database\database_schema.sql` dari folder paket pada database tersebut. Pastikan tabel master sudah memiliki data sebelum upload transaksi.
3. Push perubahan kode ke GitHub. Di Render pilih **New > Blueprint**, hubungkan repository, lalu deploy `render.yaml`. Konfigurasi ini menetapkan root paket, Gunicorn, Python 3.11, dan SSL untuk koneksi database.
4. Di pengaturan Environment service Render, isi variabel `AML_DB_HOST`, `AML_DB_PORT`, `AML_DB_NAME=postgres`, `AML_DB_USER`, `AML_DB_PASSWORD`, `AML_DB_SSLMODE=require`, `AML_APP_USERNAME`, dan `AML_APP_PASSWORD`. Ambil kredensial database dari Supabase; buat username/password dashboard yang kuat dan bagikan hanya kepada teman penguji. Simpan semua nilai rahasia di Render, bukan di repository atau chat.
5. Setelah deploy selesai, buka URL Render dan login dengan username/password dashboard yang diatur. Cek **Check Database** hingga **Ready**, lalu coba upload CSV kecil terlebih dahulu. Upload yang sukses menggantikan dataset transaksi aktif di database cloud.

`AML_APP_USERNAME` dan `AML_APP_PASSWORD` mengaktifkan HTTP Basic login pada dashboard dan seluruh API. Jika keduanya tidak diisi, aplikasi tidak meminta login; jangan membagikan URL service publik tanpa mengaktifkan keduanya. Gunakan hanya data sintetis untuk pengujian dan jangan gunakan kredensial PostgreSQL `postgres` untuk aplikasi cloud permanen—buat role database khusus dengan hak akses minimum.

Upload contoh 350 ribu transaksi dapat memerlukan waktu lama dan mungkin melewati batas waktu request hosting. Mulai dari CSV kecil untuk membuktikan alur end-to-end. Untuk demonstrasi besar atau penggunaan berkelanjutan, pindahkan upload/prediksi ke background job dan gunakan hosting/database yang kapasitasnya sesuai.

## Batasan yang penting

- Kasus, rule, laporan, akun demo, kalender, chart, dan metrik operasional merupakan data contoh untuk prototipe. Perubahan pada modul tersebut hanya disimpan sementara di memori browser, bukan database.
- Metrik overview, alert hasil model, dan upload transaksi terhubung ke PostgreSQL; kasus investigasi, rule, laporan kepatuhan, dan governance masih demo dan tidak boleh dianggap data operasional.
- Integrasi model yang benar-benar terhubung ke backend berada di modul **Data Operations** melalui `/api/health`, `/api/predict`, `/api/db/health`, `/api/datasets/upload`, `/api/dashboard/summary`, dan `/api/alerts`.
- Upload transaksi CSV dan prediksinya terhubung ke PostgreSQL setelah skema serta `.env` lokal dikonfigurasi. Tabel master customer, account, device, dan merchant tetap menjadi sumber referensi.
- Skrip training dijalankan terpisah dari dashboard; tidak ada retraining otomatis saat backend atau antarmuka dimulai.
- Login HTTP Basic opsional melindungi dashboard/API untuk pengujian bersama, tetapi belum ada RBAC, akun per pengguna, audit persisten, atau rate limiting; ini belum merupakan deployment production.
- API memetakan kelas `1` dari model ke indikasi `POTENTIAL_FRAUD` dengan dasar label sintetis `SUSPICIOUS`; hasilnya bukan konfirmasi bahwa transaksi adalah fraud.
- Sebelum penggunaan operasional, jalankan di environment yang kompatibel dan evaluasi precision/recall, false positive, calibration, serta threshold pada data validasi yang sesuai. Skor model bukan bukti tindak pidana dan keputusan memerlukan peninjauan manusia.
