# CP Group A — AML & Fraud Risk Dashboard

Proyek menggabungkan dashboard AML, API Flask, model Random Forest, dataset transaksi sintetis, dan PostgreSQL untuk demo lokal.

## Struktur repository

```text
.
├── data/                              # Dataset transaksi dan tabel referensi CSV
├── aml_dashboard_integration_package/
│   ├── app.py, train_model.py         # Backend dan training
│   ├── dashboard/                     # UI HTML
│   ├── models/                        # Pipeline model dan laporan training
│   ├── database/                      # SQL schema/migration (bukan database server)
│   └── requirements.txt, .env.example, dokumentasi
├── render.yaml                        # Blueprint deploy backend (opsional)
└── README.md
```

`.env`, virtual environment, dan konfigurasi workspace VS Code bersifat lokal dan tidak dimasukkan ke Git.

## Menjalankan lokal (Windows)

Gunakan Python 3.11 dari terminal pada folder repository:

```powershell
cd ...:\...\CP_GroupA\aml_dashboard_integration_package
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

1. Salin `aml_dashboard_integration_package\.env.example` ke `CP_GroupA\.env`
2. isi kredensial PostgreSQL lokal. Jalankan `aml_dashboard_integration_package\database\database_schema.sql` pada database yang dipakai
3. lalu buka `http://127.0.0.1:5000`. Dataset default dibaca dari `CP_GroupA\data`; pastikan tabel master PostgreSQL sudah tersedia sebelum mengunggah CSV transaksi.

Petunjuk lengkap, training model, database, serta opsi deployment tersedia di [README integrasi](aml_dashboard_integration_package/README_integrasi_AML.md).
