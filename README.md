# Klasterisasi Segmentasi Limit Finansial

Aplikasi Streamlit untuk mengelompokkan pelanggan kartu kredit menggunakan K-Means.

## Isi folder
- `app.py` — kode utama aplikasi Streamlit.
- `Credit Card Customer Data.csv` — dataset yang digunakan.
- `requirements.txt` — daftar library Python.

## Menjalankan di komputer (Windows / VS Code)

1. Pastikan Python sudah terpasang.
2. Buka folder proyek ini di VS Code.
3. Buka Terminal di VS Code.
4. (Opsional, disarankan) Buat virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

5. Instal library:

   ```bash
   pip install -r requirements.txt
   ```

6. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

7. Browser akan membuka aplikasi Streamlit. Jika tidak terbuka otomatis, klik alamat lokal yang muncul di terminal (biasanya `http://localhost:8501`).

## Deploy ke Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Unggah `app.py`, `requirements.txt`, dan `Credit Card Customer Data.csv` ke repository.
3. Buka Streamlit Community Cloud dan hubungkan akun GitHub.
4. Pilih **New app**, pilih repository, branch, dan file utama `app.py`.
5. Klik **Deploy**.

Catatan: jika repository dibuat public, dataset juga dapat diakses publik. Pastikan dataset memang boleh dibagikan.
