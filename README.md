# Parsing Alamat dengan Transformer (BERT)

Klasifikasi tiap token alamat memakai BERT for token classification.
Fokusnya pada dua perbaikan penting dari solusi awal: tokenisasi kota
multi kata yang benar dan evaluasi akurasi memakai mask.

Proyek ini adalah Module 4 dari kursus Applied Information Extraction in
Python (University of Michigan).

## Data

Folder `data` berisi `list_of_real_usa_addresses.json`: 234 alamat nyata di
AS dengan field address, city, state, dan zip. Ada 32 kota multi kata
seperti New Haven, Fall River, dan East Falmouth yang harus dipecah menjadi
token per kata.

## Yang dikerjakan

1.  Tokenisasi level kata: tiap field dipecah per kata dan tiap kata diberi
    satu label. "New Haven" menjadi ["New", "Haven"] dengan label city pada
    keduanya. Solusi awal yang menganggap tiap kota satu token gagal pada
    32 kota multi kata ini. Lihat `charts/02_kota_multikata.png`.
2.  Evaluasi memakai mask pada label: hanya posisi token valid yang
    dihitung, posisi padding dengan indeks abaikan dilewati. Cara lama yang memfilter label dulu
    merusak kesejajaran posisi dan memberi angka 0.7067 yang salah; cara
    benar memberi 0.9919. Lihat `charts/04_masking_logic.png`.
3.  Perbandingan akurasi tercatat pada data uji: CRF 0.9685, Transformer
    (BERT) 0.9919. Lihat `charts/03_akurasi_model.png`.
4.  Distribusi token per field divisualkan pada
    `charts/01_token_per_field.png`.

## Cara menjalankan

```bash
python analyze_bert.py
```

Script ini menjalankan tokenisasi dan logika evaluasi, menyimpan chart ke
folder `charts`, dan menulis ringkasan angka ke `findings.json`. Pelatihan
BERT penuh memakai pustaka transformers dan torch.

## Tools

Python, transformers, torch, NumPy, matplotlib.
