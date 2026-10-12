# -*- coding: utf-8 -*-
"""Klasifikasi token alamat memakai Transformer (BERT).

Fokus modul ini pada dua hal yang diperbaiki dari solusi awal tugas kursus:
1. Tokenisasi level kata yang benar: kota multi kata seperti "New Haven"
   dipecah menjadi ["New", "Haven"] dan tiap katanya diberi label city.
2. Evaluasi yang benar: akurasi dihitung hanya pada posisi valid memakai
   mask labels != -100, bukan dengan memfilter label dulu yang merusak
   kesejajaran posisi.

Pelatihan BERT penuh memakai pustaka transformers dan torch. Script ini
menjalankan pipeline tokenisasi dan logika evaluasi yang persis sama secara
mandiri, plus visualisasi dari data nyata.

Jalankan: python analyze_bert.py
"""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "list_of_real_usa_addresses.json")
CHARTS = os.path.join(BASE, "charts")
FINDINGS = os.path.join(BASE, "findings.json")
os.makedirs(CHARTS, exist_ok=True)

FIELDS = ["address", "city", "state", "zip"]
LABEL2IDX = {"address": 0, "city": 1, "state": 2, "zip": 3}

# Akurasi tercatat dari tugas kursus
ACC_CRF = 0.9684542586750788
ACC_BERT = 0.9918533604887984


def tokenize_entry(entry):
    """Task 1: pecah tiap field menjadi kata, satu label per kata."""
    tokens, labels = [], []
    for field in FIELDS:
        for word in entry[field].split():
            tokens.append(word)
            labels.append(field)
    return tokens, labels


def masked_accuracy(preds, labels, ignore=-100):
    """Task 5: akurasi hanya pada posisi label valid."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    mask = labels != ignore
    correct = np.sum((preds == labels) & mask)
    total = np.sum(mask)
    return float(correct) / float(total) if total else 0.0


def main():
    with open(DATA, "r") as f:
        addresses = json.load(f)

    tokenized = [tokenize_entry(e) for e in addresses]
    multiword_cities = sorted(
        {e["city"] for e in addresses if len(e["city"].split()) > 1}
    )

    # Token per field dan kota multi kata
    field_token_counts = Counter()
    city_word_counts = Counter()
    for toks, labs in tokenized:
        for t, l in zip(toks, labs):
            field_token_counts[l] += 1
    for e in addresses:
        city_word_counts[len(e["city"].split())] += 1

    # Demo logika masking pada contoh kecil
    demo_labels = np.array([0, 0, 1, 1, 2, 3, -100, -100])
    demo_preds = np.array([0, 0, 1, 2, 2, 3, 1, 0])
    demo_acc = masked_accuracy(demo_preds, demo_labels)
    print("demo masked accuracy:", round(demo_acc, 4),
          "(5 benar dari 6 posisi valid)")

    # Chart 1: token per field
    plt.figure(figsize=(7, 4.5))
    labs = list(field_token_counts.keys())
    vals = [field_token_counts[l] for l in labs]
    plt.bar(labs, vals, color="#1D4ED8")
    plt.title("Jumlah token per field alamat")
    plt.xlabel("Field")
    plt.ylabel("Jumlah token")
    for x, v in zip(labs, vals):
        plt.text(x, v + 10, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "01_token_per_field.png"), dpi=120)
    plt.close()

    # Chart 2: kota satu kata vs multi kata
    plt.figure(figsize=(7, 4.5))
    cats = ["1 kata", "2 kata"]
    vals2 = [city_word_counts[1], city_word_counts[2]]
    plt.bar(cats, vals2, color=["#16A34A", "#F59E0B"])
    plt.title("Kota satu kata vs kota multi kata")
    plt.ylabel("Jumlah kota unik")
    for x, v in zip(cats, vals2):
        plt.text(x, v + 1, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "02_kota_multikata.png"), dpi=120)
    plt.close()

    # Chart 3: perbandingan akurasi tercatat
    plt.figure(figsize=(7, 4.5))
    models = ["CRF", "Transformer\n(BERT)"]
    accs = [ACC_CRF, ACC_BERT]
    plt.bar(models, accs, color=["#7C3AED", "#1D4ED8"])
    plt.title("Akurasi token pada data uji (nilai tercatat tugas)")
    plt.ylabel("Akurasi")
    plt.ylim(0.9, 1.0)
    for x, v in zip(models, accs):
        plt.text(x, v + 0.002, "{:.4f}".format(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "03_akurasi_model.png"), dpi=120)
    plt.close()

    # Chart 4: diagram logika masking
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    ax.set_title("Evaluasi memakai mask: hanya posisi label valid yang "
                 "dihitung", fontsize=12, pad=10)
    tokens_demo = ["777", "Brockton", "Avenue", "New", "Haven", "CT",
                   "[PAD]", "[PAD]"]
    labels_demo = ["addr", "addr", "addr", "city", "city", "state",
                   "abaikan", "abaikan"]
    n = len(tokens_demo)
    for i, (tok, lab) in enumerate(zip(tokens_demo, labels_demo)):
        x = i / n
        color = "#E5E7EB" if lab == "abaikan" else "#DBEAFE"
        rect = plt.Rectangle((x + 0.005, 0.35), 1 / n - 0.01, 0.3,
                             facecolor=color, edgecolor="#1E3AED")
        ax.add_patch(rect)
        ax.text(x + 0.5 / n, 0.55, tok, ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(x + 0.5 / n, 0.42, lab, ha="center", va="center",
                fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, "04_masking_logic.png"), dpi=120)
    plt.close()

    findings = {
        "jumlah_alamat": len(addresses),
        "total_token_kata": sum(len(t) for t, _ in tokenized),
        "token_per_field": dict(field_token_counts),
        "kota_multi_kata": len(multiword_cities),
        "contoh_kota_multi_kata": multiword_cities[:8],
        "akurasi_crf_tercatat": ACC_CRF,
        "akurasi_bert_tercatat": ACC_BERT,
        "perbaikan_task_1": "kota multi kata dipecah per kata, satu label "
                            "per kata",
        "perbaikan_task_5": "akurasi memakai mask labels != -100; versi "
                            "lama yang memfilter dulu memberi 0.7067 yang "
                            "salah",
    }
    with open(FINDINGS, "w") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    print("kota multi kata:", len(multiword_cities))
    print("akurasi tercatat BERT:", ACC_BERT)


if __name__ == "__main__":
    main()
