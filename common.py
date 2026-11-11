"""Fungsi bersama untuk skrip 01-05."""
import argparse, json
from pathlib import Path
import pipeline as P

RESULTS = Path(__file__).parent / "results"
MODEL_NAMES = ["Dasar: jam sama minggu lalu", "Dasar: rata-rata 24 jam terakhir", "Ridge", "GBT", "GBT + cuaca"]


def model_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--top-k", type=int, default=60, help="jumlah zona teratas yang dimodelkan (default 60)")
    ap.add_argument("--test-days", type=int, default=7, help="jumlah hari terakhir untuk pengujian (default 7)")
    ap.add_argument("--months", nargs="*", help="bulan yang dipakai, contoh: 2025-01 2025-02 (default: semua yang ada di data/)")
    ap.add_argument("--no-weather", action="store_true", help="jangan pakai data cuaca")
    ap.add_argument("--tag", help="nama hasil (default: k<top-k>), supaya beberapa percobaan tidak saling menimpa")
    return ap


def pick_months(arg):
    avail = P.list_months().key.tolist()
    keys = arg or avail
    miss = [k for k in keys if k not in avail]
    if miss or not keys:
        raise SystemExit(f"Bulan tidak tersedia: {miss or 'tidak ada data'}. Tersedia: {avail}. Tambahkan lewat 01_prepare_data.py.")
    return keys


def save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / name; p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str)); return p
