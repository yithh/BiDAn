"""Langkah 1: siapkan data. Membersihkan file TLC (aturan R1-R7) dan meringkasnya per bulan ke folder data/.
Contoh:
  python 01_prepare_data.py                                   # lihat data yang sudah ada
  python 01_prepare_data.py --parquet yellow_tripdata_2025-01.parquet
  python 01_prepare_data.py --download 2025 1 2 3             # unduh Jan-Mar 2025 dari TLC (butuh internet)
  python 01_prepare_data.py --weather weather_2025-01.csv     # tambah cuaca IEM ASOS
"""
import argparse
from pathlib import Path
import pipeline as P

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--parquet", nargs="*", default=[], help="file parquet yellow_tripdata_YYYY-MM.parquet")
ap.add_argument("--download", nargs="+", metavar=("TAHUN", "BULAN"), help="unduh dari TLC: tahun lalu satu atau lebih bulan")
ap.add_argument("--weather", nargs="*", default=[], help="CSV cuaca dari IEM ASOS")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

P.ensure_lookup()
bar = lambda x: print(f"\r  {x*100:5.1f}%", end="", flush=True)
for f in a.parquet:
    y, m = P.guess_year_month(Path(f).name)
    if not y: raise SystemExit(f"Tahun-bulan tidak terbaca dari nama file {f}. Pakai nama asli TLC, misal yellow_tripdata_2025-01.parquet")
    print(f"Memproses {f} -> {P.month_key(y, m)}"); meta = P.process_parquet(f, y, m, bar, Path(f).name)
    print(f"\n  {meta['raw_rows']:,} baris mentah -> {meta['clean_rows']:,} bersih")
if a.download:
    y, ms = int(a.download[0]), [int(x) for x in a.download[1:]] or list(range(1, 13))
    for m in ms:
        k = P.month_key(y, m)
        if (P.MONTHS / k).exists() and not a.overwrite: print(f"{k} sudah ada, dilewati (pakai --overwrite untuk menimpa)"); continue
        print(f"Mengunduh dan memproses {k}"); meta = P.add_month_from_tlc(y, m, bar, bar)
        print(f"\n  {meta['raw_rows']:,} baris mentah -> {meta['clean_rows']:,} bersih")
for w in a.weather:
    print(f"Cuaca: {P.add_weather(w):,} jam tersimpan")

ms = P.list_months()
print("\nData tersimpan di", P.DATA)
print(ms[["key", "raw_rows", "clean_rows", "dibuang_pct"]].to_string(index=False) if len(ms) else "(kosong)")
print("Cuaca:", "ada" if P.WEATHER.exists() else "tidak ada")
