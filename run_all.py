"""Menjalankan seluruh langkah berurutan: 01 (cek data) -> 02 (scikit-learn) -> 03 (Spark, jika tersedia) -> 04 (graph) -> 05 (bandingkan).
Contoh: python run_all.py            |  python run_all.py --top-k 120 --no-spark"""
import argparse, importlib.util, shutil, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--top-k", default="60"); ap.add_argument("--test-days", default="7")
ap.add_argument("--months", nargs="*", default=[]); ap.add_argument("--no-spark", action="store_true"); ap.add_argument("--no-weather", action="store_true")
a = ap.parse_args()
common = ["--top-k", a.top_k, "--test-days", a.test_days] + (["--months", *a.months] if a.months else []) + (["--no-weather"] if a.no_weather else [])
def run(script, *args):
    print(f"\n{'=' * 8} {script} {'=' * 8}", flush=True)
    r = subprocess.run([sys.executable, script, *args]); return r.returncode == 0
steps = [("01_prepare_data.py", []), ("02_ml_sklearn.py", common)]
spark_ok = (not a.no_spark) and importlib.util.find_spec("pyspark") is not None and shutil.which("java") is not None
if spark_ok: steps.append(("03_ml_spark.py", common))
else: print("Spark dilewati (pyspark atau Java tidak ditemukan, atau --no-spark). Lihat README untuk memasangnya.")
steps += [("04_graph_analytics.py", (["--months", *a.months] if a.months else [])), ("05_compare_results.py", [])]
fail = [s for s, args in steps if not run(s, *args)]
print("\nSelesai." + (f" Langkah gagal: {fail}" if fail else " Semua langkah berhasil. Hasil ada di folder results/."))
