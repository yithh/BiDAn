"""Langkah 5: bandingkan hasil tanpa Spark (scikit-learn) dan dengan Spark MLlib: akurasi dan waktu.
Membaca results/sklearn_*.json dan results/spark_*.json, menulis results/comparison.csv dan comparison.md."""
import json, glob
import pandas as pd
import common as C

def load(prefix):
    out = {}
    for f in sorted(glob.glob(str(C.RESULTS / f"{prefix}_*.json"))):
        j = json.load(open(f)); out[j["tag"]] = j
    return out
sk, sp = load("sklearn"), load("spark")
if not sk: raise SystemExit("Belum ada hasil. Jalankan 02_ml_sklearn.py dulu.")
rows, rt = [], []
for tag, j in sk.items():
    s = sp.get(tag)
    for m, v in j["metrics"].items():
        r = dict(percobaan=tag, model=m, MAE_sklearn=v["MAE"], RMSE_sklearn=v["RMSE"], R2_sklearn=v["R2"])
        if s and m in s["metrics"]: r.update(MAE_spark=s["metrics"][m]["MAE"], RMSE_spark=s["metrics"][m]["RMSE"], R2_spark=s["metrics"][m]["R2"])
        rows.append(r)
        if m in j["timing"]["models"]:
            t = dict(percobaan=tag, model=m, baris_latih=j["n_train"], zona=j["top_k"], fit_sklearn_s=j["timing"]["models"][m]["fit_s"])
            if s and m in s["timing"]["models"]: t["fit_spark_s"] = s["timing"]["models"][m]["fit_s"]
            rt.append(t)
    rt.append(dict(percobaan=tag, model="(pembuatan fitur)", baris_latih=j["n_train"], zona=j["top_k"], fit_sklearn_s=j["timing"]["features_s"],
                   **({"fit_spark_s": s["timing"]["features_s"]} if s else {})))
acc, tim = pd.DataFrame(rows), pd.DataFrame(rt)
acc.to_csv(C.RESULTS / "comparison.csv", index=False); tim.to_csv(C.RESULTS / "comparison_time.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("AKURASI (data uji)"); print(acc.to_string(index=False)); print("\nWAKTU (detik)"); print(tim.to_string(index=False))
md = ["# Perbandingan tanpa Spark dan dengan Spark MLlib", "", "## Akurasi", acc.round(3).to_markdown(index=False) if hasattr(acc, "to_markdown") else acc.round(3).to_string(index=False),
      "", "## Waktu (detik)", tim.round(2).to_markdown(index=False) if hasattr(tim, "to_markdown") else tim.round(2).to_string(index=False)]
try: (C.RESULTS / "comparison.md").write_text("\n".join(md))
except Exception: pass
if not sp: print("\nCatatan: hasil Spark belum ada. Jalankan 03_ml_spark.py (butuh Java 17 dan pyspark).")
