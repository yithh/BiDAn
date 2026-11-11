"""Langkah 2: model prediksi permintaan TANPA Spark (pandas + scikit-learn Pipeline).
Target: jumlah trip per zona per jam. Pembagian data berdasarkan waktu (hari-hari terakhir untuk uji)."""
import os; os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # hindari galat joblib di Windows
import time, warnings
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pipeline as P, analysis as A, common as C

warnings.filterwarnings("ignore")
a = C.model_args(__doc__).parse_args(); keys = C.pick_months(a.months); tag = a.tag or f"k{a.top_k}"

# ---- 1. data dan fitur (kalender, riwayat/lag, kebijakan, cuaca)
t0 = time.time()
gold = P.load_gold(keys); wx = None if a.no_weather else P.load_weather()
panel, wf = A.build_panel(gold, wx, a.top_k)
t_feat = time.time() - t0

# ---- 2. pembagian berdasarkan waktu
cut = panel.t.max().normalize() - pd.Timedelta(days=a.test_days - 1)
tr, te = panel[panel.t < cut], panel[panel.t >= cut]
if len(tr) < 5000 or te.empty: raise SystemExit("Data terlalu sedikit. Tambah bulan atau kurangi --test-days.")
cal = ["hod", "dow", "wkend", "hol", "cong"]; lag = ["lag1", "lag24", "lag168", "roll3", "roll24"]
print(f"Baris panel {len(panel):,} | latih {len(tr):,} | uji {len(te):,} (mulai {cut:%Y-%m-%d}) | zona {a.top_k} | cuaca {'ya' if wf else 'tidak'}")

# ---- 3. pipeline scikit-learn: pra-pemrosesan + model
def gbt_pipe(cols):
    pre = ColumnTransformer([("keep", "passthrough", ["zid"] + cols)])           # zid di kolom 0 -> kategori
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.08, max_leaf_nodes=31, categorical_features=[0], random_state=0, early_stopping=False)
    return Pipeline([("fitur", pre), ("model", m)])

def ridge_pipe(cols):
    scaled = lag
    binary = ["wkend", "hol", "cong"]
    pre = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"), ["zid", "hod", "dow"]),
                             ("scale", StandardScaler(), scaled), ("bin", "passthrough", binary)])
    return Pipeline([("fitur", pre), ("model", Ridge(alpha=1.0))])

def metrics(y, p): return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(mean_squared_error(y, p) ** .5), R2=float(r2_score(y, p)))
res, timing = {}, {}
res["Dasar: jam sama minggu lalu"] = metrics(te.y, te.lag168); res["Dasar: rata-rata 24 jam terakhir"] = metrics(te.y, te.roll24)
specs = [("Ridge", ridge_pipe(cal + lag), cal + lag), ("GBT", gbt_pipe(cal + lag), cal + lag)]
if wf: specs.append(("GBT + cuaca", gbt_pipe(cal + lag + wf), cal + lag + wf))
preds = {}
for name, pipe, cols in specs:
    use = list(dict.fromkeys(["zid", "hod", "dow"] + cols))
    t1 = time.time(); pipe.fit(tr[use], tr.y); fit = time.time() - t1
    t1 = time.time(); p = pipe.predict(te[use]).clip(0); pred = time.time() - t1
    res[name] = metrics(te.y, p); timing[name] = dict(fit_s=fit, predict_s=pred); preds[name] = p
    print(f"{name:12s} MAE {res[name]['MAE']:.3f} RMSE {res[name]['RMSE']:.3f} R2 {res[name]['R2']:.3f}  latih {fit:.1f}s")
for n in ("Dasar: jam sama minggu lalu", "Dasar: rata-rata 24 jam terakhir", "Ridge", "GBT", "GBT + cuaca"):
    if n in res: print(f"{n:34s}", {k: round(v, 3) for k, v in res[n].items()})

best = "GBT"
C.RESULTS.mkdir(exist_ok=True)
pd.DataFrame({"t": te.t.dt.strftime("%Y-%m-%d %H:%M:%S"), "z": te.z, "y": te.y, "pred": preds[best]}).to_csv(C.RESULTS / f"pred_sklearn_{tag}.csv", index=False)
C.save(f"sklearn_{tag}.json", dict(engine="pandas + scikit-learn", tag=tag, months=keys, top_k=a.top_k, test_days=a.test_days, weather=bool(wf),
        n_rows=len(panel), n_train=len(tr), n_test=len(te), test_from=str(cut.date()), timing=dict(features_s=t_feat, models=timing), metrics=res))
print(f"\nFitur {t_feat:.1f}s. Hasil: results/sklearn_{tag}.json")
