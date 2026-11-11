"""Langkah 3: model prediksi permintaan DENGAN Spark MLlib (mode lokal).
Fitur, pembagian data, dan model dibuat semirip mungkin dengan 02_ml_sklearn.py supaya hasilnya bisa dibandingkan.
Butuh Java 17 dan `pip install pyspark`. Lihat README."""
import os, sys, time
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import pipeline as P, common as C
try:
    import pyspark
    from pyspark.sql import SparkSession, Window, functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
    from pyspark.ml.regression import GBTRegressor, LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator
except ImportError:
    raise SystemExit("pyspark belum terpasang. Jalankan: pip install pyspark==3.5.* (dan pasang Java 17). Lihat README.")

ap = C.model_args(__doc__)
ap.add_argument("--max-iter", type=int, default=300, help="jumlah pohon GBT (default 300, sama dengan scikit-learn)")
ap.add_argument("--partitions", type=int, default=8)
a = ap.parse_args(); keys = C.pick_months(a.months); tag = a.tag or f"k{a.top_k}"

spark = (SparkSession.builder.master("local[*]").appName("taxi-demand-mllib")
         .config("spark.sql.ansi.enabled", "false").config("spark.sql.session.timeZone", "UTC")                 # semua waktu dianggap 'naif' (waktu lokal NYC), tanpa konversi
         .config("spark.sql.shuffle.partitions", str(a.partitions)).config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
         .config("spark.driver.extraJavaOptions", "-Xss64m").config("spark.executor.extraJavaOptions", "-Xss64m")   # tumpukan JVM lebih besar: cegah StackOverflow saat GBT
         .config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ---- 1. baca ringkasan gold (zona-jam) dan cuaca
t0 = time.time()
schema = "pickup_hour timestamp, PULocationID int, trips long, fare_sum double, dist_sum double"
paths = [str(P.MONTHS / k / "gold.csv.gz") for k in keys]
TS = "yyyy-MM-dd HH:mm:ss"     # format waktu eksplisit: tanpa ini Spark bisa membaca waktu sebagai null
gold = (spark.read.csv(paths, header=True, schema=schema, timestampFormat=TS).groupBy("pickup_hour", "PULocationID").agg(F.sum("trips").alias("y")))
n_gold = gold.filter(F.col("pickup_hour").isNotNull()).count()
print(f"Baris gold terbaca: {n_gold:,}")
if n_gold == 0: raise SystemExit("Waktu di gold.csv.gz tidak terbaca (hasil semua null). Periksa format kolom pickup_hour: harus yyyy-MM-dd HH:mm:ss.")
zones = [r[0] for r in gold.groupBy("PULocationID").agg(F.sum("y").alias("s")).orderBy(F.desc("s")).limit(a.top_k).collect()]
gz = gold.filter(F.col("PULocationID").isin(zones)).select(F.col("PULocationID").alias("z"), F.col("pickup_hour").alias("t"), F.col("y").cast("double"))
bounds = gz.agg(F.min("t").alias("a"), F.max("t").alias("b"))
tmax_str = bounds.select(F.date_format("b", "yyyy-MM-dd HH:mm:ss")).first()[0]
tmin_str = bounds.select(F.date_format("a", "yyyy-MM-dd HH:mm:ss")).first()[0]
hours = bounds.select(F.explode(F.sequence("a", "b", F.expr("interval 1 hour"))).alias("t"))       # grid jam lengkap

use_weather = (not a.no_weather) and P.WEATHER.exists()
wcols = ["tmpf", "vsby", "precip_in", "sknt", "has_snow", "has_rain"]
if use_weather:
    _w = pd.read_csv(P.WEATHER); _w = _w.loc[:, ~_w.columns.duplicated()]                      # weather.csv bisa punya kolom weather_hour ganda
    _w["weather_hour"] = pd.to_datetime(_w["weather_hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    w = (spark.createDataFrame(_w[["weather_hour"] + wcols].astype({"has_snow": str, "has_rain": str}))
              .withColumn("weather_hour", F.to_timestamp("weather_hour", TS)))
    w = (w.dropDuplicates(["weather_hour"]).withColumn("has_snow", F.when(F.lower("has_snow") == "true", 1.0).otherwise(0.0))
          .withColumn("has_rain", F.when(F.lower("has_rain") == "true", 1.0).otherwise(0.0)))
    wh = hours.join(w, hours.t == w.weather_hour, "left").drop("weather_hour")
    ff = Window.orderBy("t").rowsBetween(Window.unboundedPreceding, 0)
    for c in wcols: wh = wh.withColumn(c, F.last(c, ignorenulls=True).over(ff))                     # isi maju (ffill), sama seperti pandas
    wh = wh.fillna(0.0, subset=["precip_in", "has_snow", "has_rain"])

# ---- 2. rekayasa fitur dengan window function (per zona)
panel = spark.createDataFrame([(int(z),) for z in zones], ["z"]).crossJoin(hours).join(gz, ["z", "t"], "left").fillna(0.0, subset=["y"])
w_z = Window.partitionBy("z").orderBy("t")
hol = [d.strftime("%Y-%m-%d") for d in USFederalHolidayCalendar().holidays(tmin_str[:10], tmax_str[:10])]
panel = (panel.withColumn("lag1", F.lag("y", 1).over(w_z)).withColumn("lag24", F.lag("y", 24).over(w_z)).withColumn("lag168", F.lag("y", 168).over(w_z))
         .withColumn("roll3", F.avg("y").over(w_z.rowsBetween(-3, -1))).withColumn("roll24", F.avg("y").over(w_z.rowsBetween(-24, -1)))
         .withColumn("hod", F.hour("t").cast("double")).withColumn("dow", ((F.dayofweek("t") + 5) % 7).cast("double"))   # Senin=0 seperti pandas
         .withColumn("wkend", (F.col("dow") >= 5).cast("double"))
         .withColumn("hol", F.col("t").cast("date").cast("string").isin(hol).cast("double"))
         .withColumn("cong", (F.col("t") >= F.lit("2025-01-05").cast("timestamp")).cast("double")))     # congestion pricing NYC
def stage(df, label):
    n = df.count(); print(f"  tahap {label}: {n:,} baris", flush=True); return n
n_grid = stage(panel, f"grid zona x jam (harus {a.top_k} x jumlah jam)")
panel = panel.dropna(subset=["lag168", "lag24", "roll24"])
n_lag = stage(panel, "setelah fitur riwayat (7 hari pertama dibuang)")
if n_lag == 0: raise SystemExit(f"Tidak ada baris setelah fitur riwayat (grid {n_grid:,} baris). Butuh data minimal 8 hari berurutan.")
if use_weather:
    pw = panel.join(wh, "t", "left").dropna(subset=["tmpf"])
    if stage(pw, "setelah digabung cuaca") == 0:
        print("PERINGATAN: gabungan cuaca menghasilkan 0 baris, cuaca dimatikan untuk percobaan ini. Periksa format weather.csv.")
        use_weather = False
    else: panel = pw
# putus rantai rencana kueri (window + join) agar pelatihan GBT iteratif tidak melahirkan plan yang sangat dalam
_pd = panel.withColumn("t", F.date_format("t", TS)).toPandas()                  # waktu lewat string: tanpa konversi zona waktu komputer
panel = spark.createDataFrame(_pd).withColumn("t", F.to_timestamp("t", TS)); del _pd
cal = ["hod", "dow", "wkend", "hol", "cong"]; lag = ["lag1", "lag24", "lag168", "roll3", "roll24"]
cut_str = (pd.Timestamp(tmax_str).normalize() - pd.Timedelta(days=a.test_days - 1)).strftime("%Y-%m-%d")
cut = F.lit(cut_str).cast("timestamp")
train, test = panel.filter(F.col("t") < cut).cache(), panel.filter(F.col("t") >= cut).cache()
n_tr, n_te = train.count(), test.count()                       # memicu perhitungan fitur
t_feat = time.time() - t0
if n_tr < 5000 or n_te == 0:
    raise SystemExit(f"Data terlalu sedikit (latih {n_tr}, uji {n_te}). Bila angkanya 0, kemungkinan waktu cuaca tidak terbaca: coba --no-weather dan kirim hasilnya.")
print(f"Baris panel {n_tr + n_te:,} | latih {n_tr:,} | uji {n_te:,} (mulai {cut_str}) | zona {a.top_k} | cuaca {'ya' if use_weather else 'tidak'} | fitur {t_feat:.1f}s")

# ---- 3. pipeline MLlib
def score(df, col):
    return {m: float(RegressionEvaluator(labelCol="y", predictionCol=col, metricName=m).evaluate(df)) for m in ("mae", "rmse", "r2")}
def fix(d): return dict(MAE=d["mae"], RMSE=d["rmse"], R2=d["r2"])
res = {"Dasar: jam sama minggu lalu": fix(score(test, "lag168")), "Dasar: rata-rata 24 jam terakhir": fix(score(test, "roll24"))}
timing = {}

def gbt_pipeline(cols):
    idx = StringIndexer(inputCol="z", outputCol="zid", handleInvalid="keep")                     # zona sebagai fitur kategori
    asm = VectorAssembler(inputCols=["zid"] + cols, outputCol="features", handleInvalid="skip")
    gbt = GBTRegressor(featuresCol="features", labelCol="y", predictionCol="pred", maxIter=a.max_iter, stepSize=0.08, maxDepth=5,
                       maxBins=max(32, a.top_k + 4), seed=0)
    return Pipeline(stages=[idx, asm, gbt])

def ridge_pipeline():
    idx = StringIndexer(inputCol="z", outputCol="zid", handleInvalid="keep")
    ohe = OneHotEncoder(inputCols=["zid", "hod", "dow"], outputCols=["zid_v", "hod_v", "dow_v"], handleInvalid="keep")
    a1 = VectorAssembler(inputCols=lag, outputCol="lag_raw")
    sc = StandardScaler(inputCol="lag_raw", outputCol="lag_s", withMean=True, withStd=True)
    a2 = VectorAssembler(inputCols=["zid_v", "hod_v", "dow_v", "lag_s", "wkend", "hol", "cong"], outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="y", predictionCol="pred", regParam=0.01, elasticNetParam=0.0, standardization=False)
    return Pipeline(stages=[idx, ohe, a1, sc, a2, lr])

specs = [("Ridge", ridge_pipeline()), ("GBT", gbt_pipeline(cal + lag))]
if use_weather: specs.append(("GBT + cuaca", gbt_pipeline(cal + lag + wcols)))
best_pred = None
for name, pipe in specs:
    t1 = time.time(); model = pipe.fit(train); fit = time.time() - t1
    t1 = time.time(); out = model.transform(test).withColumn("pred", F.greatest(F.col("pred"), F.lit(0.0)))
    res[name] = fix(score(out, "pred")); pred = time.time() - t1
    timing[name] = dict(fit_s=fit, predict_s=pred)
    if name == "GBT": best_pred = out
    print(f"{name:12s} MAE {res[name]['MAE']:.3f} RMSE {res[name]['RMSE']:.3f} R2 {res[name]['R2']:.3f}  latih {fit:.1f}s")

C.RESULTS.mkdir(exist_ok=True)
best_pred.select(F.date_format("t", "yyyy-MM-dd HH:mm:ss").alias("t"), "z", "y", "pred").toPandas().to_csv(C.RESULTS / f"pred_spark_{tag}.csv", index=False)
C.save(f"spark_{tag}.json", dict(engine=f"Spark MLlib {pyspark.__version__} (local[*], {os.cpu_count()} core)", tag=tag, months=keys, top_k=a.top_k, test_days=a.test_days,
        weather=use_weather, n_rows=n_tr + n_te, n_train=n_tr, n_test=n_te, test_from=cut_str,
        timing=dict(features_s=t_feat, models=timing), metrics=res))
print(f"\nFitur {t_feat:.1f}s. Hasil: results/spark_{tag}.json")
spark.stop()
