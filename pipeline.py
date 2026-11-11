"""Pipeline data: unduh/baca file TLC Yellow Taxi (parquet), bersihkan, ringkas, simpan per bulan.
Tidak bergantung pada Streamlit, jadi bisa dites dan dipakai dari skrip biasa."""
import json, os, re, shutil, tempfile, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("TAXI_DATA", ROOT / "data"))
MONTHS = DATA / "months"
WEATHER = DATA / "weather" / "weather.csv"
LOOKUP = DATA / "taxi_zone_lookup.csv"
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{y}-{m:02d}.parquet"
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

NEED = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
        "fare_amount", "total_amount", "PULocationID", "DOLocationID"]
RULES = ["R1 Pickup di luar bulan data", "R2 Dropoff <= pickup", "R3 Durasi < 1 menit",
         "R4 Durasi > 180 menit", "R5 Jarak <= 0 mil atau > 100 mil",
         "R6 Kecepatan rata-rata > 80 mph", "R7 Tarif/total <= 0 atau fare > 500 USD"]


def month_key(y, m): return f"{int(y)}-{int(m):02d}"


# ---------------------------------------------------------------- pembersihan
def clean_batch(df, start, end):
    """Aturan R1-R7 sama dengan cleaning.py. Tiap baris dihitung pada aturan PERTAMA yang dilanggar."""
    dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    mph = df.trip_distance / (dur / 60).where(dur > 0)
    bad = [
        (df.tpep_pickup_datetime < start) | (df.tpep_pickup_datetime >= end),
        dur <= 0, dur < 1, dur > 180,
        (df.trip_distance <= 0) | (df.trip_distance > 100),
        mph > 80,
        (df.fare_amount <= 0) | (df.total_amount <= 0) | (df.fare_amount > 500),
    ]
    keep = pd.Series(True, index=df.index); counts = []
    for b in bad:
        hit = b.fillna(False) & keep
        counts.append(int(hit.sum())); keep &= ~b.fillna(False)
    return df[keep], counts


def _agg(c):
    c = c.assign(pickup_hour=c.tpep_pickup_datetime.dt.floor("h"))
    gold = (c.groupby(["pickup_hour", "PULocationID"])
              .agg(trips=("fare_amount", "size"), fare_sum=("fare_amount", "sum"),
                   dist_sum=("trip_distance", "sum")).reset_index())
    od = (c.groupby(["PULocationID", "DOLocationID"])
            .agg(trips=("fare_amount", "size"), fare_sum=("fare_amount", "sum"),
                 dist_sum=("trip_distance", "sum")).reset_index())
    return gold, od


def _combine(parts, keys):
    return pd.concat(parts).groupby(keys, as_index=False).sum()


def process_batches(batches, year, month, total_rows=None, progress=None, source=""):
    """batches: iterator DataFrame (kolom NEED). Menulis ringkasan ke data/months/<tahun-bulan>/."""
    start = pd.Timestamp(year=int(year), month=int(month), day=1)
    end = start + pd.offsets.MonthBegin(1)
    raw = 0; counts = np.zeros(len(RULES), dtype=np.int64); golds = []; ods = []
    for df in batches:
        raw += len(df)
        c, cnt = clean_batch(df, start, end); counts += cnt
        if len(c):
            g, o = _agg(c); golds.append(g); ods.append(o)
        if len(golds) > 20:
            golds = [_combine(golds, ["pickup_hour", "PULocationID"])]
            ods = [_combine(ods, ["PULocationID", "DOLocationID"])]
        if progress and total_rows: progress(min(raw / total_rows, 1.0))
    if not golds:
        raise ValueError("Tidak ada baris yang lolos pembersihan. Cek tahun dan bulan yang dipilih.")
    gold = _combine(golds, ["pickup_hour", "PULocationID"]); od = _combine(ods, ["PULocationID", "DOLocationID"])
    d = MONTHS / month_key(year, month); d.mkdir(parents=True, exist_ok=True)
    gold.to_csv(d / "gold.csv.gz", index=False); od.to_csv(d / "od.csv.gz", index=False)
    meta = dict(year=int(year), month=int(month), raw_rows=int(raw), clean_rows=int(gold.trips.sum()),
                rules=dict(zip(RULES, map(int, counts))), source=source)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def _resolve_columns(names):
    low = {n.lower(): n for n in names}
    out = {}
    for n in NEED:
        if n.lower() in low: out[n] = low[n.lower()]
    miss = [n for n in NEED if n not in out]
    if miss:
        raise ValueError("Kolom wajib tidak ada di file: " + ", ".join(miss) +
                         ". Data sebelum Juli 2016 tidak punya LocationID zona, jadi tidak didukung.")
    return out


def parquet_batches(path, batch_rows=500_000):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = _resolve_columns(pf.schema_arrow.names); inv = {v: k for k, v in cols.items()}

    def gen():
        for b in pf.iter_batches(batch_size=batch_rows, columns=list(cols.values())):
            df = b.to_pandas().rename(columns=inv)
            for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
                df[c] = pd.to_datetime(df[c])
            yield df
    return gen(), pf.metadata.num_rows


def process_parquet(path, year, month, progress=None, source=""):
    gen, total = parquet_batches(path)
    return process_batches(gen, year, month, total, progress, source or Path(path).name)


# ---------------------------------------------------------------- unduh
def download(url, dest, progress=None):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "taxi-dashboard"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            size = int(r.headers.get("Content-Length") or 0); got = 0
            while chunk := r.read(1 << 20):
                f.write(chunk); got += len(chunk)
                if progress and size: progress(got / size)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gagal mengunduh ({e.code}). Data bulan itu mungkin belum dirilis TLC: {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Tidak bisa terhubung ke server TLC: {e.reason}")


def add_month_from_tlc(year, month, progress_dl=None, progress_proc=None):
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "x.parquet"
        download(TRIP_URL.format(y=int(year), m=int(month)), p, progress_dl)
        return process_parquet(p, year, month, progress_proc, source=TRIP_URL.format(y=int(year), m=int(month)))


def guess_year_month(name):
    m = re.search(r"(20\d\d)[-_](\d\d)", name or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def ensure_lookup():
    if not LOOKUP.exists():
        LOOKUP.parent.mkdir(parents=True, exist_ok=True)
        download(LOOKUP_URL, LOOKUP)
    return pd.read_csv(LOOKUP).rename(columns={"LocationID": "PULocationID"})


# ---------------------------------------------------------------- baca & kelola
def list_months():
    rows = []
    for p in sorted(MONTHS.glob("*/meta.json")):
        m = json.loads(p.read_text()); m["key"] = p.parent.name; rows.append(m)
    if not rows: return pd.DataFrame(columns=["key", "year", "month", "raw_rows", "clean_rows"])
    df = pd.DataFrame(rows); df["dibuang_pct"] = (1 - df.clean_rows / df.raw_rows) * 100
    return df


def data_version():
    ps = list(MONTHS.glob("*/meta.json")) + ([WEATHER] if WEATHER.exists() else [])
    return max((p.stat().st_mtime for p in ps), default=0)


def _load(name, keys, parse=None):
    parts = [pd.read_csv(MONTHS / k / name, parse_dates=parse) for k in keys if (MONTHS / k / name).exists()]
    if not parts: return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_gold(keys):
    g = _load("gold.csv.gz", keys, ["pickup_hour"])
    if g.empty: return g
    z = ensure_lookup()[["PULocationID", "Borough", "Zone"]]
    return g.merge(z, on="PULocationID", how="left")


def load_od(keys):
    o = _load("od.csv.gz", keys)
    return o.groupby(["PULocationID", "DOLocationID"], as_index=False).sum() if len(o) else o


def delete_month(key):
    shutil.rmtree(MONTHS / key, ignore_errors=True)


# ---------------------------------------------------------------- cuaca (IEM ASOS)
def clean_weather(w):
    w = w.copy().replace("M", np.nan)
    local = (pd.to_datetime(w["valid"]).dt.tz_localize("UTC")
               .dt.tz_convert("America/New_York").dt.tz_localize(None))   # IEM = UTC -> waktu NYC
    out = pd.DataFrame({"weather_hour": local.dt.round("h")})
    for c in ("tmpf", "sknt", "vsby", "p01i"):
        s = w[c] if c in w else np.nan
        if c == "p01i": s = pd.Series(s).replace("T", "0.001")
        out[c] = pd.to_numeric(s, errors="coerce")
    wx = w["wxcodes"].fillna("") if "wxcodes" in w else pd.Series("", index=w.index)
    out["precip_in"] = out.pop("p01i")
    out["has_snow"] = wx.str.contains("SN").values; out["has_rain"] = wx.str.contains("RA").values
    return out.drop_duplicates("weather_hour", keep="last")


def add_weather(csv_file):
    new = clean_weather(pd.read_csv(csv_file, comment="#", low_memory=False))
    WEATHER.parent.mkdir(parents=True, exist_ok=True)
    if WEATHER.exists():
        new = pd.concat([pd.read_csv(WEATHER, parse_dates=["weather_hour"]), new])
    new = new.drop_duplicates("weather_hour", keep="last").sort_values("weather_hour")
    new.to_csv(WEATHER, index=False); return len(new)


def load_weather():
    return pd.read_csv(WEATHER, parse_dates=["weather_hour"]) if WEATHER.exists() else pd.DataFrame()
