"""Analisis untuk dasbor: pola waktu, zona, model prediksi, dan graph. Tanpa Streamlit."""
import os; os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # hindari galat joblib di Windows
import numpy as np, pandas as pd

PERIODS = {"Semua jam": (0, 24), "Dini hari (00-05)": (0, 6), "Pagi (06-10)": (6, 11),
           "Siang (11-15)": (11, 16), "Sore (16-20)": (16, 21), "Malam (21-23)": (21, 24)}
DAYS = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]


def hourly(gold):
    h = gold.groupby("pickup_hour", as_index=False).trips.sum()
    h["hod"] = h.pickup_hour.dt.hour; h["dow"] = h.pickup_hour.dt.dayofweek
    h["date"] = h.pickup_hour.dt.normalize(); h["weekend"] = h.dow >= 5
    return h


def weekday_weekend(h):
    return h.groupby(["weekend", "hod"]).trips.mean().unstack(0).rename(columns={False: "Hari kerja", True: "Akhir pekan"})


def heat(h):
    return h.groupby(["dow", "hod"]).trips.mean().unstack(1).reindex(range(7))


def daily(h):
    return h.groupby("date").trips.sum()


def top_zones(gold, period="Semua jam", borough=None, n=10):
    a, b = PERIODS[period]
    g = gold[(gold.pickup_hour.dt.hour >= a) & (gold.pickup_hour.dt.hour < b)]
    if borough and borough != "Semua": g = g[g.Borough == borough]
    return (g.groupby(["Zone", "Borough"], as_index=False).trips.sum()
              .sort_values("trips", ascending=False).head(n))


# ------------------------------------------------------------------ prediksi
def build_panel(gold, weather=None, top_k=60):
    zones = gold.groupby("PULocationID").trips.sum().nlargest(top_k).index
    g = gold[gold.PULocationID.isin(zones)]
    idx = pd.date_range(g.pickup_hour.min(), g.pickup_hour.max(), freq="h")
    piv = (g.pivot_table(index="pickup_hour", columns="PULocationID", values="trips", aggfunc="sum")
             .reindex(idx).reindex(columns=zones).fillna(0))
    feats = {"y": piv, "lag1": piv.shift(1), "lag24": piv.shift(24), "lag168": piv.shift(168),
             "roll3": piv.shift(1).rolling(3).mean(), "roll24": piv.shift(1).rolling(24).mean()}
    d = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    d.index.names = ["t", "z"]; d = d.reset_index()
    d["zid"] = d.z.astype("category").cat.codes
    d["hod"] = d.t.dt.hour; d["dow"] = d.t.dt.dayofweek; d["wkend"] = (d.dow >= 5).astype(int)
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(idx.min(), idx.max())
    d["hol"] = d.t.dt.normalize().isin(hol).astype(int)
    d["cong"] = (d.t >= "2025-01-05").astype(int)       # congestion pricing NYC mulai 5 Jan 2025
    wf = []
    if weather is not None and len(weather):
        w = weather.drop_duplicates("weather_hour").set_index("weather_hour")[["tmpf", "vsby", "precip_in", "sknt", "has_snow", "has_rain"]]
        w = w.astype(float).reindex(idx).ffill()
        d = d.merge(w.rename_axis("t").reset_index(), on="t", how="left")
        d[["precip_in", "has_snow", "has_rain"]] = d[["precip_in", "has_snow", "has_rain"]].fillna(0)
        d = d.dropna(subset=["tmpf"]); wf = ["tmpf", "vsby", "precip_in", "sknt", "has_snow", "has_rain"]
    d = d.dropna(subset=["lag168", "lag24", "roll24"])
    return d, wf


def train_eval(panel, weather_feats, test_days=7, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    cut = panel.t.max().normalize() - pd.Timedelta(days=test_days - 1)
    tr, te = panel[panel.t < cut], panel[panel.t >= cut]
    if len(tr) < 5000 or te.empty:
        raise ValueError("Data terlalu sedikit untuk dilatih. Tambah bulan data atau perkecil hari uji.")
    base = ["zid", "hod", "dow", "wkend", "hol", "cong"]; lag = ["lag1", "lag24", "lag168", "roll3", "roll24"]
    res = {}
    def ev(n, p): res[n] = dict(MAE=mean_absolute_error(te.y, p), RMSE=mean_squared_error(te.y, p) ** .5, R2=r2_score(te.y, p))
    ev("Dasar: jam sama minggu lalu", te.lag168); ev("Dasar: rata-rata 24 jam terakhir", te.roll24)
    def gbt(f):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.08, max_leaf_nodes=31,
                                          categorical_features=[f.index("zid")], random_state=seed, early_stopping=False).fit(tr[f], tr.y)
        return m, m.predict(te[f]).clip(0)
    f1 = base + lag; m1, p1 = gbt(f1); ev("GBT: kalender + riwayat", p1)
    best, pred, feats = m1, p1, f1
    if weather_feats:
        f2 = f1 + weather_feats; m2, p2 = gbt(f2); ev("GBT: kalender + riwayat + cuaca", p2)
        if res["GBT: kalender + riwayat + cuaca"]["MAE"] < res["GBT: kalender + riwayat"]["MAE"]: best, pred, feats = m2, p2, f2
    s = te.sample(min(len(te), 15000), random_state=seed)
    pi = permutation_importance(best, s[feats], s.y, n_repeats=2, random_state=seed, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    series = te.assign(pred=pred).groupby("t")[["y", "pred"]].sum()
    detail = te[["t", "z", "y", "lag168", "roll24", "hod", "dow"]].assign(pred=pred).reset_index(drop=True)
    hist = tr[["z", "y"]].reset_index(drop=True)
    return (pd.DataFrame(res).T, imp, series,
            dict(n_train=len(tr), n_test=len(te), test_from=cut, detail=detail, hist=hist))


# ------------------------------------------------------------------ graph
def graph_analysis(od, lookup, top_edges=150):
    import networkx as nx
    nm = lookup.set_index("PULocationID").Zone.to_dict(); bo = lookup.set_index("PULocationID").Borough.to_dict()
    tot = od.trips.sum()
    g = od[(od.PULocationID != od.DOLocationID) & ~od.PULocationID.isin([264, 265]) & ~od.DOLocationID.isin([264, 265])]
    G = nx.DiGraph()
    for r in g.itertuples(): G.add_edge(r.PULocationID, r.DOLocationID, weight=r.trips, dist=1 / r.trips)
    outs = pd.Series(dict(G.out_degree(weight="weight"))); ins = pd.Series(dict(G.in_degree(weight="weight")))
    T = pd.DataFrame({"trip_keluar": outs, "trip_masuk": ins}).fillna(0)
    T["selisih"] = T.trip_masuk - T.trip_keluar
    T["pagerank"] = pd.Series(nx.pagerank(G, weight="weight"))
    T["betweenness"] = pd.Series(nx.betweenness_centrality(G, weight="dist"))
    T["zona"] = T.index.map(nm); T["borough"] = T.index.map(bo)
    U = nx.Graph()
    for u, v, d in G.edges(data=True):
        if not U.has_edge(u, v): U.add_edge(u, v, weight=d["weight"] + (G[v][u]["weight"] if G.has_edge(v, u) else 0))
    com = sorted(nx.community.louvain_communities(U, weight="weight", seed=1), key=lambda s: -sum(outs.get(n, 0) for n in s))
    cid = {n: i for i, c in enumerate(com) for n in c}; T["kelompok"] = T.index.map(cid)
    inside = g[g.PULocationID.map(cid) == g.DOLocationID.map(cid)].trips.sum()
    e = g.sort_values("trips", ascending=False); cs = e.trips.cumsum() / e.trips.sum()
    info = dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), used_pct=g.trips.sum() / tot * 100,
                modularity=nx.community.modularity(U, com, weight="weight"), within_pct=inside / g.trips.sum() * 100,
                edges50=int((cs < .5).sum() + 1), n_comm=len(com))
    E = sorted(U.edges(data=True), key=lambda x: -x[2]["weight"])[:top_edges]
    H = nx.Graph(); H.add_weighted_edges_from([(u, v, d["weight"]) for u, v, d in E])
    pos = nx.spring_layout(H, weight="weight", k=0.9, seed=4, iterations=200)
    top_pairs = g.sort_values("trips", ascending=False).head(10).assign(
        asal=lambda d: d.PULocationID.map(nm), tujuan=lambda d: d.DOLocationID.map(nm))[["asal", "tujuan", "trips"]]
    return T, info, dict(H=H, pos=pos), top_pairs


# ------------------------------------------------------------------ keputusan bisnis (ambang / threshold)
def zone_thresholds(hist, hi_q=.8, lo_q=.2, min_hi=10):
    """Ambang per zona dari distribusi permintaan zona itu sendiri (data latih).
    hi: permintaan tinggi (kuantil hi_q, minimal min_hi trip/jam). lo: surplus (kuantil lo_q), hanya untuk zona yang cukup besar."""
    q = hist.groupby("z").y.quantile([lo_q, hi_q]).unstack(); q.columns = ["lo", "hi_raw"]
    q["big"] = q.hi_raw >= min_hi
    q["hi"] = q.hi_raw.clip(lower=min_hi)
    q.loc[~q.big, "lo"] = np.nan
    return q[["lo", "hi", "big"]]


def alert_metrics(detail, hist, hi_q=.8, min_hi=10):
    """Seberapa baik tiap metode menandai jam 'permintaan tinggi' yang benar-benar terjadi."""
    d = detail.join(zone_thresholds(hist, hi_q, .2, min_hi), on="z"); act = d.y >= d.hi; rows = {}
    for name, col in [("Model GBT", "pred"), ("Dasar: jam sama minggu lalu", "lag168"), ("Dasar: rata-rata 24 jam", "roll24")]:
        al = d[col] >= d.hi; tp = int((al & act).sum()); fp = int((al & ~act).sum()); fn = int((~al & act).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        rows[name] = {"Presisi": p, "Recall": r, "F1": 2 * p * r / max(p + r, 1e-9), "Peringatan": tp + fp, "Peringatan salah": fp, "Terlewat": fn}
    return pd.DataFrame(rows).T, int(act.sum())


def alert_sweep(detail, hist, min_hi=10, qs=(.6, .7, .8, .9)):
    out = []
    for q in qs:
        m, n = alert_metrics(detail, hist, q, min_hi); r = m.loc["Model GBT"]
        out.append({"Kuantil ambang": q, "Presisi": r.Presisi, "Recall": r.Recall, "Jam tinggi (aktual)": n})
    return pd.DataFrame(out)


def _ranges(hours):
    hours = sorted(hours); out = []; i = 0
    while i < len(hours):
        j = i
        while j + 1 < len(hours) and hours[j + 1] == hours[j] + 1: j += 1
        out.append(f"{hours[i]:02d}.00-{(hours[j] + 1) % 24:02d}.00"); i = j + 1
    return ", ".join(out)


def _daytype(dow): return np.where(dow >= 5, "Akhir pekan", "Hari kerja")


def plan_windows(detail, hist, names, hi_q=.8, lo_q=.2, min_hi=10, top_n=12):
    """Jendela jam siaga (permintaan tinggi) dan surplus per zona, untuk hari kerja dan akhir pekan."""
    th = zone_thresholds(hist, hi_q, lo_q, min_hi)
    m = (detail.assign(daytype=_daytype(detail.dow)).groupby(["z", "daytype", "hod"], as_index=False).pred.mean().join(th, on="z"))
    rows = []
    for (z, dt), g in m.groupby(["z", "daytype"]):
        hi = g[g.pred >= g.hi]; lo = g[g.pred <= g.lo]
        if hi.empty: continue
        rows.append({"Zona": names.get(z, z), "Hari": dt, "Jam siaga": _ranges(hi.hod), "Prediksi di jam siaga (trip/jam)": hi.pred.mean(),
                     "Ambang (trip/jam)": g.hi.iloc[0], "Jam surplus": _ranges(lo.hod) if len(lo) else "-", "_tot": g.pred.sum()})
    return pd.DataFrame(rows).sort_values("_tot", ascending=False).drop(columns="_tot").head(top_n).reset_index(drop=True)


def hour_view(detail, hist, names, daytype, hod, hi_q=.8, lo_q=.2, min_hi=10, n=8):
    th = zone_thresholds(hist, hi_q, lo_q, min_hi)
    d = detail.assign(daytype=_daytype(detail.dow)); d = d[(d.daytype == daytype) & (d.hod == hod)]
    m = d.groupby("z", as_index=False).pred.mean().join(th, on="z"); m["Zona"] = m.z.map(names)
    short = m[m.pred >= m.hi].assign(Rasio=lambda x: x.pred / x.hi).sort_values("Rasio", ascending=False).head(n)
    extra = m[m.pred <= m.lo].assign(Rasio=lambda x: x.pred / x.lo.clip(lower=1e-9)).sort_values("Rasio").head(n)
    f = lambda x, c: x[["Zona", "pred", c]].rename(columns={"pred": "Prediksi (trip/jam)", c: "Ambang (trip/jam)"})
    return f(short, "hi"), f(extra, "lo")


def fleet_sim(detail, fleet, rate, avg_fare):
    """Simulasi armada tetap vs kebutuhan menurut prediksi. rate = trip yang bisa dilayani satu taksi per jam (asumsi)."""
    tot = detail.groupby("t")[["y", "pred"]].sum().reset_index()
    tot["hod"] = tot.t.dt.hour; tot["daytype"] = _daytype(tot.t.dt.dayofweek)
    h = tot.groupby(["daytype", "hod"], as_index=False).pred.mean()
    h["need"] = np.ceil(h.pred / rate); h["unmet"] = (h.pred - fleet * rate).clip(lower=0); h["idle"] = (fleet - h.need).clip(lower=0)
    summ = h.groupby("daytype").agg(peak_need=("need", "max"), low_need=("need", "min"), unmet=("unmet", "sum"), idle=("idle", "sum"))
    summ["rugi_usd"] = summ.unmet * avg_fare
    return h, summ


def _d(x, k=1): return f"{x:.{k}f}".replace(".", ",")


def _n(x): return f"{x:,.0f}".replace(",", ".")


def insights(gold, h, months_txt, met, alerts, hi_q, plan, summ, fleet, rate, avg_fare):
    """Narasi keputusan 'berdasarkan data yang diupload'. Semua angka dihitung dari data."""
    ww = weekday_weekend(h); wd = ww["Hari kerja"]; pk, lo = int(wd.idxmax()), int(wd.idxmin())
    z0 = top_zones(gold, "Semua jam", None, 3); zs = ", ".join(z0.Zone)
    out = []
    out.append(("Data yang dipakai", f"Berdasarkan data yang diupload ({months_txt}; {_n(gold.trips.sum())} trip bersih), permintaan paling tinggi pada hari kerja "
                f"terjadi pukul {pk:02d}.00 (rata-rata {_n(wd.max())} trip/jam) dan paling rendah pukul {lo:02d}.00 ({_n(wd.min())} trip/jam). "
                f"Zona dengan penjemputan terbanyak: {zs}."))
    if "Akhir pekan" in ww:
        r = ww["Akhir pekan"].iloc[lo] / max(wd.iloc[lo], 1)
        out.append(("Pola hari", f"Pada pukul {lo:02d}.00 permintaan akhir pekan {_d(r)} kali hari kerja. "
                    + ("Jadwal sopir malam akhir pekan perlu dibuat terpisah dari hari kerja." if r > 1.3 else "Selisih hari kerja dan akhir pekan pada jam itu kecil.")))
    g, b1, b2 = met.loc["GBT: kalender + riwayat"], met.loc["Dasar: jam sama minggu lalu"], alerts.loc["Model GBT"]
    gain = (1 - g.MAE / b1.MAE) * 100
    out.append(("Seberapa bisa dipercaya", f"Model meleset rata-rata {_d(g.MAE)} trip per zona-jam, {gain:.0f}% lebih kecil daripada tebakan 'jam sama minggu lalu'. "
                f"Dengan ambang permintaan tinggi di kuantil {int(hi_q*100)}% tiap zona, model menandai {b2.Presisi*100:.0f}% peringatan dengan benar (presisi) "
                f"dan menangkap {b2.Recall*100:.0f}% jam tinggi yang benar-benar terjadi (recall). "
                + ("Peringatannya cukup andal untuk dijadikan dasar alokasi armada." if b2.Presisi >= .7 and b2.Recall >= .7 else
                   "Ada jam tinggi yang terlewat atau peringatan yang salah, jadi pakai sebagai acuan awal dan tetap beri ruang cadangan.")))
    if "GBT: kalender + riwayat + cuaca" in met.index:
        c = met.loc["GBT: kalender + riwayat + cuaca"]
        out.append(("Cuaca", "Menambah cuaca " + ("memperbaiki" if c.MAE < g.MAE else "tidak memperbaiki") + f" prediksi pada data ini (MAE {_d(c.MAE)} dibanding {_d(g.MAE)}). "
                    + ("Cuaca layak dipakai dalam keputusan." if c.MAE < g.MAE else "Keputusan sebaiknya tidak bergantung pada cuaca dulu; uji lagi saat data lebih panjang.")))
    if len(plan):
        p = plan.head(3)
        out.append(("Rencana siaga per zona", "Siagakan armada lebih banyak pada: " + "; ".join(f"{r.Zona} ({r.Hari.lower()}, {r['Jam siaga']})" for _, r in p.iterrows())
                    + ". Jam di luar itu cukup armada normal."))
    if summ is not None and len(summ):
        wk = summ.loc["Hari kerja"] if "Hari kerja" in summ.index else summ.iloc[0]
        out.append(("Rencana armada", f"Untuk zona yang dianalisis, kebutuhan armada berkisar {_n(wk.low_need)} sampai {_n(wk.peak_need)} taksi per jam pada hari kerja "
                    f"(asumsi {rate:g} trip per taksi per jam). Dengan armada tetap {_n(fleet)} taksi, diperkirakan {_n(wk.unmet)} trip per hari berada di atas kapasitas "
                    f"(sekitar USD {_n(wk.rugi_usd)} tarif) dan {_n(wk.idle)} taksi-jam menganggur. Menggeser jadwal mengikuti kurva kebutuhan mengurangi keduanya."))
    out.append(("Batasan", "Rekomendasi memakai data historis dan asumsi kapasitas per taksi yang bisa kamu ubah. Data TLC mencatat trip yang berhasil terlayani, bukan penumpang yang gagal mendapat taksi, jadi permintaan sebenarnya di jam sibuk bisa lebih tinggi. Hasilnya perlu dicocokkan dengan biaya operasional, jumlah sopir, dan aturan armada yang sebenarnya sebelum dijalankan."))
    return out


def default_fleet(detail, rate):
    """Armada tetap awal: rata-rata kebutuhan per jam pada hari kerja (dibulatkan ke 50)."""
    tot = detail.groupby("t").pred.sum(); tot = tot[tot.index.dayofweek < 5]
    return int(max(50, round(np.ceil(tot / rate).mean() / 50) * 50))
