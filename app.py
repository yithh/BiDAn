"""Dasbor Taksi Kuning NYC (Streamlit). Jalankan: streamlit run app.py"""
import os; os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # hindari galat joblib di Windows
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
import analysis as A, pipeline as P

BLUE, ORANGE, YELLOW = "#1F4E79", "#E0701F", "#F2B705"
COMM = [BLUE, ORANGE, "#2A9D8F", "#8D99AE", "#E9C46A", "#9C6ADE"]
st.set_page_config(page_title="Dasbor Taksi Kuning NYC", layout="wide")
fmt = lambda n: f"{n:,.0f}".replace(",", ".")


def style(fig, h=380):
    """Judul grafik ditulis sebagai teks di atas grafik (bukan di dalam grafik) supaya tidak menempel dengan legenda."""
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else None
    if title: st.markdown(f"**{title}**")
    fig.update_layout(title=dict(text=""), height=h, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False); return fig


@st.cache_data(show_spinner="Membaca data...")
def get_gold(keys, ver): return P.load_gold(list(keys))
@st.cache_data(show_spinner="Membaca data...")
def get_od(keys, ver): return P.load_od(list(keys))
@st.cache_data
def get_weather(ver): return P.load_weather()
@st.cache_data(show_spinner="Melatih model (beberapa detik sampai menit)...")
def run_model(keys, ver, top_k, test_days):
    panel, wf = A.build_panel(get_gold(keys, ver), get_weather(ver), top_k)
    return A.train_eval(panel, wf, test_days)
@st.cache_data(show_spinner="Menganalisis graph...")
def run_graph(keys, ver, top_edges): return A.graph_analysis(get_od(keys, ver), P.ensure_lookup(), top_edges)


months = P.list_months(); ver = P.data_version()
st.title("Permintaan Taksi Kuning NYC")

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Filter")
    avail = months.key.tolist()
    sel = st.multiselect("Bulan data", avail, default=avail, help="Bulan baru ditambahkan di tab Kelola Data.") if avail else []
    st.caption(f"{len(avail)} bulan tersimpan di folder data/")

import os
def _public():
    """Mode publik: dinyalakan lewat variabel lingkungan DASHBOARD_PUBLIC=1 atau Secrets Streamlit (public_mode = true)."""
    if os.environ.get("DASHBOARD_PUBLIC", "").lower() in ("1", "true", "yes"): return True
    try: return bool(st.secrets.get("public_mode", False))
    except Exception: return False
PUBLIC = _public()

tab_names = ["Ringkasan", "Pola waktu", "Zona", "Prediksi", "Keputusan bisnis", "Graph", "Kelola data"]
t_sum, t_time, t_zone, t_pred, t_dec, t_graph, t_data = st.tabs(tab_names)

# ------------------------------------------------------------ kelola data
with t_data:
    if PUBLIC:
        st.info("Mode publik: menambah dan menghapus data dimatikan agar dasbor tetap utuh. Untuk menambah data TLC, jalankan dasbor di komputermu sendiri dengan streamlit run app.py.")
    if not PUBLIC:
        st.subheader("Tambah data TLC")
        st.write("Pilih salah satu cara. Setiap bulan dibersihkan dengan aturan R1–R7 lalu diringkas, "
                 "jadi file mentah (40–70 MB per bulan) tidak disimpan.")
        mode = st.radio("Sumber", ["Unduh dari TLC (tahun dan bulan)", "Unggah file parquet", "Path file di komputer"], horizontal=True)
        if mode.startswith("Unduh"):
            c1, c2, c3 = st.columns([1, 3, 1])
            year = c1.selectbox("Tahun", list(range(2026, 2016, -1)), index=1)
            mm = c2.multiselect("Bulan", list(range(1, 13)), default=[1], format_func=lambda m: f"{m:02d}")
            over = c3.checkbox("Timpa jika ada")
            if st.button("Tambah ke data", type="primary", disabled=not mm):
                for m in mm:
                    k = P.month_key(year, m)
                    if k in avail and not over: st.info(f"{k} sudah ada, dilewati."); continue
                    bar = st.progress(0.0, text=f"{k}: mengunduh...")
                    try:
                        meta = P.add_month_from_tlc(year, m, lambda x: bar.progress(x * .5, text=f"{k}: mengunduh..."),
                                                    lambda x: bar.progress(.5 + x * .5, text=f"{k}: membersihkan dan meringkas..."))
                        bar.progress(1.0, text=f"{k}: selesai, {fmt(meta['clean_rows'])} trip bersih")
                    except Exception as e:
                        bar.empty(); st.error(f"{k}: {e}")
                st.cache_data.clear()
                st.success("Selesai. Muat ulang halaman (tekan R) untuk melihat bulan baru di semua tab.")
        elif mode.startswith("Unggah"):
            up = st.file_uploader("File parquet yellow_tripdata_YYYY-MM.parquet", type="parquet")
            gy, gm = P.guess_year_month(up.name if up else "")
            c1, c2 = st.columns(2)
            y = c1.number_input("Tahun", 2017, 2026, gy or 2025); m = c2.number_input("Bulan", 1, 12, gm or 1)
            if up and st.button("Proses file", type="primary"):
                with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
                    f.write(up.getbuffer()); f.flush(); bar = st.progress(0.0, text="Memproses...")
                    try:
                        meta = P.process_parquet(f.name, y, m, lambda x: bar.progress(x), up.name)
                        st.cache_data.clear(); st.success(f"Selesai: {fmt(meta['clean_rows'])} trip bersih. Muat ulang halaman (tekan R).")
                    except Exception as e: st.error(str(e))
        else:
            path = st.text_input("Path file parquet", placeholder=r"C:\data\yellow_tripdata_2024-03.parquet")
            gy, gm = P.guess_year_month(path)
            c1, c2 = st.columns(2)
            y = c1.number_input("Tahun", 2017, 2026, gy or 2025, key="py"); m = c2.number_input("Bulan", 1, 12, gm or 1, key="pm")
            if path and st.button("Proses file", type="primary", key="pp"):
                if not Path(path).exists(): st.error("File tidak ditemukan.")
                else:
                    bar = st.progress(0.0, text="Memproses...")
                    try:
                        meta = P.process_parquet(path, y, m, lambda x: bar.progress(x))
                        st.cache_data.clear(); st.success(f"Selesai: {fmt(meta['clean_rows'])} trip bersih. Muat ulang halaman (tekan R).")
                    except Exception as e: st.error(str(e))
        st.divider(); st.subheader("Data cuaca (opsional)")
        st.write("Unggah CSV dari IEM ASOS (stasiun NYC). Waktu UTC otomatis diubah ke waktu New York. Dipakai di tab Prediksi.")
        wf_up = st.file_uploader("CSV cuaca IEM", type="csv", key="wx")
        if wf_up and st.button("Tambah cuaca"):
            try: st.success(f"Tersimpan: {fmt(P.add_weather(wf_up))} jam cuaca."); st.cache_data.clear()
            except Exception as e: st.error(f"Format tidak dikenali: {e}")
    st.divider(); st.subheader("Data tersimpan")
    if len(months):
        show = months[["key", "raw_rows", "clean_rows", "dibuang_pct", "source"]].rename(columns={
            "key": "Bulan", "raw_rows": "Baris mentah", "clean_rows": "Trip bersih", "dibuang_pct": "Dibuang (%)", "source": "Sumber"})
        st.dataframe(show.style.format({"Baris mentah": fmt, "Trip bersih": fmt, "Dibuang (%)": "{:.2f}"}), use_container_width=True, hide_index=True)
        if not PUBLIC:
            c1, c2 = st.columns([2, 1]); dk = c1.selectbox("Hapus bulan", avail)
            if c2.button("Hapus"): P.delete_month(dk); st.cache_data.clear(); st.rerun()
        with st.expander("Baris yang dibuang per aturan"):
            r = pd.DataFrame([{"Bulan": r.key, **r.rules} for r in months.itertuples()]).set_index("Bulan"); st.dataframe(r)
    else: st.warning("Belum ada data. Tambahkan bulan di atas.")

if not sel:
    for t in (t_sum, t_time, t_zone, t_pred, t_dec, t_graph): t.info("Pilih minimal satu bulan di sidebar, atau tambah data di tab Kelola data.")
    st.stop()
keys = tuple(sel); gold = get_gold(keys, ver); h = A.hourly(gold)

# ------------------------------------------------------------ ringkasan
with t_sum:
    sub = months[months.key.isin(sel)]
    pk = A.weekday_weekend(h)["Hari kerja"]
    man = gold[gold.Borough == "Manhattan"].trips.sum() / gold.trips.sum() * 100
    c = st.columns(4)
    c[0].metric("Trip bersih", fmt(gold.trips.sum()), f"{100 - sub.dibuang_pct.mean():.2f}% dari data mentah" if len(sub) else None, delta_color="off")
    c[1].metric("Jam puncak hari kerja", f"{pk.idxmax():02d}.00", f"{fmt(pk.max())} trip/jam", delta_color="off")
    c[2].metric("Penjemputan di Manhattan", f"{man:.1f}%")
    c[3].metric("Total tarif (USD)", fmt(gold.fare_sum.sum()))
    d = A.daily(h)
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=[ORANGE if x.dayofweek >= 5 else BLUE for x in d.index],
                           hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f} trip<extra></extra>"))
    fig.update_layout(title="Total trip harian (biru hari kerja, oranye akhir pekan)", yaxis_title="trip"); st.plotly_chart(style(fig), use_container_width=True)
    st.caption("Data: TLC Yellow Taxi, setelah pembersihan R1–R7. Waktu pickup dalam waktu lokal New York.")

# ------------------------------------------------------------ pola waktu
with t_time:
    ww = A.weekday_weekend(h)
    fig = go.Figure()
    for col, color in (("Hari kerja", BLUE), ("Akhir pekan", ORANGE)):
        if col in ww: fig.add_scatter(x=ww.index, y=ww[col], name=col, line=dict(color=color, width=3), hovertemplate="%{x:02d}.00: %{y:,.0f} trip/jam<extra></extra>")
    fig.update_layout(title="Rata-rata trip per jam", xaxis_title="jam pickup", yaxis_title="trip per jam", xaxis=dict(dtick=2)); st.plotly_chart(style(fig), use_container_width=True)
    hm = A.heat(h)
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=A.DAYS, colorscale=[[0, "#EAF0F6"], [1, BLUE]], hovertemplate="%{y} %{x:02d}.00<br>%{z:,.0f} trip/jam<extra></extra>"))
    fig.update_layout(title="Hari × jam (rata-rata trip per jam)", yaxis=dict(autorange="reversed")); st.plotly_chart(style(fig, 330), use_container_width=True)

# ------------------------------------------------------------ zona
with t_zone:
    c1, c2, c3 = st.columns(3)
    per = c1.selectbox("Waktu", list(A.PERIODS)); bor = c2.selectbox("Borough", ["Semua"] + sorted(gold.Borough.dropna().unique()))
    n = c3.slider("Jumlah zona", 5, 25, 10)
    z = A.top_zones(gold, per, bor, n).iloc[::-1]
    fig = go.Figure(go.Bar(y=z.Zone, x=z.trips, orientation="h", marker_color=BLUE, text=[fmt(v) for v in z.trips], textposition="outside",
                           hovertemplate="%{y}<br>%{x:,.0f} trip<extra></extra>"))
    fig.update_layout(title=f"Zona penjemputan teratas: {per}", xaxis_title="trip"); st.plotly_chart(style(fig, 120 + 30 * len(z)), use_container_width=True)

# ------------------------------------------------------------ prediksi
with t_pred:
    st.write("Memprediksi trip per zona-jam dengan Gradient Boosted Trees. Data uji adalah hari-hari paling akhir, dan model hanya dilatih dari hari sebelumnya.")
    c1, c2 = st.columns(2); k = c1.slider("Jumlah zona teratas", 20, 120, 60, 10); td = c2.slider("Hari uji (terakhir)", 3, 14, 7)
    days = (h.date.max() - h.date.min()).days + 1
    if days < 21: st.warning("Butuh minimal sekitar 3 minggu data. Tambah bulan di tab Kelola data.")
    elif st.button("Latih dan evaluasi", type="primary"):
        try:
            met, imp, ser, info = run_model(keys, ver, k, td)
            st.session_state["pred"] = (met, imp, ser, info)
        except Exception as e: st.error(str(e))
    if "pred" in st.session_state and days >= 21:
        met, imp, ser, info = st.session_state["pred"]
        st.caption(f"Latih: {fmt(info['n_train'])} baris. Uji: {fmt(info['n_test'])} baris mulai {info['test_from']:%d %b %Y}.")
        st.dataframe(met.style.format("{:.3f}").highlight_min(subset=["MAE", "RMSE"], color="#D7EAD9").highlight_max(subset=["R2"], color="#D7EAD9"), use_container_width=True)
        if "GBT: kalender + riwayat + cuaca" not in met.index: st.caption("Data cuaca tidak mencakup seluruh periode, jadi model dengan cuaca tidak dijalankan.")
        fig = go.Figure(); fig.add_scatter(x=ser.index, y=ser.y, name="Aktual", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=ser.index, y=ser.pred, name="Prediksi", line=dict(color=ORANGE, width=2, dash="dash"))
        fig.update_layout(title="Total trip per jam: aktual vs prediksi", yaxis_title="trip"); st.plotly_chart(style(fig), use_container_width=True)
        i = imp.head(8)[::-1]
        fig = go.Figure(go.Bar(y=i.index, x=i.values, orientation="h", marker_color="#2A9D8F")); fig.update_layout(title="Fitur paling berpengaruh (kenaikan MAE jika diacak)")
        st.plotly_chart(style(fig, 330), use_container_width=True)

# ------------------------------------------------------------ keputusan bisnis
with t_dec:
    st.write("Prediksi diubah menjadi keputusan lewat **ambang (threshold)**. Tiap zona dinilai terhadap permintaan normalnya sendiri: "
             "di atas ambang berarti siaga, di bawah ambang bawah berarti surplus armada.")
    if days < 21: st.warning("Butuh minimal sekitar 3 minggu data. Tambah bulan di tab Kelola data.")
    else:
        if "pred" not in st.session_state:
            if st.button("Latih model untuk analisis keputusan", type="primary"):
                try: st.session_state["pred"] = run_model(keys, ver, 60, 7); st.rerun()
                except Exception as e: st.error(str(e))
        else:
            met, imp, ser, info = st.session_state["pred"]; detail, hist = info["detail"], info["hist"]
            names = P.ensure_lookup().set_index("PULocationID").Zone.to_dict()
            st.caption(f"Model dilatih sebelum {info['test_from']:%d %b %Y} dan diuji pada hari sesudahnya. Latih ulang di tab Prediksi jika bulan data berubah.")
            c1, c2, c3 = st.columns(3)
            hi_q = c1.slider("Ambang siaga: kuantil permintaan zona", 0.60, 0.95, 0.80, 0.05, help="0,80 berarti siaga saat prediksi masuk 20% jam tersibuk zona itu.")
            lo_q = c2.slider("Ambang surplus: kuantil", 0.05, 0.40, 0.20, 0.05)
            min_hi = c3.number_input("Ambang minimum (trip/jam)", 1, 200, 10, help="Zona kecil tidak dianggap siaga kecuali prediksinya minimal sebesar ini.")

            st.subheader("1. Seberapa bisa dipercaya?")
            al, n_act = A.alert_metrics(detail, hist, hi_q, min_hi)
            st.write(f"Ada **{fmt(n_act)}** zona-jam permintaan tinggi di periode uji. Berikut seberapa baik tiap metode menandainya.")
            st.dataframe(al.style.format({"Presisi": "{:.0%}", "Recall": "{:.0%}", "F1": "{:.2f}", "Peringatan": fmt, "Peringatan salah": fmt, "Terlewat": fmt}), use_container_width=True)
            st.caption("Presisi: dari semua peringatan, berapa yang benar. Recall: dari semua jam tinggi yang terjadi, berapa yang tertangkap.")
            sw = A.alert_sweep(detail, hist, min_hi)
            fig = go.Figure(); fig.add_scatter(x=sw["Kuantil ambang"], y=sw.Presisi, name="Presisi", line=dict(color=BLUE, width=3))
            fig.add_scatter(x=sw["Kuantil ambang"], y=sw.Recall, name="Recall", line=dict(color=ORANGE, width=3))
            fig.update_layout(title="Ambang lebih ketat: presisi dan recall model", xaxis_title="kuantil ambang", yaxis=dict(tickformat=".0%", range=[0, 1])); st.plotly_chart(style(fig, 300), use_container_width=True)

            st.subheader("2. Rencana siaga per zona")
            plan = A.plan_windows(detail, hist, names, hi_q, lo_q, min_hi)
            st.dataframe(plan.style.format({"Prediksi di jam siaga (trip/jam)": fmt, "Ambang (trip/jam)": fmt}), use_container_width=True, hide_index=True)
            st.caption("Jam siaga: jam ketika rata-rata prediksi zona melewati ambang. Jam surplus: jam ketika di bawah ambang bawah.")
            c1, c2 = st.columns(2); dt = c1.selectbox("Jenis hari", ["Hari kerja", "Akhir pekan"]); hr = c2.slider("Jam", 0, 23, 18)
            short, extra = A.hour_view(detail, hist, names, dt, hr, hi_q, lo_q, min_hi)
            c1, c2 = st.columns(2)
            c1.markdown(f"**Kekurangan armada, {dt.lower()} pukul {hr:02d}.00**")
            if short.empty: c1.info("Tidak ada zona di atas ambang.")
            else: c1.dataframe(short.style.format(fmt, subset=["Prediksi (trip/jam)", "Ambang (trip/jam)"]), hide_index=True)
            c2.markdown("**Armada bisa dialihkan dari**")
            if extra.empty: c2.info("Tidak ada zona di bawah ambang bawah.")
            else: c2.dataframe(extra.style.format("{:.1f}", subset=["Prediksi (trip/jam)", "Ambang (trip/jam)"]), hide_index=True)

            st.subheader("3. Simulasi armada")
            st.caption("Hasil simulasi memakai asumsi di bawah. Ganti dengan angka operasional kamu sendiri.")
            c1, c2, c3 = st.columns(3)
            rate = c1.number_input("Trip per taksi per jam (asumsi)", 0.5, 6.0, 2.0, 0.1)
            avg_fare = gold.fare_sum.sum() / gold.trips.sum()
            fleet = c2.number_input("Armada tetap (taksi, untuk zona ini)", 10, 20000, A.default_fleet(detail, rate), 50)
            c3.metric("Tarif rata-rata per trip", f"USD {avg_fare:.2f}")
            hh, summ = A.fleet_sim(detail, fleet, rate, avg_fare)
            fig = go.Figure()
            for dtp, color in (("Hari kerja", BLUE), ("Akhir pekan", ORANGE)):
                x = hh[hh.daytype == dtp]; fig.add_scatter(x=x.hod, y=x.need, name=f"Kebutuhan {dtp.lower()}", line=dict(color=color, width=3))
            fig.add_hline(y=fleet, line_dash="dash", line_color="#5B6572", annotation_text="Armada tetap")
            fig.update_layout(title="Kebutuhan armada per jam menurut prediksi", xaxis_title="jam", yaxis_title="taksi", xaxis=dict(dtick=2)); st.plotly_chart(style(fig), use_container_width=True)
            st.dataframe(summ.rename(columns={"peak_need": "Kebutuhan puncak", "low_need": "Kebutuhan terendah", "unmet": "Trip di atas kapasitas/hari",
                                              "idle": "Taksi-jam menganggur/hari", "rugi_usd": "Tarif terkait (USD)"}).style.format(fmt), use_container_width=True)

            st.subheader("4. Kesimpulan dan rencana keputusan")
            months_txt = ", ".join(sel) if len(sel) <= 4 else f"{sel[0]} sampai {sel[-1]}, {len(sel)} bulan"
            for title, txt in A.insights(gold, h, months_txt, met, al, hi_q, plan, summ, fleet, rate, avg_fare):
                st.markdown(f"**{title}.** {txt}")

# ------------------------------------------------------------ graph
with t_graph:
    te = st.slider("Jumlah sisi terkuat yang digambar", 50, 400, 150, 25)
    try: T, info, gr, tp = run_graph(keys, ver, te)
    except Exception as e: st.error(str(e)); st.stop()
    c = st.columns(4)
    c[0].metric("Simpul / sisi", f"{info['nodes']} / {fmt(info['edges'])}"); c[1].metric("Sisi untuk 50% trip", fmt(info["edges50"]))
    c[2].metric("Trip dalam kelompok", f"{info['within_pct']:.1f}%"); c[3].metric("Modularitas", f"{info['modularity']:.2f}", "rendah = batas kabur" if info["modularity"] < .3 else None, delta_color="off")
    H, pos = gr["H"], gr["pos"]
    fig = go.Figure(); ws = np.array([d["weight"] for *_, d in H.edges(data=True)]); mx = ws.max()
    for (u, v, d) in H.edges(data=True):
        fig.add_scatter(x=[pos[u][0], pos[v][0], None], y=[pos[u][1], pos[v][1], None], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(width=0.5 + 5 * d["weight"] / mx, color="rgba(140,150,165,.45)"))
    for ci in sorted(T.loc[list(H.nodes)].kelompok.unique()):
        ns = [n for n in H.nodes if T.loc[n, "kelompok"] == ci]
        fig.add_scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers", name=f"Kelompok {ci + 1}",
                        marker=dict(color=COMM[ci % len(COMM)], size=[10 + 900 * T.loc[n, "pagerank"] for n in ns], line=dict(color="white", width=1)),
                        text=[f"{T.loc[n, 'zona']}<br>PageRank {T.loc[n, 'pagerank']:.4f}" for n in ns], hoverinfo="text")
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), title="Jaringan antar-zona (ukuran = PageRank)"); st.plotly_chart(style(fig, 520), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Zona paling penting (PageRank)**"); c1.dataframe(T.sort_values("pagerank", ascending=False).head(10)[["zona", "pagerank", "betweenness", "kelompok"]].style.format({"pagerank": "{:.4f}", "betweenness": "{:.3f}"}), hide_index=True)
    c2.markdown("**Selisih trip masuk dan keluar**")
    sd = T.sort_values("selisih"); c2.dataframe(pd.concat([sd.head(5), sd.tail(5)])[["zona", "trip_keluar", "trip_masuk", "selisih"]].style.format(fmt, subset=["trip_keluar", "trip_masuk", "selisih"]), hide_index=True)
    st.markdown("**Pasangan asal-tujuan terbanyak**"); st.dataframe(tp.style.format({"trips": fmt}), hide_index=True)
    st.caption("Trip dalam zona yang sama dan zona Unknown tidak dihitung. Kelompok dicari dengan algoritma Louvain.")
