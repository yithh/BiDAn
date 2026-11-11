"""Langkah 4: graph analytics perpindahan antar-zona. Simpul = zona taksi, sisi = pasangan zona jemput -> tujuan, bobot = jumlah trip.
Menghitung PageRank, betweenness, selisih trip masuk/keluar, dan kelompok zona (Louvain), lalu menyimpan tabel dan gambar ke results/."""
import argparse, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pipeline as P, analysis as A, common as C

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--months", nargs="*"); ap.add_argument("--top-edges", type=int, default=150, help="jumlah sisi terkuat yang digambar")
a = ap.parse_args(); keys = C.pick_months(a.months)
od = P.load_od(keys); lookup = P.ensure_lookup()
T, info, gr, pairs = A.graph_analysis(od, lookup, a.top_edges)
C.RESULTS.mkdir(exist_ok=True)
T.sort_values("pagerank", ascending=False).to_csv(C.RESULTS / "graph_nodes.csv", index_label="LocationID")
pairs.to_csv(C.RESULTS / "graph_top_pairs.csv", index=False)
C.save("graph_summary.json", {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in dict(info, months=keys).items()})
print({k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v for k, v in info.items()})
print("\nPageRank tertinggi:\n", T.sort_values("pagerank", ascending=False).head(5)[["zona", "pagerank", "trip_masuk", "trip_keluar"]].round(4).to_string())
print("\nBetweenness tertinggi:\n", T.sort_values("betweenness", ascending=False).head(3)[["zona", "betweenness"]].round(3).to_string())
print("\nPelepas terbesar (masuk - keluar):\n", T.sort_values("selisih").head(3)[["zona", "selisih"]].to_string())
print("\nPasangan teratas:\n", pairs.to_string(index=False))

# gambar 1: jaringan sisi terkuat
H, pos = gr["H"], gr["pos"]; col = ["#1F4E79", "#E0701F", "#2A9D8F", "#8D99AE", "#E9C46A", "#9C6ADE"]
fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=200); ax.set_axis_off()
ws = np.array([d["weight"] for *_, d in H.edges(data=True)])
for (u, v, d) in H.edges(data=True): ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#9AA5B1", alpha=.55, lw=.4 + 4 * d["weight"] / ws.max(), zorder=1)
nodes = list(H.nodes)
ax.scatter([pos[n][0] for n in nodes], [pos[n][1] for n in nodes], s=[60 + 9000 * T.loc[n, "pagerank"] * 1.6 for n in nodes],
           c=[col[int(T.loc[n, "kelompok"]) % len(col)] for n in nodes], edgecolors="white", linewidths=.8, zorder=2)
for n in T.loc[nodes].sort_values("pagerank", ascending=False).head(7).index:
    ax.text(pos[n][0], pos[n][1], T.loc[n, "zona"], fontsize=6.5, ha="center", zorder=3, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.75))
ax.set_title("Jaringan antar-zona: ukuran = PageRank, warna = kelompok", fontsize=9); plt.tight_layout(); plt.savefig(C.RESULTS / "fig_graph_network.png"); plt.close()
# gambar 2: selisih trip masuk dan keluar
s = T.sort_values("selisih"); n = pd.concat([s.head(5), s.tail(5)])
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=200)
ax.barh(n.zona, n.selisih / 1000, color=["#E0701F" if v < 0 else "#1F4E79" for v in n.selisih]); ax.axvline(0, color="#5B6572", lw=.8)
ax.set_xlabel("Trip masuk dikurangi trip keluar (ribu)"); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
plt.savefig(C.RESULTS / "fig_graph_netflow.png"); plt.close()
print("\nTersimpan di results/: graph_nodes.csv, graph_top_pairs.csv, graph_summary.json, fig_graph_network.png, fig_graph_netflow.png")
