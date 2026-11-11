# %% [markdown]
# ## 5. Enriching the data with geography (web data collection, Week 4 skills)
#
# *Business logic:* in Hong Kong, the walk to an **MTR station** and the distance to the main business and tourist hubs are
# what guests pay for – the city is compact, hilly and car-light. Inside Airbnb only gives the district name and the
# coordinates of the listing, so we **collect the location of every MTR station from OpenStreetMap** through the public
# Overpass API (`requests` + JSON, exactly the pattern used in Week 4) and compute distances ourselves.
#
# *Data-quality decisions:* the query returns Shenzhen metro stations near the border and Light Rail stops. We keep
# only **Hong Kong heavy-rail MTR stations** (subway/train) and drop Shenzhen stations. The result is cached to
# `data/mtr_stations.csv` so the notebook runs offline and is reproducible.

# %%
def get_mtr_stations(cache=DATA / "mtr_stations.csv"):
    if cache.exists():
        return pd.read_csv(cache)
    query = '[out:json][timeout:60];nwr["railway"="station"]["station"~"subway|train"](22.15,113.80,22.60,114.45);out center;'
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query},
                         headers={"User-Agent": "IS3240-student-project/1.0 (CityU)"}, timeout=120)
    resp.raise_for_status()
    rows = []
    for e in resp.json()["elements"]:
        t = e["tags"]
        rows.append(dict(name=t.get("name:en") or t.get("name"), network=t.get("network"),
                         lat=e.get("lat") or e["center"]["lat"], lon=e.get("lon") or e["center"]["lon"]))
    st = pd.DataFrame(rows).dropna(subset=["name"])
    st = st[~st.network.fillna("").str.contains("Shenzhen|深圳")]                  # other metro systems
    st = st[~(st.network.isna() & (st.lat > 22.51))]
    st = st[~((st.lat > 22.45) & (st.lon < 114.0))]                                    # Shenzhen stations near the border
    st = st.groupby("name", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
    st.to_csv(cache, index=False)
    return st

MTR = get_mtr_stations()
print(f"{len(MTR)} MTR stations loaded (lat {MTR.lat.min():.3f}-{MTR.lat.max():.3f}, lon {MTR.lon.min():.3f}-{MTR.lon.max():.3f})")

EARTH_KM = 6371.0
mtr_tree = BallTree(np.radians(MTR[["lat", "lon"]].values), metric="haversine")
HUBS = {n: MTR.loc[MTR.name == n, ["lat", "lon"]].iloc[0].values for n in ["Central", "Tsim Sha Tsui", "Mong Kok", "Causeway Bay"]}

def hav_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))

def geo_features(lat, lon):
    """Transit & centrality features for arrays of coordinates (straight-line distances, in km)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    pts = np.radians(np.c_[lat, lon])
    d1, _ = mtr_tree.query(pts, k=1)
    n1 = mtr_tree.query_radius(pts, r=1.0 / EARTH_KM, count_only=True)
    out = pd.DataFrame({"dist_mtr_km": d1[:, 0] * EARTH_KM, "n_mtr_1km": n1})
    out["dist_central_km"] = hav_km(lat, lon, *HUBS["Central"])
    out["dist_tst_km"] = hav_km(lat, lon, *HUBS["Tsim Sha Tsui"])
    out["dist_hub_min_km"] = np.min([hav_km(lat, lon, *v) for v in HUBS.values()], axis=0)
    return out

GEO = geo_features(L.latitude, L.longitude)
GEO.index = L.index

# local competition: how many other listings (any type) are within 500 m / 1 km. Uses locations only - no price information.
list_tree = BallTree(np.radians(L[["latitude", "longitude"]].values), metric="haversine")
pts_all = np.radians(L[["latitude", "longitude"]].values)
GEO["listings_500m"] = list_tree.query_radius(pts_all, r=0.5 / EARTH_KM, count_only=True) - 1
GEO["listings_1km"] = list_tree.query_radius(pts_all, r=1.0 / EARTH_KM, count_only=True) - 1
display(GEO.describe().T[["mean", "50%", "min", "max"]].round(2))

# %%
# Relative price: price divided by the median price of listings with the same room type AND stay regime. This removes the
# (huge) regime/room-type effect so that the map and distance chart show the *location* premium only.
df = df.join(GEO)
df["price_rel"] = df.price_hkd / df.groupby(["room_type", "regime"]).price_hkd.transform("median")

from matplotlib.colors import LinearSegmentedColormap
div = LinearSegmentedColormap.from_list("bluered", [PAL["blue"], "#f0efec", PAL["red"]])

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
a = ax[0]
core = df[(df.longitude.between(114.10, 114.27)) & (df.latitude.between(22.255, 22.36))]
hb = a.hexbin(core.longitude, core.latitude, C=np.log2(core.price_rel), reduce_C_function=np.median, gridsize=34, mincnt=6,
              cmap=div, vmin=-0.6, vmax=0.6, linewidths=0.2, edgecolors="white")
m_core = MTR[(MTR.lon.between(114.10, 114.27)) & (MTR.lat.between(22.255, 22.36))]
a.scatter(m_core.lon, m_core.lat, marker="^", s=11, color=PAL["ink"], label="MTR station", linewidths=0, zorder=3)
a.set_aspect(1 / np.cos(np.radians(22.3))); a.grid(False); a.set_xticks([]); a.set_yticks([])
a.set_title("Price vs. similar listings, urban core (hex medians)", fontsize=8.2)
cb = fig.colorbar(hb, ax=a, fraction=0.04, pad=0.02, ticks=[-0.6, 0, 0.6])
cb.ax.set_yticklabels(["-34%", "median", "+52%"], fontsize=6.5); cb.outline.set_visible(False)
a.legend(loc="lower right", fontsize=6.5)

b = ax[1]
dist_tbl = (df.groupby("neighbourhood_cleansed").agg(n=("price_rel", "size"), rel=("price_rel", "median"))
              .query("n >= 80").sort_values("rel"))
cols_b = [PAL["blue"] if v >= 1.0 else PAL["orange"] for v in dist_tbl.rel]
b.barh(dist_tbl.index, dist_tbl.rel - 1, color=cols_b, height=0.62)
for i, (v, n) in enumerate(zip(dist_tbl.rel - 1, dist_tbl.n)):
    b.text(v + (0.006 if v >= 0 else -0.006), i, f"{v*100:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    b.text(0.235, i, f"n={n:,}", va="center", ha="right", fontsize=6.5, color=PAL["muted"])
b.axvline(0, color=PAL["axis"], lw=0.9); b.set_xlim(-0.06, 0.24)
b.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0)); b.grid(axis="y", visible=False)
b.set_xlabel("Median price vs. same room type and stay regime")
b.set_title("Premiums appear outside the urban core", fontsize=8.2)
fig.tight_layout(w_pad=1.0)
savefig("fig2_geography", fig)

edges = [0, 0.25, 0.5, 1.0, 2.0, 50]
df["mtr_band"] = pd.cut(df.dist_mtr_km, edges, labels=["< 250 m", "250-500 m", "0.5-1 km", "1-2 km", "2+ km"])
bb = df.groupby("mtr_band").agg(rel=("price_rel", "median"), n=("price_rel", "size"))
RESULTS["geo"] = dict(n_stations=int(len(MTR)), median_dist_mtr_m=float(df.dist_mtr_km.median() * 1000),
                      share_within_500m=float((df.dist_mtr_km <= 0.5).mean()),
                      band_rel_price={str(k): float(v) for k, v in bb.rel.items()}, band_n={str(k): int(v) for k, v in bb.n.items()},
                      district_rel_price={k: float(v) for k, v in dist_tbl.rel.items()})
print(bb.round(3))
print(f"Median distance to nearest MTR station: {df.dist_mtr_km.median()*1000:.0f} m; {pct((df.dist_mtr_km<=0.5).mean())} of listings are within 500 m")

# %% [markdown]
# **Business reading (an honest, partly negative finding).** Raw medians show **no simple "closer to the MTR = more
# expensive" gradient**: 40% of listings are within 250 m of a station and half within 300 m, because supply itself is
# located where the transit is. Price premiums *relative to similar listings* appear mostly **outside the urban core**
# (Islands, Kowloon City, Sai Kung), where listings are scarcer and more often whole homes for holidays. This is a first warning
# that location effects are entangled with *what kind of property* is offered, which is exactly why a multivariate model –
# and not a chart of averages – is needed to separate them (sections 9 and 12).
