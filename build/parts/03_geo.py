# %% [markdown]
# ## 5. Enriching the data with geography (web data collection, Week 4 skills)
# Inside Airbnb only gives the district and the coordinates of each listing. In Hong Kong the walk to an **MTR station** and the distance to the
# main business and tourist hubs are what guests pay for, so we add this information ourselves: we download the location of every MTR station from
# OpenStreetMap through its public **Overpass API** (`requests` + JSON, as in Week 4), compute for every listing the distance to the nearest station and to
# four hubs, and count how many other listings are nearby (local competition). The stations are saved in `data/mtr_stations.csv`, so the notebook runs offline.

# %%
def get_mtr_stations(cache=DATA / "mtr_stations.csv"):
    """MTR stations from the saved file, or from the OpenStreetMap API if the file does not exist yet."""
    if cache.exists():
        return pd.read_csv(cache)
    query = '[out:json][timeout:60];nwr["railway"="station"]["station"~"subway|train"](22.15,113.80,22.60,114.45);out center;'   # stations in a box around Hong Kong
    resp = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, headers={"User-Agent": "IS3240-student-project/1.0"}, timeout=120)
    resp.raise_for_status()
    rows = [dict(name=e["tags"].get("name:en") or e["tags"].get("name"), network=e["tags"].get("network"),
                 lat=e.get("lat") or e["center"]["lat"], lon=e.get("lon") or e["center"]["lon"]) for e in resp.json()["elements"]]
    st = pd.DataFrame(rows).dropna(subset=["name"])
    st = st[~st.network.fillna("").str.contains("Shenzhen|深圳")]                        # the box also contains Shenzhen stations: remove them
    st = st[~(st.network.isna() & (st.lat > 22.51)) & ~((st.lat > 22.45) & (st.lon < 114.0))]
    st = st.groupby("name", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
    st.to_csv(cache, index=False)
    return st

MTR = get_mtr_stations()
HUBS = {n: MTR.loc[MTR.name == n, ["lat", "lon"]].iloc[0].values for n in ["Central", "Tsim Sha Tsui", "Mong Kok", "Causeway Bay"]}   # the four main hubs

def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between points given in degrees (works with NumPy arrays, so it can handle whole columns at once)."""
    p = np.pi / 180                                                # degrees -> radians
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))                        # 6371 km = radius of the earth

def distance_features(lat, lon):
    """Distance features for arrays of coordinates: nearest MTR station, stations within 1 km, and distances to the hubs."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    d = haversine_km(lat[:, None], lon[:, None], MTR.lat.values[None, :], MTR.lon.values[None, :])    # d[i, j] = distance from listing i to station j
    out = pd.DataFrame({"dist_mtr_km": d.min(axis=1), "n_mtr_1km": (d <= 1).sum(axis=1)})
    for name, col in [("Central", "dist_central_km"), ("Tsim Sha Tsui", "dist_tst_km")]:
        out[col] = haversine_km(lat, lon, *HUBS[name])
    out["dist_hub_min_km"] = np.minimum.reduce([haversine_km(lat, lon, *h) for h in HUBS.values()])   # nearest of the 4 hubs
    return out

GEO = distance_features(L.latitude, L.longitude)
GEO.index = L.index
# Local competition: number of OTHER listings within 500 m and 1 km (only locations are used, never prices)
lat_all, lon_all = L.latitude.values, L.longitude.values
n500, n1km = [], []
for la, lo in zip(lat_all, lon_all):
    d = haversine_km(la, lo, lat_all, lon_all)                     # distance from this listing to all listings
    n500.append((d <= 0.5).sum() - 1)                              # "- 1": the listing itself is not a competitor
    n1km.append((d <= 1.0).sum() - 1)
GEO["listings_500m"], GEO["listings_1km"] = n500, n1km

# Relative price = price / median price of listings with the same room type and stay type (1.2 = 20% above similar listings)
df = df.join(GEO)
df["price_rel"] = df.price_hkd / df.groupby(["room_type", "regime"]).price_hkd.transform("median")
df["mtr_band"] = pd.cut(df.dist_mtr_km, [0, 0.25, 0.5, 1.0, 2.0, 50], labels=["< 250 m", "250-500 m", "0.5-1 km", "1-2 km", "2+ km"])
bb = df.groupby("mtr_band").agg(rel=("price_rel", "median"), n=("price_rel", "size"))
RESULTS["geo"] = dict(n_stations=int(len(MTR)), median_dist_mtr_m=float(df.dist_mtr_km.median() * 1000), share_within_500m=float((df.dist_mtr_km <= 0.5).mean()),
                      band_rel_price={str(k): float(v) for k, v in bb.rel.items()}, band_n={str(k): int(v) for k, v in bb.n.items()})
display(bb.round(3))

# %% [markdown]
# **Finding (partly negative).** There is **no simple "closer to the MTR = more expensive" gradient**: 40% of the listings are within 250 m of a station and half within 300 m,
# because supply itself is located where the transit is. Location effects are entangled with *what kind of property* is offered, which is why a model that controls for
# all factors at once, and not a chart of averages, is needed to separate them (sections 8 and 9).
#
# ### 5.1 Where is it expensive? A price map of Hong Kong
# We map the asking price of **whole flats**, for short stays and for monthly stays. The colour scale is centred on the Hong Kong median (blue = cheaper, red = more expensive;
# log scale). At every point of a fine grid we take the **median of the 18-22 nearest flats** (districts are too unequal and plain hexagon bins were too noisy) and we leave the map
# blank where no flat is nearby.

# %%
from matplotlib.colors import LinearSegmentedColormap
blue_red = LinearSegmentedColormap.from_list("bluered", [PAL["blue"], "#f0efec", PAL["red"]])

def price_map(regime, title, name):
    """Map of the price of whole flats for one stay type: all Hong Kong (left) and the urban core (right)."""
    flats = df[(df.room_type == "Entire home/apt") & (df.regime == regime)]
    ref = flats.price_hkd.median()
    score = np.log2(flats.price_hkd.values / ref)                  # 0 = median, +1 = double the median, -1 = half the median
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for ax, (xlim, ylim, step, k, max_km, sub) in zip(axes, [((113.84, 114.40), (22.17, 22.56), 0.006, 18, 2.2, "All Hong Kong"),
                                                            ((114.12, 114.25), (22.265, 22.345), 0.001, 22, 0.3, "Urban core")]):
        gx, gy = np.meshgrid(np.arange(*xlim, step), np.arange(*ylim, step * 0.92))                  # grid of points
        d = haversine_km(gy.ravel()[:, None], gx.ravel()[:, None], flats.latitude.values[None, :], flats.longitude.values[None, :])   # grid points x flats
        z = np.median(score[np.argsort(d, axis=1)[:, :k]], axis=1)                                    # median score of the k nearest flats
        z[d.min(axis=1) > max_km] = np.nan                                                            # no colour where no flat is nearby
        ax.scatter(df.longitude, df.latitude, s=1, color="#d9d8d1", zorder=1)                         # grey dots: all listings (outline of the land)
        im = ax.pcolormesh(gx, gy, np.ma.masked_invalid(z.reshape(gx.shape)), cmap=blue_red, vmin=-0.7, vmax=0.7, zorder=2, rasterized=True)
        ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[], aspect=1 / np.cos(np.radians(22.3)), title=f"{sub} (median of the {k} nearest flats)")
        ax.grid(False)
        if sub == "Urban core":
            for hub, (la, lo) in HUBS.items():
                ax.annotate(hub, (lo, la), fontsize=6.5, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    cb = fig.colorbar(im, ax=axes, fraction=0.035, ticks=[-0.7, 0, 0.7])
    cb.ax.set_yticklabels([f"{ref * 2 ** -0.7:,.0f}", f"{ref:,.0f} (median)", f"{ref * 2 ** 0.7:,.0f}"], fontsize=6.5)
    cb.set_label("Typical price per night of a whole flat (HKD)", fontsize=7)
    fig.suptitle(f"Where whole flats are expensive (red) or cheap (blue) – {title}", x=0.01, ha="left", fontsize=9, fontweight="bold")
    savefig(name, fig)
    return ref

ref_short = price_map("Short stay (<28 nights)", "short stays (< 28 nights)", "fig2b_price_map")
ref_long = price_map("Long stay (28+ nights)", "monthly stays (28+ nights)", "fig2c_price_map_monthly")

# Ranking of the districts: median price of a whole flat and number of flats behind each median
rank = df[df.room_type == "Entire home/apt"].groupby(["neighbourhood_cleansed", "regime"]).price_hkd.agg(["median", "size"]).unstack("regime")
rank.columns = [f"{a} | {'short' if 'Short' in b else 'monthly'}" for a, b in rank.columns]
rank = rank.sort_values("median | short", ascending=False)
big = rank[rank["size | short"] >= 30]                             # only districts with at least 30 short-stay flats are reliable
RESULTS["price_map"] = dict(ref_short=float(ref_short), ref_monthly=float(ref_long), most_expensive=big.index[0], cheapest=big.index[-1],
                            districts_ge30_short=[dict(district=i, median=float(r["median | short"]), n=int(r["size | short"])) for i, r in big.iterrows()])
display(rank.round(0))

# %% [markdown]
# **Finding.** The colour shows the *level* of the asking price of whole flats, not how good or profitable an area is: a red zone can simply contain larger flats, which is why
# the models in the next sections control for size, type and stay before attributing a premium to location.
