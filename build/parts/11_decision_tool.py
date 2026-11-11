# %% [markdown]
# ## 14. The decision tool: a price benchmark with a plausible band
#
# A single number invites false precision. Hosts need **a recommended price and a band** inside which the market would not be
# surprised. We estimate the band from the model's own out-of-fold errors (the 10th–90th percentile of the log error, computed
# separately for short and long stays because errors differ), and **check its coverage on the held-out test hosts** – a band
# labelled "80%" must contain about 80% of unseen listings, otherwise it is misleading.

# %%
# band from OUT-OF-FOLD residuals on the TRAINING hosts only, then tested on the held-out test hosts
oof_tr = cross_val_predict(clone(tuned[BEST]), Xtr, ytr, cv=GroupKFold(n_splits=5), groups=gtr, n_jobs=-1)
res_tr = ytr - oof_tr
Q_BAND = {int(r): np.percentile(res_tr[Xtr.long_stay.values == r], [10, 90]) for r in (0, 1)}
pred_te_log = tuned[BEST].predict(Xte)
lo_te = pred_te_log + np.array([Q_BAND[int(r)][0] for r in Xte.long_stay.values])
hi_te = pred_te_log + np.array([Q_BAND[int(r)][1] for r in Xte.long_stay.values])
cov = float(np.mean((yte >= lo_te) & (yte <= hi_te)))
cov_by = {("long" if r else "short"): float(np.mean(((yte >= lo_te) & (yte <= hi_te))[Xte.long_stay.values == r])) for r in (0, 1)}
print(f"Nominal 80% band covers {pct(cov)} of unseen listings (short stay {pct(cov_by['short'])}, long stay {pct(cov_by['long'])}); "
      f"median band width: x{np.exp(np.median(hi_te - lo_te)):.2f} (e.g. 500 HKD -> {500*np.exp(np.median(Q_BAND[0][0])):.0f}-{500*np.exp(np.median(Q_BAND[0][1])):.0f} HKD for short stays)")
RESULTS["band"] = dict(coverage=cov, coverage_by=cov_by, q_short=[float(v) for v in Q_BAND[0]], q_long=[float(v) for v in Q_BAND[1]])

# final model for the tool: refit on ALL modelled listings; band from all-data out-of-fold residuals
FINAL = clone(tuned[BEST]).fit(X_all, y)
res_all = y - oof
Q_ALL = {int(r): np.percentile(res_all[X_all.long_stay.values == r], [10, 90]) for r in (0, 1)}

# %%
TYPICAL = {c: X_all[c].median() for c in NEW_COLS if c not in CAT_COLS}
for c in CAT_COLS: TYPICAL[c] = X_all[c].mode()[0]
UNFLAGGED_AMEN = float(np.median(X_all.n_amenities - X_all[list(AMEN_COLS.values())].sum(axis=1)))
BASIC_KIT = ["Wifi", "Air conditioning", "Hot water", "Kitchen", "TV"]

def make_row(district, room_type, property_type, accommodates, bedrooms, baths, lat, lon, long_stay=0, weekend=0,
             amenities=BASIC_KIT, shared_bath=0, min_nights=None, pro_host=0, host_local=1):
    """Builds ONE listing from plain business inputs. Everything not specified takes the market-typical value."""
    r = dict(TYPICAL)
    g = geo_features([lat], [lon]).iloc[0]
    r.update(room_type=room_type, property_type_grp=property_type if property_type in TOP_PTYPES else "Other", accommodates=accommodates,
             bedrooms=bedrooms, studio=int(bedrooms == 0), beds=max(1, round(accommodates / 2)), baths=baths, bath_shared=shared_bath,
             long_stay=long_stay, district_grp=district if district in KEEP_DISTRICTS else "Other districts",
             region=REGION.get(district, "New Territories & Islands"), lat=lat, lon=lon,
             q_weekend=weekend if not long_stay else 0, pro_host=pro_host, host_local=host_local)
    mn = min_nights if min_nights else (30 if long_stay else 2)
    r.update(log_min_nights=np.log(mn), q_log_nights=np.log(mn))
    for c in ["dist_mtr_km", "n_mtr_1km", "dist_central_km", "dist_tst_km", "dist_hub_min_km"]:
        r[c] = float(g[c])
    pt = np.radians([[lat, lon]])
    r["listings_500m"] = int(list_tree.query_radius(pt, r=0.5 / EARTH_KM, count_only=True)[0])
    r["listings_1km"] = int(list_tree.query_radius(pt, r=1.0 / EARTH_KM, count_only=True)[0])
    for a, c in AMEN_COLS.items():
        r[c] = int(a in amenities)
    r["n_amenities"] = len(amenities) + UNFLAGGED_AMEN
    return pd.DataFrame([r])[NEW_COLS]

def benchmark(row_df, model=None):
    """Returns (benchmark price, low, high) in HKD."""
    model = model or FINAL
    lp = float(model.predict(row_df)[0]); ls = int(row_df.long_stay.iloc[0])
    return np.exp(lp), np.exp(lp + Q_ALL[ls][0]), np.exp(lp + Q_ALL[ls][1])

def station(name):
    return MTR.loc[MTR.name == name, ["lat", "lon"]].iloc[0].values + np.array([0.0012, 0.0010])   # ~150 m from the station

PROFILES = {
    "Wan Chai flat\n(1 bed, 2 guests)": dict(district="Wan Chai", room_type="Entire home/apt", property_type="Entire rental unit", accommodates=2,
                                           bedrooms=1, baths=1, lat=station("Wan Chai")[0], lon=station("Wan Chai")[1]),
    "Mong Kok room\n(shared bath)": dict(district="Yau Tsim Mong", room_type="Private room", property_type="Private room in rental unit", accommodates=2,
                                        bedrooms=1, baths=1, shared_bath=1, lat=station("Mong Kok")[0], lon=station("Mong Kok")[1]),
    "Sai Kung house\n(3 bed, 6 guests)": dict(district="Sai Kung", room_type="Entire home/apt", property_type="Entire home", accommodates=6,
                                             bedrooms=3, baths=2, lat=22.3815, lon=114.2700,
                                             amenities=BASIC_KIT + ["Washer", "Free parking on premises"]),
}
scen_rows = []
for nm, kw in PROFILES.items():
    for lab, extra in [("Short stay, weekday", dict(long_stay=0, weekend=0)), ("Short stay, weekend", dict(long_stay=0, weekend=1)),
                       ("Monthly stay", dict(long_stay=1, weekend=0))]:
        p_, lo_, hi_ = benchmark(make_row(**kw, **extra))
        scen_rows.append(dict(profile=nm.replace("\n", " "), profile_short=nm, stay=lab, price=p_, lo=lo_, hi=hi_))
scen = pd.DataFrame(scen_rows)
display(scen[["profile", "stay", "price", "lo", "hi"]].round(0))
RESULTS["scenarios"] = scen.drop(columns="profile_short").to_dict(orient="records")

# %% [markdown]
# **Why not rank amenities by "model-implied uplift"?** We tried it and rejected it: when a flag such as *hair dryer* is switched on
# for budget rooms that lack it, the model treats the room as a different kind of (hotel-like, professionally managed) listing and
# returns implausibly large jumps. Amenities are *markers* of listing type rather than priced upgrades, and the ablation confirms they add
# only a little accuracy once type, stay and district are known. Instead we show levers a host really controls, on one realistic profile.

# %%
base_kw = {**PROFILES[list(PROFILES)[0]], "long_stay": 0, "weekend": 0}
def bench(**ov):
    return benchmark(make_row(**{**base_kw, **ov}))[0]
p0 = bench()
LEVERS = {
    "Accept monthly stays (28+ nights)": dict(long_stay=1),
    "Weekend check-in (vs. weekday)": dict(weekend=1),
    "+1 guest accommodated (sofa bed)": dict(accommodates=3),
    "+1 bedroom (sleeps 4)": dict(bedrooms=2, accommodates=4),
    "Add dedicated workspace": dict(amenities=BASIC_KIT + ["Dedicated workspace"]),
    "Add washer": dict(amenities=BASIC_KIT + ["Washer"]),
    "Add self check-in": dict(amenities=BASIC_KIT + ["Self check-in"]),
}
lev = pd.DataFrame([dict(lever=k, price=bench(**v), change=bench(**v) / p0 - 1) for k, v in LEVERS.items()]).sort_values("change")
display(lev.round(3))
print(f"base: Wan Chai flat, short weekday stay = {p0:,.0f} HKD")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
a = ax[0]
profs = list(PROFILES)
xs = np.arange(len(profs)); w = 0.26
for j, (stay, col) in enumerate([("Short stay, weekday", PAL["blue"]), ("Short stay, weekend", PAL["aqua"]), ("Monthly stay", PAL["orange"])]):
    sub = scen[scen.stay == stay].set_index("profile_short").loc[profs]
    a.bar(xs + (j - 1) * w, sub.price, w * 0.9, color=col, label=stay,
          yerr=[sub.price - sub.lo, sub.hi - sub.price], error_kw=dict(ecolor=PAL["ink2"], lw=0.8, capsize=1.8))
    for xi, v in zip(xs + (j - 1) * w, sub.price):
        a.text(xi, 40, f"{v:,.0f}", ha="center", va="bottom", fontsize=6.6, color="white", fontweight="bold", rotation=90)
a.set_xticks(xs); a.set_xticklabels(profs, fontsize=7); a.set_ylabel("Benchmark price per night (HKD), with 80% band")
a.legend(loc="upper left", fontsize=6.8); a.grid(axis="x", visible=False)
a.set_title("Benchmark for three example listings", fontsize=8.2)
b = ax[1]
yy = np.arange(len(lev))
b.barh(yy, lev.change * 100, color=[PAL["blue"] if v >= 0 else PAL["orange"] for v in lev.change], height=0.62)
b.set_yticks(yy); b.set_yticklabels(lev.lever, fontsize=7)
for yi, v in zip(yy, lev.change):
    b.text(v * 100 + (1.2 if v >= 0 else -1.2), yi, f"{v*100:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=7)
b.axvline(0, color=PAL["ink2"], lw=0.9); b.set_xlim(lev.change.min() * 100 * 1.35, max(lev.change.max() * 100 * 1.5, 12))
b.set_xlabel(f"Change vs. {p0:,.0f} HKD (Wan Chai flat, weekday)"); b.grid(axis="y", visible=False)
b.set_title("What moves the price (same flat)", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig8_decision_tool", fig)
RESULTS["levers"] = dict(base_price=float(p0), items=lev.to_dict(orient="records"))

# %% [markdown]
# **How much should we trust a single lever?** Individual effects depend on the model and the profile, so we cross-check the two levers
# that matter most commercially – *weekend check-in* and *one more guest* – with the average effect over **all short-stay listings**
# under each of the three tuned models (and compare with the hedonic regression of section 11.2). We treat the levers as direction and
# order of magnitude, not as exact returns.

# %%
sh_ = X_all.long_stay == 0
chk = {}
for nm in ML_NAMES:
    m_ = tuned[nm]
    A, B = X_all[sh_].copy(), X_all[sh_].copy(); B["q_weekend"] = 1; A["q_weekend"] = 0
    wk = float(np.exp((m_.predict(B) - m_.predict(A)).mean()) - 1)
    A, B = X_all[sh_].copy(), X_all[sh_].copy(); B["accommodates"] = A.accommodates + 1
    gu = float(np.exp((m_.predict(B) - m_.predict(A)).mean()) - 1)
    chk[nm] = dict(weekend=wk, plus_one_guest=gu)
chk["Hedonic regression"] = dict(weekend=float(RESULTS["hedonic"]["terms"]["Weekend check-in (short stays)"]["premium"]),
                                 plus_one_guest=float(RESULTS["hedonic"]["terms"]["+1 guest accommodated"]["premium"]))
display((pd.DataFrame(chk).T * 100).round(1).rename(columns={"weekend": "Weekend check-in (%)", "plus_one_guest": "+1 guest (%)"}))
RESULTS["lever_check"] = chk
save_results()
