# %% [markdown]
# ## 12. The decision tool: a price benchmark with a plausible band
# A single number invites false precision, so the tool returns **a price and an 80% band**. The band comes from the model's own errors on the training hosts (10th-90th percentile
# of the error on the log price, separately for short and monthly stays), and we **check its coverage on the test hosts**: a band labelled "80%" must contain about 80% of unseen listings.

# %%
# Band: residual = real log price - predicted log price (predictions from models that did not see the host)
pred_train_oof = cross_val_predict(clone(models[BEST]), X_train, y_train, cv=GroupKFold(n_splits=5), groups=g_train, n_jobs=-1)
resid_train = y_train - pred_train_oof
stay_train, stay_test = X_all.long_stay.iloc[train_idx].values, X_all.long_stay.iloc[test_idx].values
BAND = {s: np.percentile(resid_train[stay_train == s], [10, 90]) for s in (0, 1)}      # 0 = short stay, 1 = monthly stay

pred_test_log = models[BEST].predict(X_test)
low, high = pred_test_log + np.array([BAND[s][0] for s in stay_test]), pred_test_log + np.array([BAND[s][1] for s in stay_test])
inside = (y_test >= low) & (y_test <= high)                          # is the real price inside the band?
coverage_by = {"short": float(inside[stay_test == 0].mean()), "long": float(inside[stay_test == 1].mean())}
print(f"The 80% band covers {pct(inside.mean())} of the unseen listings (short stay {pct(coverage_by['short'])}, long stay {pct(coverage_by['long'])})")
RESULTS["band"] = dict(coverage=float(inside.mean()), coverage_by=coverage_by, q_short=[float(v) for v in BAND[0]], q_long=[float(v) for v in BAND[1]])

# The model used by the tool is trained again on ALL listings; its band comes from the all-listing predictions of section 10
FINAL = clone(models[BEST]).fit(X, y)
BAND_ALL = {s: np.percentile((y - oof)[X_all.long_stay.values == s], [10, 90]) for s in (0, 1)}

# %% [markdown]
# **From simple inputs to a prediction.** A host only knows a handful of facts (district, size, location...). `make_row` builds a one-row table from them; everything else takes the
# market-typical value (median of each column, most common category) and the distance features are computed from the coordinates. `encode` prepares the row exactly like the
# training data and `benchmark` returns the price and the band.

# %%
TYPICAL = {c: X_all[c].median() for c in NEW_COLS if c not in CAT_COLS}      # typical listing: median of the numbers ...
TYPICAL.update({c: X_all[c].mode()[0] for c in CAT_COLS})                    # ... and most common value of the categories
UNFLAGGED_AMEN = float(np.median(X_all.n_amenities - X_all[list(AMEN_COLS.values())].sum(axis=1)))   # typical number of amenities outside our 30 flags
BASIC_KIT = ["Wifi", "Air conditioning", "Hot water", "Kitchen", "TV"]       # default amenities of the example listings

def make_row(district, room_type, property_type, accommodates, bedrooms, baths, lat, lon, long_stay=0, weekend=0, amenities=BASIC_KIT, shared_bath=0):
    """One listing (a one-row table) from simple inputs; everything not given takes the market-typical value."""
    r = dict(TYPICAL)
    dist = distance_features([lat], [lon]).iloc[0]
    nights = 30 if long_stay else 2                                          # default minimum stay: 30 nights (monthly) or 2 nights
    r.update(room_type=room_type, property_type_grp=property_type if property_type in TOP_PTYPES else "Other", accommodates=accommodates, bedrooms=bedrooms,
             studio=int(bedrooms == 0), beds=max(1, round(accommodates / 2)), baths=baths, bath_shared=shared_bath, long_stay=long_stay,
             log_min_nights=np.log(nights), q_log_nights=np.log(nights), q_weekend=weekend if not long_stay else 0, pro_host=0, host_local=1,   # an individual host living in Hong Kong
             
             district_grp=district if district in KEEP_DISTRICTS else "Other districts", region=REGION.get(district, "New Territories & Islands"), lat=lat, lon=lon)
    r.update({c: float(dist[c]) for c in ["dist_mtr_km", "n_mtr_1km", "dist_central_km", "dist_tst_km", "dist_hub_min_km"]})
    d = haversine_km(lat, lon, lat_all, lon_all)                              # competition: listings already near this location
    r["listings_500m"], r["listings_1km"] = int((d <= 0.5).sum()), int((d <= 1.0).sum())
    r.update({col: int(a in amenities) for a, col in AMEN_COLS.items()})
    r["n_amenities"] = len(amenities) + UNFLAGGED_AMEN
    return pd.DataFrame([r])[NEW_COLS]

def encode(row_df):
    """Prepare listings like the training data: 0/1 columns for the categories, same columns, training median for missing values."""
    return pd.get_dummies(row_df, columns=CAT_COLS, dtype=int).reindex(columns=X.columns, fill_value=0).fillna(train_median)

def benchmark(row_df):
    """Benchmark price and 80% band (low, high) in HKD for a one-row table."""
    log_price, stay = float(FINAL.predict(encode(row_df))[0]), int(row_df.long_stay.iloc[0])
    return np.exp(log_price), np.exp(log_price + BAND_ALL[stay][0]), np.exp(log_price + BAND_ALL[stay][1])

def station(name):
    """A point about 150 m from an MTR station (to place the example listings)."""
    return MTR.loc[MTR.name == name, ["lat", "lon"]].iloc[0].values + np.array([0.0012, 0.0010])

# Three example listings, each priced for a weekday short stay, a weekend short stay and a monthly stay
PROFILES = {
    "Wan Chai flat\n(1 bed, 2 guests)": dict(district="Wan Chai", room_type="Entire home/apt", property_type="Entire rental unit", accommodates=2, bedrooms=1, baths=1,
                                           lat=station("Wan Chai")[0], lon=station("Wan Chai")[1]),
    "Mong Kok room\n(shared bath)": dict(district="Yau Tsim Mong", room_type="Private room", property_type="Private room in rental unit", accommodates=2, bedrooms=1, baths=1,
                                        shared_bath=1, lat=station("Mong Kok")[0], lon=station("Mong Kok")[1]),
    "Sai Kung house\n(3 bed, 6 guests)": dict(district="Sai Kung", room_type="Entire home/apt", property_type="Entire home", accommodates=6, bedrooms=3, baths=2,
                                             lat=22.3815, lon=114.2700, amenities=BASIC_KIT + ["Washer", "Free parking on premises"])}
STAYS = {"Short stay, weekday": dict(long_stay=0, weekend=0), "Short stay, weekend": dict(long_stay=0, weekend=1), "Monthly stay": dict(long_stay=1, weekend=0)}
rows = []
for name, listing in PROFILES.items():
    for stay_name, stay in STAYS.items():
        price, lo, hi = benchmark(make_row(**listing, **stay))                # "**" passes the dictionary entries as named arguments
        rows.append(dict(profile=name.replace("\n", " "), profile_short=name, stay=stay_name, price=price, lo=lo, hi=hi))
scen = pd.DataFrame(rows)
display(scen[["profile", "stay", "price", "lo", "hi"]].round(0))
RESULTS["scenarios"] = scen.drop(columns="profile_short").to_dict(orient="records")

# %% [markdown]
# **What-if levers.** We change one thing of the Wan Chai flat and see how its benchmark moves. We do *not* rank amenities by "model-implied uplift": when a flag such as *hair dryer*
# is switched on for budget rooms, the model treats the room as a different (hotel-like) listing and returns implausible jumps; amenities are markers of listing type rather than priced
# upgrades. Because single levers depend on the model, we cross-check the two that matter most (*weekend check-in* and *one more guest*) with the average effect over all short-stay
# listings under each model.

# %%
base_listing = {**PROFILES[list(PROFILES)[0]], "long_stay": 0, "weekend": 0}
def bench(**changes):
    """Benchmark price of the Wan Chai flat with some inputs changed, e.g. bench(weekend=1)."""
    return benchmark(make_row(**{**base_listing, **changes}))[0]

base_price = bench()
LEVERS = {"Accept monthly stays (28+ nights)": dict(long_stay=1), "Weekend check-in (vs. weekday)": dict(weekend=1), "+1 guest accommodated (sofa bed)": dict(accommodates=3),
          "+1 bedroom (sleeps 4)": dict(bedrooms=2, accommodates=4), "Add dedicated workspace": dict(amenities=BASIC_KIT + ["Dedicated workspace"])}
lev = pd.DataFrame([dict(lever=k, price=bench(**v), change=bench(**v) / base_price - 1) for k, v in LEVERS.items()]).sort_values("change")
display(lev.round(3))
RESULTS["levers"] = dict(base_price=float(base_price), items=lev.to_dict(orient="records"))

# Cross-check: average effect over all short-stay listings (everybody checks in on a weekend instead of a weekday / gets one more guest)
short = X_all.long_stay == 0
weekday, weekend, more_guests = X[short].copy(), X[short].copy(), X[short].copy()
weekday["q_weekend"], weekend["q_weekend"], more_guests["q_weekend"] = 0, 1, 0
more_guests["accommodates"] = weekday.accommodates + 1
effect = lambda model, after: float(np.exp((model.predict(after) - model.predict(weekday)).mean()) - 1)      # average % change of the predicted price
check = {name: dict(weekend=effect(model, weekend), plus_one_guest=effect(model, more_guests)) for name, model in models.items()}
display((pd.DataFrame(check).T * 100).round(1).rename(columns={"weekend": "Weekend check-in (%)", "plus_one_guest": "+1 guest (%)"}))
RESULTS["lever_check"] = check

# Figure 8: left = benchmark with price band for the three examples, right = what-if levers
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
a = ax[0]
names = list(PROFILES)
xpos, width = np.arange(len(names)), 0.26
for j, (stay_name, col) in enumerate([("Short stay, weekday", PAL["blue"]), ("Short stay, weekend", PAL["aqua"]), ("Monthly stay", PAL["orange"])]):
    sub = scen[scen.stay == stay_name].set_index("profile_short").loc[names]
    a.bar(xpos + (j - 1) * width, sub.price, width * 0.9, color=col, label=stay_name, yerr=[sub.price - sub.lo, sub.hi - sub.price],
          error_kw=dict(ecolor=PAL["ink2"], lw=0.8, capsize=1.8))                 # the whiskers are the 80% band
    for x_, v in zip(xpos + (j - 1) * width, sub.price):
        a.text(x_, 40, f"{v:,.0f}", ha="center", va="bottom", fontsize=6.6, color="white", fontweight="bold", rotation=90)
a.set_xticks(xpos); a.set_xticklabels(names, fontsize=7); a.set_ylabel("Benchmark price per night (HKD), with 80% band")
a.legend(loc="upper left", fontsize=6.8); a.grid(axis="x", visible=False); a.set_title("Benchmark for three example listings", fontsize=8.2)
b = ax[1]
ypos = np.arange(len(lev))
b.barh(ypos, lev.change * 100, color=[PAL["blue"] if v >= 0 else PAL["orange"] for v in lev.change], height=0.62)
b.set_yticks(ypos); b.set_yticklabels(lev.lever, fontsize=7)
for y_, v in zip(ypos, lev.change):
    b.text(v * 100 + (1.2 if v >= 0 else -1.2), y_, f"{v*100:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=7)
b.axvline(0, color=PAL["ink2"], lw=0.9); b.set_xlim(lev.change.min() * 100 * 1.35, max(lev.change.max() * 100 * 1.5, 12))
b.set_xlabel(f"Change vs. {base_price:,.0f} HKD (Wan Chai flat, weekday)"); b.grid(axis="y", visible=False); b.set_title("What moves the price (same flat)", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig8_decision_tool", fig)
save_results()
