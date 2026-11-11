# %% [markdown]
# ## 10. Which factor families earn their place? (ablation study)
#
# A host or a manager does not want a black box; they want to know **where the predictive value comes from**, and a data team
# needs to know which data is worth collecting. We therefore rebuild the model **adding one family at a time** (same algorithm
# and tuned hyper-parameters) and watch how the typical pricing error falls.
#
# *Design decision:* differences between neighbouring steps are small (often 1-2 points), so a single 5-fold split is too noisy to
# rank them – fold-to-fold variation is larger than the effect. We therefore use **3 repetitions of host-grouped 5-fold
# cross-validation with different random host assignments**, pool the out-of-fold predictions of all 5,555 listings in each
# repetition, and report the mean and the spread across repetitions. (This is a diagnostic of *information value*; it is not used to
# select models, so using all listings does not contaminate the test result.) The two experience families (orange) are only
# available once a listing has guests.

# %%
XX = pd.concat([X_all, EXP_ALL], axis=1)          # launch-time + experience factors
gb_model = tuned["Gradient boosting"].named_steps["model"]
STEPS = list(FAMILIES) + list(EXP_FAMILIES)
SEEDS = [11, 22, 33]
p_all = df.price_hkd.values

def pooled_oof(cols, seed):
    cvr = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    o = cross_val_predict(make_pipe(cols, clone(gb_model)), XX, y, cv=cvr, groups=groups, n_jobs=-1)
    return metrics(p_all, np.exp(o))

def pooled_baseline(seed):
    cvr = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    o = np.zeros(len(df))
    for a_, b_ in cvr.split(X_all, y, groups=groups):
        o[b_] = CompsBaseline().fit(X_all.iloc[a_], p_all[a_]).predict(X_all.iloc[b_])
    return metrics(p_all, o)

t0 = time.time()
runs = {"Comparable-listings baseline": [pooled_baseline(sd) for sd in SEEDS]}
cols_acc = []
for fam in STEPS:
    cols_acc = cols_acc + (FAMILIES[fam] if fam in FAMILIES else EXP_FAMILIES[fam])
    runs[fam] = [pooled_oof(list(cols_acc), sd) for sd in SEEDS]
print(f"ablation took {time.time()-t0:.0f}s")

names = list(runs)
rows = []
for i, nm in enumerate(names):
    med = np.array([r["MedAPE"] for r in runs[nm]])
    prev = np.array([r["MedAPE"] for r in runs[names[i - 1]]]) if i > 0 else None
    rows.append(dict(step=nm if i < 2 else "+ " + nm, kind="baseline" if i == 0 else ("launch" if nm in FAMILIES else "experience"),
                     MedAPE=med.mean(), MedAPE_sd=med.std(), MAE=np.mean([r["MAE"] for r in runs[nm]]),
                     Within20=np.mean([r["Within20"] for r in runs[nm]]),
                     gain_pts=(prev - med).mean() * 100 if prev is not None else np.nan,
                     gain_sd=(prev - med).std() * 100 if prev is not None else np.nan))
abl = pd.DataFrame(rows)
display(abl.round(3))

fig, ax = plt.subplots(figsize=(7.2, 3.7))
yy = np.arange(len(abl))[::-1]
colors = [PAL["muted"] if k == "baseline" else (PAL["blue"] if k == "launch" else PAL["orange"]) for k in abl.kind]
ax.barh(yy, abl.MedAPE, xerr=abl.MedAPE_sd, color=colors, height=0.62, error_kw=dict(ecolor=PAL["ink2"], lw=0.8, capsize=2))
ax.set_yticks(yy); ax.set_yticklabels(abl.step, fontsize=7.4)
for yi, v, sd, g_, k in zip(yy, abl.MedAPE, abl.MedAPE_sd, abl.gain_pts, abl.kind):
    txt = f"{v*100:.1f}%" + ("" if k == "baseline" or pd.isna(g_) else f"   ({-g_:+.1f} pts)")
    ax.text(v + sd + 0.004, yi, txt, va="center", fontsize=6.9)
ax.set_xlim(0, abl.MedAPE.max() * 1.3); ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Typical (median) absolute % pricing error; mean of 3 repetitions of host-grouped CV (bars: spread)")
ax.grid(axis="y", visible=False)
launch = abl[abl.kind == "launch"]
big = launch.iloc[1:].gain_pts.idxmax()
ax.set_title(f"Launch-time factors bring the typical error to {launch.MedAPE.iloc[-1]*100:.0f}%; "
             f"'{launch.loc[big, 'step'].replace('+ ', '')}' is the biggest single step", fontsize=8.2, loc="left")
fig.tight_layout()
savefig("fig5_ablation", fig)
RESULTS["ablation"] = abl.to_dict(orient="records")

# %% [markdown]
# ## 11. Final benchmark: refit, interpret, and size the uncertainty
#
# ### 11.1 How much does each family matter? (grouped permutation importance on unseen hosts)
# For each family we scramble *all* its columns at once on the test set and measure how much the error grows. A family that
# is correlated with others can look small in the ablation (others already carry the signal) but still matter when removed;
# looking at both views protects us from over-interpreting either one.

# %%
best_pipe = tuned[BEST]
base_err = np.mean(np.abs(best_pipe.predict(Xte) - yte))
rng = np.random.default_rng(RANDOM_STATE)
imp_rows = []
for fam, cols in FAMILIES.items():
    deltas = []
    for _ in range(15):
        Xp = Xte.copy()
        perm = rng.permutation(len(Xp))
        Xp[cols] = Xte[cols].values[perm]
        deltas.append(np.mean(np.abs(best_pipe.predict(Xp) - yte)) - base_err)
    imp_rows.append(dict(family=fam, delta=np.mean(deltas) * 100, sd=np.std(deltas) * 100))
imp = pd.DataFrame(imp_rows).sort_values("delta", ascending=False)
display(imp.round(2))
RESULTS["importance"] = dict(base_mae_log=float(base_err), table=imp.to_dict(orient="records"))

# %% [markdown]
# ### 11.2 What is a feature worth? A hedonic (log-linear) regression
# Tree ensembles predict well but do not give a clean "price of a feature". For *explanation* we fit an interpretable log-price
# regression on all modelled listings with **standard errors clustered by host** (a host's listings are not independent). In a
# log-linear model, a coefficient β means that a one-unit change is associated with a `exp(β) − 1` percentage change in price.
# This is **association, not causation**: e.g. a host who installs a washer may also have a nicer flat.

# %%
mid_am = [a for a in AMEN if 0.15 <= amen_prev[a] <= 0.85][:10]
print("amenities used in the hedonic model:", mid_am)
H = X_all.copy()
H["ly"] = y
for c in ["bedrooms", "baths", "beds"]:
    H[c] = H[c].fillna(H[c].median())
H["short_weekend"] = H.q_weekend * (1 - H.long_stay)
H["mtr_500m"] = H.dist_mtr_km * 2          # per +500 m
H["central_5km"] = H.dist_central_km / 5   # per +5 km
H["density_100"] = H.listings_500m / 100   # per +100 listings within 500 m
H["host_id"] = df.host_id
am_terms = " + ".join(f"Q('{AMEN_COLS[a]}')" for a in mid_am)
formula = ("ly ~ long_stay + short_weekend + C(room_type, Treatment('Private room')) + accommodates + bedrooms + baths + bath_shared"
           " + C(district_grp, Treatment('Yau Tsim Mong')) + mtr_500m + central_5km + density_100 + pro_host + host_local + " + am_terms)
ols = smf.ols(formula, data=H).fit(cov_type="cluster", cov_kwds={"groups": H.host_id})
print(f"hedonic model: n = {int(ols.nobs):,}, R2 = {ols.rsquared:.2f} (log price), {H.host_id.nunique():,} host clusters")

def prem(term):
    b, se = ols.params[term], ols.bse[term]
    return dict(premium=np.exp(b) - 1, lo=np.exp(b - 1.96 * se) - 1, hi=np.exp(b + 1.96 * se) - 1, p=ols.pvalues[term])

key = {
    "Monthly-stay policy (28+ nights)": "long_stay",
    "Weekend check-in (short stays)": "short_weekend",
    "Entire home (vs. private room)": "C(room_type, Treatment('Private room'))[T.Entire home/apt]",
    "+1 guest accommodated": "accommodates",
    "+1 bedroom": "bedrooms",
    "+1 bathroom": "baths",
    "Shared bathroom": "bath_shared",
    "+500 m from nearest MTR": "mtr_500m",
    "+5 km from Central": "central_5km",
    "+100 other listings within 500 m": "density_100",
    "Professional host (5+ listings)": "pro_host",
    "Host lives in Hong Kong": "host_local",
}
for dname in ["Central & Western", "Wan Chai", "Islands", "Kowloon City"]:
    t = f"C(district_grp, Treatment('Yau Tsim Mong'))[T.{dname}]"
    if t in ols.params.index: key[f"District: {dname} (vs. Yau Tsim Mong)"] = t
for a in mid_am:
    key[f"Amenity: {a}"] = f"Q('{AMEN_COLS[a]}')"
hed = pd.DataFrame({k: prem(v) for k, v in key.items()}).T.sort_values("premium")
display((hed * [100, 100, 100, 1]).round(2))
RESULTS["hedonic"] = dict(n=int(ols.nobs), r2=float(ols.rsquared), terms={k: {a: float(b) for a, b in r.items()} for k, r in hed.iterrows()},
                          amenities_used=mid_am)

# figure 6: forest plot of premiums
show = hed[~hed.index.str.startswith("Amenity")]          # amenity coefficients are controls (they proxy listing quality) - not shown
show = show[(show.p < 0.10) | show.index.isin(["+500 m from nearest MTR", "+5 km from Central", "+100 other listings within 500 m"])].copy()
fig, ax = plt.subplots(figsize=(7.2, 0.21 * len(show) + 0.95))
yy = np.arange(len(show))
cols_f = [PAL["blue"] if v >= 0 else PAL["orange"] for v in show.premium]
ax.hlines(yy, show.lo * 100, show.hi * 100, color=PAL["muted"], lw=1.1, zorder=2)
ax.scatter(show.premium * 100, yy, color=cols_f, s=28, zorder=3, edgecolor="white", linewidth=0.8)
ax.axvline(0, color=PAL["ink2"], lw=0.9)
ax.set_yticks(yy); ax.set_yticklabels(show.index, fontsize=7.3)
for yi, (v, hi_) in enumerate(zip(show.premium, show.hi)):
    ax.text(max(show.hi.max() * 100, 10) + 4, yi, f"{v*100:+.0f}%", va="center", fontsize=7.2, color=PAL["ink"])
ax.set_xlabel("Associated price difference (%), other factors held constant; bars = 95% CI"); ax.grid(axis="y", visible=False)
ax.set_xlim(show.lo.min() * 100 - 5, show.hi.max() * 100 + 16)
ax.set_title("What the market pays for: price differences associated with each factor", fontsize=8.2)
fig.tight_layout()
savefig("fig6_hedonic", fig)
