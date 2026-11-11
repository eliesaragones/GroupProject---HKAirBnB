# %% [markdown]
# ## 12. From prediction to action: who is priced away from the benchmark, and does it cost them guests?
#
# A prediction is only useful if it changes a decision. We use the benchmark as a **price audit**:
# 1. For *every* listing we obtain an **out-of-fold prediction** – the price a model that has never seen that host would have
#    suggested (host-grouped 5-fold cross-validation, so a listing is never predicted by a model trained on its own host).
# 2. The **price gap** = actual price ÷ benchmark − 1. We call a listing *above benchmark* if it is priced ≥ 25% above, *below
#    benchmark* if ≥ 20% below (the same ±25% band on a ratio scale), and *in line* otherwise.
# 3. We then ask the question a host really cares about: **do listings priced above the benchmark attract fewer guests?**
#    Guests cannot be observed directly, so we use the number of reviews in the last 12 months (`number_of_reviews_ltm`)
#    as a demand proxy (reviews are a roughly constant fraction of stays).

# %%
oof = cross_val_predict(clone(tuned[BEST]), X_all, y, cv=GroupKFold(n_splits=5), groups=groups, n_jobs=-1)
p_all = df.price_hkd.values
m_oof = metrics(p_all, np.exp(oof))
# comparable-listings baseline, same folds
oof_base = np.zeros(len(df))
for a_, b_ in GroupKFold(n_splits=5).split(X_all, y, groups=groups):
    oof_base[b_] = CompsBaseline().fit(X_all.iloc[a_], p_all[a_]).predict(X_all.iloc[b_])
m_base = metrics(p_all, oof_base)
print(f"All {len(df):,} listings, host-grouped 5-fold CV ->  {BEST}: MAE {m_oof['MAE']:.0f} HKD, median % error {pct(m_oof['MedAPE'],1)}, within +-20%: {pct(m_oof['Within20'])}, R2(log) {m_oof['R2_log']:.2f}")
print(f"                                  Comparable-listings baseline: MAE {m_base['MAE']:.0f} HKD, median % error {pct(m_base['MedAPE'],1)}, within +-20%: {pct(m_base['Within20'])}")
RESULTS["oof"] = dict(model=m_oof, baseline=m_base, mae_reduction=float(1 - m_oof["MAE"] / m_base["MAE"]))

df["pred"] = np.exp(oof)
df["gap"] = df.price_hkd / df.pred - 1
TH_HI, TH_LO = 0.25, 1 / 1.25 - 1
df["position"] = np.select([df.gap >= TH_HI, df.gap <= TH_LO], ["Above benchmark", "Below benchmark"], "In line")
pos_share = df.position.value_counts(normalize=True)
display(pos_share.round(3))
print("Median gap by regime / room type (should be ~0 if the benchmark is unbiased):")
display(df.groupby(["regime", "room_type"]).gap.agg(["size", "median"]).round(3).query("size >= 20"))
df[["id", "price_hkd", "pred", "gap", "position", "segment", "regime", "neighbourhood_cleansed", "room_type"]].round(3).to_csv(OUT / "listing_price_audit.csv", index=False)

# --- demand test --------------------------------------------------------------------------------------------------------
elig = df[(df.first_review <= SCRAPE_DATE - pd.Timedelta(days=365)) & df.review_scores_rating.notna()].copy()
elig["log_gap"] = np.log(elig.price_hkd / elig.pred)
elig["host_id"] = elig.host_id.astype(int)
elig["acc_log"] = np.log(elig.accommodates)
print(f"Demand test sample: {len(elig):,} listings with at least a full year of history (first review > 12 months before the scrape)")
f0 = "number_of_reviews_ltm ~ log_gap + C(regime) + C(room_type) + C(neighbourhood_cleansed) + acc_log"
f1 = f0 + " + review_scores_rating"
glm = {}
for nm, f in [("controls", f0), ("controls + rating", f1)]:
    glm[nm] = smf.glm(f, data=elig, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": elig.host_id})
irr = {}
for nm, mdl in glm.items():
    b, se = mdl.params["log_gap"], mdl.bse["log_gap"]
    k = np.log(1.10)    # +10% price relative to benchmark
    irr[nm] = dict(irr_10=float(np.exp(b * k)), lo=float(np.exp((b - 1.96 * se) * k)), hi=float(np.exp((b + 1.96 * se) * k)), p=float(mdl.pvalues["log_gap"]))
    print(f"{nm:18s}: +10% above benchmark -> reviews x{irr[nm]['irr_10']:.3f}  (95% CI {irr[nm]['lo']:.3f}-{irr[nm]['hi']:.3f}), p = {irr[nm]['p']:.3g}")
# robustness: Inside Airbnb's own occupancy estimate, OLS on log(1+occupancy nights)
elig["lo"] = np.log1p(elig.estimated_occupancy_l365d)
ols_occ = smf.ols("lo ~ log_gap + C(regime) + C(room_type) + C(neighbourhood_cleansed) + acc_log + review_scores_rating", data=elig).fit(cov_type="cluster", cov_kwds={"groups": elig.host_id})
print(f"robustness (estimated occupancy nights, log): +10% above benchmark -> {100*(np.exp(ols_occ.params['log_gap']*np.log(1.1))-1):+.1f}%  (p = {ols_occ.pvalues['log_gap']:.3g})")
RESULTS["demand"] = dict(n=len(elig), irr=irr, occupancy_effect_10pct=float(np.exp(ols_occ.params["log_gap"] * np.log(1.1)) - 1),
                         occupancy_p=float(ols_occ.pvalues["log_gap"]))

# %%
elig["gap_q"] = pd.qcut(elig.log_gap, 5, labels=False)
qt = elig.groupby("gap_q").agg(gap_med=("gap", "median"), rev_mean=("number_of_reviews_ltm", "mean"), rev_med=("number_of_reviews_ltm", "median"), n=("gap", "size"))
display(qt.round(2))

fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]
g_ = np.clip(df.gap.values, -0.7, 1.5)
bins_ = np.linspace(-0.7, 1.5, 45)
cnt, edges_ = np.histogram(g_, bins=bins_)
centers = (edges_[:-1] + edges_[1:]) / 2
colr = [PAL["orange"] if c >= TH_HI else (PAL["aqua"] if c <= TH_LO else PAL["blue"]) for c in centers]
a.bar(centers, cnt, width=(edges_[1] - edges_[0]) * 0.92, color=colr)
a.set_xlabel("Price gap vs. benchmark (actual ÷ benchmark − 1)"); a.set_ylabel("Number of listings")
a.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
a.text(0.78, cnt.max() * 0.92, f"Above\nbenchmark\n{pct(pos_share.get('Above benchmark',0))}", color=PAL["orange"], fontsize=7.5, fontweight="bold", va="top")
a.text(-0.68, cnt.max() * 0.92, f"Below\nbenchmark\n{pct(pos_share.get('Below benchmark',0))}", color=PAL["aqua"], fontsize=7.5, fontweight="bold", va="top")
a.text(0.0, cnt.max() * 1.02, f"In line {pct(pos_share.get('In line',0))}", color=PAL["blue"], fontsize=7.5, fontweight="bold", ha="center")
a.set_ylim(0, cnt.max() * 1.12); a.grid(axis="x", visible=False)
a.set_title(f"{pct(1 - pos_share.get('In line', 0))} of listings sit outside the ±25% band (clipped at 150%)", fontsize=8.2)
b = ax[1]
labs = [f"{v*100:+.0f}%" for v in qt.gap_med]
b.bar(range(5), qt.rev_mean, color=PAL["blue"], width=0.66)
for i, (v, n) in enumerate(zip(qt.rev_mean, qt.n)):
    b.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=7.5)
b.set_xticks(range(5)); b.set_xticklabels(labs); b.grid(axis="x", visible=False)
b.set_xlabel("Price gap vs. benchmark, by quintile (median)"); b.set_ylabel("Reviews in the last 12 months (mean)")
b.set_title("Priced above benchmark = fewer reviews", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig7_price_gap_demand", fig)
RESULTS["gap"] = dict(shares={k: float(v) for k, v in pos_share.items()}, thresholds=[TH_LO, TH_HI],
                      quintiles=qt.reset_index().to_dict(orient="records"))
