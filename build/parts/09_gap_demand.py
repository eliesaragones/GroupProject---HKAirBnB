# %% [markdown]
# ## 10. From prediction to action: who is priced away from the benchmark, and does it cost them guests?
# A prediction is only useful if it changes a decision. We use the benchmark as a **price audit**:
# 1. for *every* listing we take the prediction of a model that has never seen that host (5-fold cross-validation grouped by host);
# 2. **price gap** = real price ÷ benchmark − 1; a listing is *above benchmark* if it is ≥ 25% above, *below benchmark* if ≥ 20% below (the same ±25% band on a ratio scale);
# 3. we ask what a host cares about: **do listings priced above the benchmark attract fewer guests?** We use the number of reviews in the last 12 months as a demand proxy
#    (reviews are a roughly constant fraction of stays).

# %%
p_all = df.price_hkd.values                                  # real prices (HKD)
# Predictions for all listings from models that never saw the listing's host, for the selected model and for the baseline
folds = GroupKFold(n_splits=5)
oof = cross_val_predict(clone(models[BEST]), X, y, cv=folds, groups=groups, n_jobs=-1)       # predicted log price
oof_base = np.zeros(len(df))
for rows_train, rows_test in folds.split(X, y, groups):
    oof_base[rows_test] = comparable_price(rows_train, rows_test)
m_oof, m_base = evaluate(p_all, np.exp(oof)), evaluate(p_all, oof_base)
print(f"All {len(df):,} listings, host-grouped 5-fold CV: {BEST} MAE {m_oof['MAE']:.0f} HKD, typical error {pct(m_oof['MedAPE'], 1)}, within ±20%: {pct(m_oof['Within20'])} | "
      f"baseline MAE {m_base['MAE']:.0f} HKD, typical error {pct(m_base['MedAPE'], 1)}")
RESULTS["oof"] = dict(model=m_oof, baseline=m_base, mae_reduction=float(1 - m_oof["MAE"] / m_base["MAE"]))

# Price gap of every listing and its label (np.select: the first condition that is true decides the label)
df["pred"] = np.exp(oof)                                    # benchmark price (HKD)
df["gap"] = df.price_hkd / df.pred - 1                      # +0.30 = "30% above the benchmark"
TH_HI, TH_LO = 0.25, 1 / 1.25 - 1                           # +25% and -20%
df["position"] = np.select([df.gap >= TH_HI, df.gap <= TH_LO], ["Above benchmark", "Below benchmark"], "In line")
pos_share = df.position.value_counts(normalize=True)
display(pos_share.round(3))
# listing-by-listing price audit that a host or manager can open in Excel
df[["id", "price_hkd", "pred", "gap", "position", "segment", "regime", "neighbourhood_cleansed", "room_type"]].round(3).to_csv(OUT / "listing_price_audit.csv", index=False)

# %% [markdown]
# **Does being above the benchmark cost guests?** We only use listings with at least a year of history (a new listing would look "unpopular" just because it is new) and compare
# listings *of the same kind* (stay type, room type, district, size). The outcome is a **count** (0, 1, 2... reviews), so we use a **Poisson regression**.

# %%
elig = df[(df.first_review <= SCRAPE_DATE - pd.Timedelta(days=365)) & df.review_scores_rating.notna()].copy()
elig["log_gap"] = np.log(elig.price_hkd / elig.pred)         # 0 = priced exactly at the benchmark
elig["acc_log"] = np.log(elig.accommodates)
controls = "C(regime) + C(room_type) + C(neighbourhood_cleansed) + acc_log"
formulas = {"controls": f"number_of_reviews_ltm ~ log_gap + {controls}", "controls + rating": f"number_of_reviews_ltm ~ log_gap + {controls} + review_scores_rating"}
irr = {}                                                     # effect on the number of reviews of a price 10% above the benchmark
for name, formula in formulas.items():
    fit = smf.glm(formula, data=elig, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": elig.host_id.astype(int)})
    b, se, k = fit.params["log_gap"], fit.bse["log_gap"], np.log(1.10)      # k = "+10% price" on the log scale
    irr[name] = dict(irr_10=float(np.exp(b * k)), lo=float(np.exp((b - 1.96 * se) * k)), hi=float(np.exp((b + 1.96 * se) * k)), p=float(fit.pvalues["log_gap"]))
    print(f"{len(elig):,} listings | {name:18s}: +10% above benchmark -> reviews x{irr[name]['irr_10']:.3f} (95% CI {irr[name]['lo']:.3f}-{irr[name]['hi']:.3f}), p = {irr[name]['p']:.3g}")
RESULTS["demand"] = dict(n=len(elig), irr=irr)

# The same relation as a simple table: five equal-sized groups ("quintiles") by price gap and the average number of reviews of each
elig["gap_group"] = pd.qcut(elig.log_gap, 5, labels=False)   # 0 = cheapest relative to the benchmark ... 4 = most expensive
quint = elig.groupby("gap_group").agg(gap_med=("gap", "median"), rev_mean=("number_of_reviews_ltm", "mean"), n=("gap", "size"))
display(quint.round(2))

# Figure 7: left = histogram of the price gaps, right = reviews by price-gap group
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]
counts, edges = np.histogram(np.clip(df.gap.values, -0.7, 1.5), bins=np.linspace(-0.7, 1.5, 45))     # gaps beyond the limits are clipped
centers = (edges[:-1] + edges[1:]) / 2
a.bar(centers, counts, width=(edges[1] - edges[0]) * 0.92, color=[PAL["orange"] if c >= TH_HI else (PAL["aqua"] if c <= TH_LO else PAL["blue"]) for c in centers])
a.set_xlabel("Price gap vs. benchmark (actual ÷ benchmark − 1)"); a.set_ylabel("Number of listings")
a.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
a.text(0.78, counts.max() * 0.92, f"Above\nbenchmark\n{pct(pos_share.get('Above benchmark', 0))}", color=PAL["orange"], fontsize=7.5, fontweight="bold", va="top")
a.text(-0.68, counts.max() * 0.92, f"Below\nbenchmark\n{pct(pos_share.get('Below benchmark', 0))}", color=PAL["aqua"], fontsize=7.5, fontweight="bold", va="top")
a.text(0.0, counts.max() * 1.02, f"In line {pct(pos_share.get('In line', 0))}", color=PAL["blue"], fontsize=7.5, fontweight="bold", ha="center")
a.set_ylim(0, counts.max() * 1.12); a.grid(axis="x", visible=False)
a.set_title(f"{pct(1 - pos_share.get('In line', 0))} of listings sit outside the ±25% band (clipped at 150%)", fontsize=8.2)
b = ax[1]
b.bar(range(5), quint.rev_mean, color=PAL["blue"], width=0.66)
for i, v in enumerate(quint.rev_mean):
    b.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=7.5)
b.set_xticks(range(5)); b.set_xticklabels([f"{v*100:+.0f}%" for v in quint.gap_med]); b.grid(axis="x", visible=False)
b.set_xlabel("Price gap vs. benchmark, by quintile (median)"); b.set_ylabel("Reviews in the last 12 months (mean)")
b.set_title("Priced above benchmark = fewer reviews", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig7_price_gap_demand", fig)
RESULTS["gap"] = dict(shares={k: float(v) for k, v in pos_share.items()}, thresholds=[TH_LO, TH_HI], quintiles=quint.reset_index().to_dict(orient="records"))

# %% [markdown]
# **Finding.** About half of the listings are priced more than 20-25% away from their benchmark. Among listings of the same kind, **the more a listing is priced above its
# benchmark, the fewer reviews it gets** (the most expensive fifth gets about half the reviews of those in line), while discounts do not bring much extra. Careful: this is an
# *association* (a listing can be expensive *and* unpopular for a reason we cannot see, such as poor photos), not proof that lowering the price will raise demand.

# %% [markdown]
# ## 11. Existing listings: do ratings and reviews add information?
# The benchmark is for a **new** listing. An operator with a live listing also knows its ratings and review activity. We compare two gradient-boosting models with the same
# settings, one with the launch-time factors only and one that also knows the ratings and review activity, on the test hosts.

# %%
EXP_FILLED = EXP_ALL.fillna(EXP_ALL.iloc[train_idx].median())      # missing ratings -> training median
XX = pd.concat([X, EXP_FILLED], axis=1)                            # launch-time factors + experience factors
best_gb = gb.best_estimator_                                       # tuned gradient boosting
launch_cols = list(X.columns)
model_launch = clone(best_gb).fit(XX.iloc[train_idx][launch_cols], y_train)
model_exp = clone(best_gb).fit(XX.iloc[train_idx], y_train)                       # all columns: launch-time + experience
has_rating = EXP_ALL.rs_rating.iloc[test_idx].notna().values                       # test listings that have a rating

def test_errors(model, cols, rows):
    return evaluate(p_test[rows], np.exp(model.predict(XX.iloc[test_idx][cols][rows])))

all_rows = np.ones(len(test_idx), dtype=bool)
compare = pd.DataFrame({"Launch-time model (all test)": test_errors(model_launch, launch_cols, all_rows),
                        "+ experience (all test)": test_errors(model_exp, list(XX.columns), all_rows),
                        f"Launch-time model (rated, n={has_rating.sum()})": test_errors(model_launch, launch_cols, has_rating),
                        "+ experience (rated)": test_errors(model_exp, list(XX.columns), has_rating)}).T
display(compare[["MAE", "MedAPE", "Within20", "R2_log"]].round(3))
RESULTS["existing_model"] = {k: r.to_dict() for k, r in compare.iterrows()}

# %% [markdown]
# **Finding.** Ratings and review activity improve the prediction only slightly: **guests do not pay a large premium for reputation in this market; the price is mostly set by what the listing is.**
