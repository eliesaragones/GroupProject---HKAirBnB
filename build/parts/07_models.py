# %% [markdown]
# ## 9. Supervised learning: designing a benchmark that is *honest* for a real user
#
# **Target.** `log(price per night)`. Errors then read as percentages, which is how hosts think ("20% above market"), and
# the skewed HKD distribution stops a few expensive listings from dominating the fit. All results are reported back in HKD.
#
# **Evaluation design – the three decisions that protect the credibility of the result**
# 1. **Split by host, not by listing.** A professional host with 40 near-identical listings would otherwise appear in both
#    train and test and the model would be rewarded for memorising a host. All listings of a host go to the *same* side
#    (`GroupShuffleSplit`, 80/20) and cross-validation also respects hosts (`GroupKFold`). The test set is touched **once**.
# 2. **All preprocessing lives inside a `Pipeline`.** Imputation values, scaling and one-hot categories are learned on the training
#    folds only, so no information leaks from validation/test data.
# 3. **Model selection uses cross-validation on the training data only**; the test set is used only to report final performance.
#
# **Models (and why).**
# | Model | Why it is in the comparison |
# |---|---|
# | *Comparable-listings baseline* | What hosts do today: the median price of listings in the same district, room type and stay regime. Any ML model must beat this to be worth building |
# | **Ridge regression** (regularised linear) | Transparent, stable with many correlated dummies (amenities); shows how far a purely linear logic gets |
# | **Random forest** | Captures non-linear effects and interactions (e.g. a bedroom is worth more on HK Island) without hand-made terms |
# | **Gradient boosting** (`HistGradientBoostingRegressor`) | Usually the strongest model on tabular data; sequentially corrects residuals |
# | **Blend (RF + GB)** | Averages two different error profiles |
#
# **Metrics (business first).** *MAE* in HKD (typical size of the error), *median absolute % error* (typical relative miss),
# *share of listings predicted within ±20%* (a host's practical "is this price about right?"), plus RMSE and R² for completeness.

# %%
y = np.log(df.price_hkd.values)
groups = df.host_id.values
# A single random split by host can be badly unbalanced because a few hosts own hundreds of listings (one host: 344). We therefore
# draw 300 candidate host-level splits and keep the one whose test set best mirrors the full sample in stay regime, room type and
# price level. This uses only the *composition* of the data, never any model result.
def split_imbalance(te_i):
    t = np.zeros(len(y), bool); t[te_i] = True
    return (abs(X_all.long_stay[t].mean() - X_all.long_stay.mean()) / 0.04
            + abs((X_all.room_type[t] == "Entire home/apt").mean() - (X_all.room_type == "Entire home/apt").mean()) / 0.04
            + abs(np.median(y[t]) - np.median(y)) / 0.08 + abs(t.mean() - 0.2) / 0.02)
cands = {}
for seed in range(300):
    tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_all, y, groups=groups))
    cands[seed] = (split_imbalance(te_i), tr_i, te_i)
SPLIT_SEED = min(cands, key=lambda k: cands[k][0])
_, tr, te = cands[SPLIT_SEED]
print(f"chosen split seed {SPLIT_SEED} (imbalance score {cands[SPLIT_SEED][0]:.2f}; median over 300 candidates {np.median([v[0] for v in cands.values()]):.2f})")
assert not set(groups[tr]) & set(groups[te]), "host leakage between train and test!"
Xtr, Xte, ytr, yte, gtr = X_all.iloc[tr], X_all.iloc[te], y[tr], y[te], groups[tr]
ptr, pte = np.exp(ytr), np.exp(yte)
print(f"train {len(tr):,} listings / {len(set(gtr)):,} hosts | test {len(te):,} listings / {len(set(groups[te])):,} hosts | hosts shared: 0")
print("median price train vs test:", round(np.median(ptr)), "vs", round(np.median(pte)),
      "| long-stay share:", pct(Xtr.long_stay.mean()), "vs", pct(Xte.long_stay.mean()))
RESULTS["split"] = dict(seed=int(SPLIT_SEED), n_train=len(tr), n_test=len(te), hosts_train=len(set(gtr)), hosts_test=len(set(groups[te])),
                        test_size=0.2, cv_folds=5)

def make_pre(cols, scale=False):
    cat = [c for c in cols if c in CAT_COLS]
    num = [c for c in cols if c not in CAT_COLS]
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler() if scale else "passthrough")])
    return ColumnTransformer([("num", num_pipe, num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat)])

def make_pipe(cols, model, scale=False):
    return Pipeline([("pre", make_pre(cols, scale)), ("model", model)])

def metrics(p_true, p_pred):
    e = p_pred - p_true
    ape = np.abs(e) / p_true
    lt, lp = np.log(p_true), np.log(p_pred)
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                R2=float(1 - np.sum(e ** 2) / np.sum((p_true - p_true.mean()) ** 2)),
                R2_log=float(1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2)),
                MedAPE=float(np.median(ape)), Within20=float(np.mean(ape <= 0.20)), Within30=float(np.mean(ape <= 0.30)))

class CompsBaseline:
    """What a host does by hand: median price of 'comparable' listings; falls back to coarser groups when a cell is thin."""
    LEVELS = [["district_grp", "room_type", "long_stay"], ["room_type", "long_stay"], ["long_stay"]]
    def __init__(self, levels=None, min_n=5): self.levels, self.min_n = levels or self.LEVELS, min_n
    def fit(self, X, price):
        d = X.assign(_p=np.asarray(price)); self.tables_ = []
        for lv in self.levels:
            t = d.groupby(lv)._p.agg(["median", "size"]); self.tables_.append((lv, t[t["size"] >= self.min_n]["median"]))
        self.global_ = float(np.median(price)); return self
    def predict(self, X):
        out = pd.Series(np.nan, index=X.index)
        for lv, t in self.tables_:
            key = pd.MultiIndex.from_frame(X[lv]) if len(lv) > 1 else X[lv[0]]
            vals = pd.Series(t.reindex(key).values, index=X.index)
            out = out.fillna(vals)
        return out.fillna(self.global_).values

# %%
cv = GroupKFold(n_splits=5)
t0 = time.time()
SEARCH = {
    "Ridge": (make_pipe(NEW_COLS, Ridge(random_state=RANDOM_STATE), scale=True),
              {"model__alpha": np.logspace(-1, 2.5, 25)}, 15),
    "Random forest": (make_pipe(NEW_COLS, RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=1)),
                      {"model__min_samples_leaf": [1, 2, 3, 5, 8], "model__max_features": [0.2, 0.35, 0.5, 0.7, 1.0],
                       "model__max_depth": [None, 12, 20]}, 10),
    "Gradient boosting": (make_pipe(NEW_COLS, HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
                          {"model__learning_rate": [0.03, 0.05, 0.08, 0.12], "model__max_iter": [200, 350, 500],
                           "model__max_leaf_nodes": [8, 15, 31], "model__min_samples_leaf": [10, 20, 40],
                           "model__l2_regularization": [0, 0.1, 1, 5], "model__max_depth": [None, 6, 10]}, 20),
}
import os, joblib
tuned, cv_scores, best_params = {}, {}, {}
CACHE_TUNED = OUT / "_tuned_cache.joblib"          # development shortcut only (set IS3240_CACHE=1); a normal run always re-tunes
if os.environ.get("IS3240_CACHE") == "1" and CACHE_TUNED.exists():
    tuned, cv_scores, best_params = joblib.load(CACHE_TUNED)
else:
    for name, (pipe, space, n_iter) in SEARCH.items():
        rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=cv, scoring="neg_mean_absolute_error",
                                random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        rs.fit(Xtr, ytr, groups=gtr)
        tuned[name], cv_scores[name], best_params[name] = rs.best_estimator_, -rs.best_score_, rs.best_params_
    if os.environ.get("IS3240_CACHE") == "1":
        joblib.dump((tuned, cv_scores, best_params), CACHE_TUNED)
for name in SEARCH:
    print(f"{name:18s} CV MAE (log-price) = {cv_scores[name]:.3f}   best: { {k.split('__')[1]: v for k, v in best_params[name].items()} }")
print(f"tuning took {time.time()-t0:.0f}s")

# %%
# ---- final test-set evaluation (touched once) ----
preds = {}
preds["Global median"] = np.full(len(te), np.median(ptr))
comps = CompsBaseline().fit(Xtr, ptr)
preds["Comparable-listings baseline"] = comps.predict(Xte)
for name, est in tuned.items():
    preds[name] = np.exp(est.predict(Xte))
preds["Blend (RF + GB)"] = np.exp((np.log(preds["Random forest"]) + np.log(preds["Gradient boosting"])) / 2)

res = pd.DataFrame({k: metrics(pte, v) for k, v in preds.items()}).T
res["CV MAE (log)"] = pd.Series({**cv_scores})
display(res.round(3))
ML_NAMES = ["Ridge", "Random forest", "Gradient boosting"]
BEST = min(ML_NAMES, key=lambda n: cv_scores[n])        # selection by training-set CV, never by test score
print("Model selected by cross-validation:", BEST)
base_mae, best_mae = res.loc["Comparable-listings baseline", "MAE"], res.loc[BEST, "MAE"]
print(f"MAE: baseline {base_mae:.0f} HKD -> {BEST} {best_mae:.0f} HKD  ({pct(1-best_mae/base_mae)} lower)")
print(f"Median % error: baseline {pct(res.loc['Comparable-listings baseline','MedAPE'],1)} -> {pct(res.loc[BEST,'MedAPE'],1)};  within +-20%: {pct(res.loc['Comparable-listings baseline','Within20'])} -> {pct(res.loc[BEST,'Within20'])}")

# host-clustered bootstrap CI of the MAE reduction vs. the baseline (hosts are resampled, not listings)
rng = np.random.default_rng(RANDOM_STATE)
hid = pd.factorize(groups[te])[0]; nh = hid.max() + 1
ab_base, ab_best = np.abs(preds["Comparable-listings baseline"] - pte), np.abs(preds[BEST] - pte)
boot = []
for _ in range(2000):
    w = rng.multinomial(nh, np.ones(nh) / nh)[hid]
    boot.append(1 - (w * ab_best).sum() / (w * ab_base).sum())
ci = np.percentile(boot, [2.5, 97.5])
print(f"MAE reduction vs. baseline: {pct(1-best_mae/base_mae,1)}  (95% host-bootstrap CI {pct(ci[0],1)} to {pct(ci[1],1)})")

RESULTS["models"] = dict(table={k: {m: float(v) for m, v in r.items() if pd.notna(v)} for k, r in res.iterrows()},
                         best=BEST, best_params={k: {a.split('__')[1]: (float(b) if isinstance(b, (float, np.floating)) else b) for a, b in v.items()} for k, v in best_params.items()},
                         mae_reduction=float(1 - best_mae / base_mae), mae_reduction_ci=[float(ci[0]), float(ci[1])],
                         cv_mae_log={k: float(v) for k, v in cv_scores.items()})

# %%
# ---- performance by segment (test set) ----
def seg_table(by):
    d = pd.DataFrame({"g": by, "p": pte, "f": preds[BEST]})
    rows = []
    for k, s in d.groupby("g"):
        m = metrics(s.p.values, s.f.values)
        rows.append(dict(group=k, n=len(s), median_price=s.p.median(), MAE=m["MAE"], MedAPE=m["MedAPE"], Within20=m["Within20"]))
    return pd.DataFrame(rows).set_index("group")

reg_room = (X_all.iloc[te].room_type.where(X_all.iloc[te].room_type.isin(["Private room", "Entire home/apt"]), "Other") + " | " +
            np.where(Xte.long_stay == 1, "long stay", "short stay"))
seg_perf = seg_table(reg_room.values)
dist_perf = seg_table(Xte.district_grp.values).sort_values("n", ascending=False)
cluster_perf = seg_table(df.segment.iloc[te].values).sort_values("median_price")
display(seg_perf.round(3)); display(dist_perf.round(3)); display(cluster_perf.round(3))
RESULTS["models"]["by_regime_room"] = {k: {c: float(v) for c, v in r.items()} for k, r in seg_perf.iterrows()}
RESULTS["models"]["by_district"] = {k: {c: float(v) for c, v in r.items()} for k, r in dist_perf.iterrows()}
RESULTS["models"]["by_segment"] = {k: {c: float(v) for c, v in r.items()} for k, r in cluster_perf.iterrows()}

# %%
# ---- figure 4: how much better than the host's current approach, and what does a prediction look like? ----
order = ["Global median", "Comparable-listings baseline", "Ridge", "Random forest", "Gradient boosting", "Blend (RF + GB)"]
nice = {"Global median": "Global median", "Comparable-listings baseline": "Comparable listings\n(district x room x stay)", "Ridge": "Ridge regression",
        "Random forest": "Random forest", "Gradient boosting": "Gradient boosting", "Blend (RF + GB)": "Blend (RF + GB)"}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.05]))
a = ax[0]
vals = [res.loc[o, "MAE"] for o in order]
cols = [PAL["muted"] if o in ("Global median", "Comparable-listings baseline") else PAL["blue"] for o in order]
cols = [PAL["orange"] if o == BEST else c for o, c in zip(order, cols)]
a.barh([nice[o] for o in order][::-1], vals[::-1], color=cols[::-1], height=0.62)
for i, (o, v) in enumerate(zip(order[::-1], vals[::-1])):
    a.text(v + 8, i, f"{v:,.0f} HKD  ({pct(res.loc[o,'MedAPE'])} typical miss)", va="center", fontsize=6.8)
a.set_xlim(0, max(vals) * 1.55); a.set_xlabel("Mean absolute error on unseen hosts (HKD per night)"); a.grid(axis="y", visible=False)
a.set_title("The model beats comparable-listing pricing", fontsize=8.2)
b = ax[1]
mreg = Xte.long_stay.values == 1
for flag, col, lab in [(False, PAL["blue"], "Short stay"), (True, PAL["orange"], "Long stay")]:
    m = mreg == flag
    b.scatter(preds[BEST][m], pte[m], s=6, alpha=0.45, color=col, linewidths=0, label=lab)
lim = np.array([60, 8000])
b.plot(lim, lim, color=PAL["ink"], lw=0.9); b.plot(lim, lim * 1.2, color=PAL["muted"], lw=0.7, ls="--"); b.plot(lim, lim / 1.2, color=PAL["muted"], lw=0.7, ls="--")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lim); b.set_ylim(lim)
b.set_xticks([100, 300, 1000, 3000]); b.set_yticks([100, 300, 1000, 3000])
b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()); b.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
b.set_xlabel("Predicted price (HKD)"); b.set_ylabel("Actual price (HKD)")
b.legend(loc="upper left", markerscale=2.2); b.text(2500, 2000 / 1.25 * 0.74, "±20%", fontsize=6.5, color=PAL["muted"], ha="right")
b.set_title(f"{pct(res.loc[BEST,'Within20'])} of unseen listings within ±20%", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig4_model_performance", fig)
