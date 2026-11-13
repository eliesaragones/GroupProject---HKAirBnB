# %% [markdown]
# ## 8. Supervised learning: a benchmark that is *honest* for a real user
# **Target:** `log(price per night)`: errors read as percentages ("20% above market") and a few very expensive listings do not dominate the fit. Results are reported in HKD.
#
# **Three decisions protect the credibility of the result:**
# 1. **Split by host, not by listing.** A professional host with 40 near-identical listings would otherwise appear in both train and test and the model
#    would be rewarded for memorising the host. All listings of a host go to the same side (`GroupShuffleSplit`, 80/20); cross-validation also keeps hosts together (`GroupKFold`).
# 2. **Everything that is "learned" from data uses the training part only** (the median that fills missing values, the settings of the models).
# 3. **The model is chosen with cross-validation on the training data**; the test set is used once, to report the final performance.
#
# **Models:** the *comparable-listings baseline* (what hosts do today: median price of listings in the same district, room type and stay type), **linear regression**
# (simple and transparent), **random forest** and **gradient boosting** (non-linear effects and interactions without hand-made terms).
# **Metrics:** MAE in HKD, median absolute % error, share of listings predicted within ±20%, and RMSE and R² (on log prices).

# %%
X = pd.get_dummies(X_all, columns=CAT_COLS, dtype=int)         # predictors: categories become 0/1 columns (one-hot encoding)
y = np.log(df.price_hkd.values)                                # target: log price
groups = df.host_id.values                                     # the host of every listing

# Train/test split by host (80/20). One random split can be unbalanced because a few hosts own hundreds of listings (one: 344), so we try 50 random
# splits and keep the one whose test set is closest to the whole sample (share of monthly stays, median price, size). No model result is used for this.
def imbalance(test_rows):
    return (abs(X_all.long_stay.iloc[test_rows].mean() - X_all.long_stay.mean()) + abs(np.median(y[test_rows]) - np.median(y)) + abs(len(test_rows) / len(y) - 0.2))

splits = {s: next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s).split(X, y, groups)) for s in range(50)}
SPLIT_SEED = min(splits, key=lambda s: imbalance(splits[s][1]))
train_idx, test_idx = splits[SPLIT_SEED]
assert not set(groups[train_idx]) & set(groups[test_idx]), "a host appears in both train and test!"

train_median = X.iloc[train_idx].median()
X = X.fillna(train_median)                                     # missing values -> median of the TRAINING rows
X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
g_train = groups[train_idx]
p_train, p_test = np.exp(y_train), np.exp(y_test)              # prices in HKD (exp undoes the log)
print(f"split seed {SPLIT_SEED}: train {len(train_idx):,} listings / {len(set(g_train)):,} hosts | test {len(test_idx):,} listings / {len(set(groups[test_idx])):,} hosts")
print("monthly-stay share train vs test:", pct(X_all.long_stay.iloc[train_idx].mean()), "vs", pct(X_all.long_stay.iloc[test_idx].mean()))
RESULTS["split"] = dict(seed=int(SPLIT_SEED), n_train=len(train_idx), n_test=len(test_idx), hosts_train=len(set(g_train)),
                        hosts_test=len(set(groups[test_idx])), test_size=0.2, cv_folds=5)

def evaluate(p_true, p_pred):
    """Accuracy measures from real and predicted prices (HKD)."""
    err = np.abs(p_pred - p_true) / p_true                     # absolute percentage error of each prediction
    return dict(MAE=float(mean_absolute_error(p_true, p_pred)), RMSE=float(np.sqrt(np.mean((p_pred - p_true) ** 2))),
                R2_log=float(r2_score(np.log(p_true), np.log(p_pred))), MedAPE=float(np.median(err)), Within20=float(np.mean(err <= 0.20)))

def comparable_price(train_rows, test_rows):
    """The benchmark to beat: median price of the comparable listings of the training data (same district, room type and stay type);
    with fewer than 5 comparables: median of the same room type and stay type."""
    keys = ["district_grp", "room_type", "long_stay"]
    train = X_all.iloc[train_rows][keys].assign(price=np.exp(y[train_rows]))
    fine = train.groupby(keys).price.agg(["median", "size"]).query("size >= 5")["median"].rename("fine")
    coarse = train.groupby(["room_type", "long_stay"]).price.median().rename("coarse")
    pred = X_all.iloc[test_rows][keys].join(fine, on=keys).join(coarse, on=["room_type", "long_stay"])
    return pred.fine.fillna(pred.coarse).fillna(train.price.median()).values

# %% [markdown]
# **Training.** Random forest and gradient boosting have settings (e.g. the depth of the trees). We choose them with a small grid search: every combination is
# evaluated with 5-fold cross-validation on the training data (folds keep each host's listings together) and the best one is kept. The error used is the MAE of the log price.

# %%
cv = GroupKFold(n_splits=5)
def tune(model, grid):
    """Grid search with host-grouped 5-fold cross-validation on the training data."""
    return GridSearchCV(model, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1).fit(X_train, y_train, groups=g_train)

linear = LinearRegression().fit(X_train, y_train)
rf = tune(RandomForestRegressor(n_estimators=200, random_state=SEED), {"max_depth": [10, 20], "min_samples_leaf": [2, 5], "max_features": [0.3, 0.6]})
gb = tune(GradientBoostingRegressor(random_state=SEED), {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]})
models = {"Linear regression": linear, "Random forest": rf.best_estimator_, "Gradient boosting": gb.best_estimator_}
best_params = {"Linear regression": {}, "Random forest": rf.best_params_, "Gradient boosting": gb.best_params_}
cv_mae = {"Linear regression": -cross_val_score(LinearRegression(), X_train, y_train, cv=cv, groups=g_train, scoring="neg_mean_absolute_error").mean(),
          "Random forest": -rf.best_score_, "Gradient boosting": -gb.best_score_}
BEST = min(cv_mae, key=cv_mae.get)                             # chosen by cross-validation, never by the test score
print("CV error (MAE of the log price):", {k: round(v, 3) for k, v in cv_mae.items()}, "-> selected:", BEST)

# %% [markdown]
# **The final exam: the test set (used once).** We predict the 20% of listings that the models have never seen and compare with the real prices and with the two baselines.

# %%
preds = {"Global median": np.full(len(test_idx), np.median(p_train)),            # naive: everyone gets the overall median
         "Comparable-listings baseline": comparable_price(train_idx, test_idx)}  # what hosts do today
for name, model in models.items():
    preds[name] = np.exp(model.predict(X_test))                # exp: back from log price to HKD
results = pd.DataFrame({name: evaluate(p_test, p) for name, p in preds.items()}).T
results["CV MAE (log)"] = pd.Series(cv_mae)
display(results.round(3))
base_mae, best_mae = results.loc["Comparable-listings baseline", "MAE"], results.loc[BEST, "MAE"]
print(f"MAE: baseline {base_mae:.0f} HKD -> {BEST} {best_mae:.0f} HKD ({pct(1 - best_mae / base_mae)} lower)")

# How sure are we of the improvement? Bootstrap over HOSTS (a host's listings are not independent): add up the errors of each host, resample the hosts
# with replacement 2,000 times and recompute the improvement; the 2.5th and 97.5th percentiles give a 95% interval.
host_err = pd.DataFrame({"host": groups[test_idx], "baseline": np.abs(preds["Comparable-listings baseline"] - p_test), "best": np.abs(preds[BEST] - p_test)}).groupby("host").sum()
rng = np.random.default_rng(SEED)
gains = []
for _ in range(2000):
    s = host_err.sample(frac=1, replace=True, random_state=int(rng.integers(1_000_000_000)))
    gains.append(1 - s.best.sum() / s.baseline.sum())
ci = np.percentile(gains, [2.5, 97.5])
print(f"95% host-bootstrap interval of the improvement: {pct(ci[0], 1)} to {pct(ci[1], 1)}")

# Accuracy of the selected model by group (test set)
def error_by(group):
    d = pd.DataFrame({"group": np.asarray(group), "real": p_test, "pred": preds[BEST]})
    rows = {}
    for g, s in d.groupby("group"):
        e = evaluate(s.real.values, s.pred.values)
        rows[g] = dict(n=len(s), median_price=s.real.median(), MAE=e["MAE"], MedAPE=e["MedAPE"], Within20=e["Within20"])
    return pd.DataFrame(rows).T

room = X_all.room_type.iloc[test_idx].where(X_all.room_type.iloc[test_idx].isin(["Private room", "Entire home/apt"]), "Other")
stay = np.where(X_all.long_stay.iloc[test_idx] == 1, "long stay", "short stay")
by_room_stay = error_by(room.values + " | " + stay)
by_segment = error_by(df.segment.iloc[test_idx].values).sort_values("median_price")      # segments of section 7
display(by_room_stay.round(3)); display(by_segment.round(3))

RESULTS["models"] = dict(table={k: {m: float(v) for m, v in r.items() if pd.notna(v)} for k, r in results.iterrows()}, best=BEST, best_params=best_params,
                         mae_reduction=float(1 - best_mae / base_mae), mae_reduction_ci=[float(ci[0]), float(ci[1])], cv_mae_log={k: float(v) for k, v in cv_mae.items()},
                         by_regime_room={k: r.to_dict() for k, r in by_room_stay.iterrows()}, by_segment={k: r.to_dict() for k, r in by_segment.iterrows()})

# %%
# Figure 4: left = error of every method, right = predicted vs. real price of the selected model
order = ["Global median", "Comparable-listings baseline", "Linear regression", "Random forest", "Gradient boosting"]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.05]))
a = ax[0]
vals = [results.loc[o, "MAE"] for o in order]
colors = [PAL["muted"] if o in order[:2] else (PAL["orange"] if o == BEST else PAL["blue"]) for o in order]     # grey = baselines, orange = selected model
a.barh([o.replace("Comparable-listings baseline", "Comparable listings\n(district x room x stay)") for o in order][::-1], vals[::-1], color=colors[::-1], height=0.62)
for i, (o, v) in enumerate(zip(order[::-1], vals[::-1])):
    a.text(v + 8, i, f"{v:,.0f} HKD  ({pct(results.loc[o, 'MedAPE'])} typical miss)", va="center", fontsize=6.8)
a.set_xlim(0, max(vals) * 1.55); a.set_xlabel("Mean absolute error on unseen hosts (HKD per night)"); a.grid(axis="y", visible=False)
a.set_title("The model beats comparable-listing pricing", fontsize=8.2)
b = ax[1]
is_long = X_all.long_stay.iloc[test_idx].values == 1
for flag, col, lab in [(False, PAL["blue"], "Short stay"), (True, PAL["orange"], "Long stay")]:
    b.scatter(preds[BEST][is_long == flag], p_test[is_long == flag], s=6, alpha=0.45, color=col, linewidths=0, label=lab)     # each dot = one unseen listing
lim = np.array([60, 8000])
b.plot(lim, lim, color=PAL["ink"], lw=0.9)                                                                      # perfect prediction
b.plot(lim, lim * 1.2, color=PAL["muted"], lw=0.7, ls="--"); b.plot(lim, lim / 1.2, color=PAL["muted"], lw=0.7, ls="--")   # +-20%
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lim); b.set_ylim(lim); b.set_xticks([100, 300, 1000, 3000]); b.set_yticks([100, 300, 1000, 3000])
b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()); b.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
b.set_xlabel("Predicted price (HKD)"); b.set_ylabel("Actual price (HKD)"); b.legend(loc="upper left", markerscale=2.2)
b.text(2500, 2000 / 1.25 * 0.74, "±20%", fontsize=6.5, color=PAL["muted"], ha="right")
b.set_title(f"{pct(results.loc[BEST, 'Within20'])} of unseen listings within ±20%", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig4_model_performance", fig)

# %% [markdown]
# **Finding.** The best machine-learning model reduces the typical error by about a fifth or more compared with what a host does today, and the improvement is not luck
# (the bootstrap interval stays well above zero). The points in the right-hand chart follow the diagonal but are far from perfect, which is realistic: asking prices
# are noisy. That is why the benchmark always comes with a **price band** (section 12).
