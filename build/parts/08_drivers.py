# %% [markdown]
# ## 9. What drives the price? Importance of each factor family
# A host or manager wants to know **which information really matters**. We use *permutation importance*, an explainability method that is easy to state: take the selected model,
# **scramble one family of factors** (for example all location columns) among the test listings, and see how much worse the predictions become. If scrambling a family hardly
# changes the error, the model does not need it; if the error jumps, the family is important. We use the **test hosts** (never seen in training), scramble every family 5 times
# and report the increase in the typical (median) percentage error, in percentage points.

# %%
def family_columns(factors):
    """Columns of the model table X that belong to the given factors (a category became several 0/1 columns)."""
    return [c for c in X.columns if c in factors or any(c.startswith(f + "_") for f in factors if f in CAT_COLS)]

rng = np.random.default_rng(SEED)
model_best = models[BEST]
error_normal = evaluate(p_test, np.exp(model_best.predict(X_test)))["MedAPE"]        # typical error without scrambling
rows = []
for family, factors in FAMILIES.items():
    cols = family_columns(factors)
    increases = []
    for repetition in range(5):
        X_scrambled = X_test.copy()
        X_scrambled[cols] = X_test[cols].values[rng.permutation(len(X_test))]        # shuffle the rows of this family together
        increases.append(evaluate(p_test, np.exp(model_best.predict(X_scrambled)))["MedAPE"] - error_normal)
    rows.append(dict(family=family, increase_pts=100 * np.mean(increases), n_columns=len(cols)))
imp = pd.DataFrame(rows).sort_values("increase_pts", ascending=False)
print(f"Typical error of the {BEST} model on test hosts: {pct(error_normal, 1)}")
display(imp.round(2))
RESULTS["importance"] = dict(base_error=float(error_normal), table=imp.to_dict(orient="records"))

# Figure 5: one bar per family
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ypos = np.arange(len(imp))[::-1]
ax.barh(ypos, imp.increase_pts, color=PAL["blue"], height=0.62)
ax.set_yticks(ypos); ax.set_yticklabels(imp.family, fontsize=7.6); ax.grid(axis="y", visible=False)
for y_, v in zip(ypos, imp.increase_pts):
    ax.text(v + 0.15, y_, f"+{v:.1f} pts", va="center", fontsize=7.4)
ax.set_xlim(0, imp.increase_pts.max() * 1.18)
ax.set_xlabel("Increase of the typical pricing error when the family is scrambled (percentage points, test hosts)")
ax.set_title(f"Importance of each factor family ({BEST}); '{imp.family.iloc[0]}' matters most", fontsize=8.4)
fig.tight_layout()
savefig("fig5_importance", fig)

# %% [markdown]
# **Finding.** The **property** (room type, size) and the **stay policy** (short stay or monthly) are the most important families: scrambling them makes the benchmark much worse.
# Amenities come third mostly because they mark the *type* of listing. Location, competition, host flags and **transit distance add very little** once the property and the stay are known.
# Caution: importance says what the model *uses*, not what *causes* the price; and correlated factors (for instance district and distance to Central) can share importance.
