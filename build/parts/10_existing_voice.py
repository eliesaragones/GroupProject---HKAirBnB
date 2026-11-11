# %% [markdown]
# ## 13. Existing listings: what do reputation and guest feedback add?
#
# The launch-time benchmark answers *"what should a new listing charge?"*. Operators with live listings ask a second question:
# *"given my ratings and reviews, is my price justified?"* We therefore fit a second model that **adds the experience families**
# (ratings, review activity, guest-voice text features) to the launch-time factors, using the same split and hyper-parameters,
# and compare it with the launch-time model on the test hosts that have reviews.
#
# We then look at the **guest voice**: what do guests of listings priced *above* vs. *below* benchmark talk about? This links
# the quantitative price audit with qualitative feedback.

# %%
rated_te = XX.iloc[te].rs_rating.notna().values
mdl_exp = make_pipe(NEW_COLS + EXP_COLS, clone(gb_model)).fit(XX.iloc[tr], ytr)
mdl_new = make_pipe(NEW_COLS, clone(gb_model)).fit(XX.iloc[tr], ytr)
m_new_all, m_exp_all = metrics(pte, np.exp(mdl_new.predict(XX.iloc[te]))), metrics(pte, np.exp(mdl_exp.predict(XX.iloc[te])))
m_new_r = metrics(pte[rated_te], np.exp(mdl_new.predict(XX.iloc[te][rated_te])))
m_exp_r = metrics(pte[rated_te], np.exp(mdl_exp.predict(XX.iloc[te][rated_te])))
cmp_exp = pd.DataFrame({"Launch-time model (all test)": m_new_all, "+ experience (all test)": m_exp_all,
                        f"Launch-time model (rated, n={rated_te.sum()})": m_new_r, "+ experience (rated)": m_exp_r}).T
display(cmp_exp[["MAE", "MedAPE", "Within20", "R2_log"]].round(3))
RESULTS["existing_model"] = {k: {a: float(b) for a, b in r.items()} for k, r in cmp_exp.iterrows()}

# %%
# --- guest voice by price position ---
gv = df[["position", "regime"] + []].join(EXP_ALL[GV_COLS]).dropna(subset=["gv_sentiment"])
topic_names = {"gv_location": "Mentions location / transit", "gv_clean": "Praises cleanliness", "gv_dirty": "Complains of dirt / smell / bugs",
               "gv_noise": "Mentions noise", "gv_small": "Says small / cramped", "gv_value": "Says good value",
               "gv_pricey": "Says expensive / overpriced", "gv_host": "Praises host service"}
tab = gv.groupby("position")[list(topic_names)].mean().T.rename(index=topic_names)
tab["sentiment"] = np.nan
tab.loc["Average review sentiment (VADER, -1..+1)"] = gv.groupby("position").gv_sentiment.mean()
tab = tab.drop(columns="sentiment")
cols_order = [c for c in ["Below benchmark", "In line", "Above benchmark"] if c in tab.columns]
tab = tab[cols_order]
display((tab * 100).round(1).rename(columns={c: f"{c} (n={int((gv.position==c).sum())})" for c in cols_order}))
# significance of above vs below for each topic (Mann-Whitney) - exploratory, so we also report how many tests were run
from scipy.stats import mannwhitneyu
sig = {}
for c in list(topic_names) + ["gv_sentiment"]:
    a_ = gv.loc[gv.position == "Above benchmark", c]; b_ = gv.loc[gv.position == "Below benchmark", c]
    if len(a_) > 10 and len(b_) > 10:
        sig[c] = float(mannwhitneyu(a_, b_).pvalue)
print("Mann-Whitney p-values, above vs below benchmark (9 exploratory tests; Bonferroni threshold 0.0056):",
      {k: round(v, 4) for k, v in sig.items()})
RESULTS["guest_voice"] = dict(n=int(len(gv)), table={r: {c: float(v) for c, v in tab.loc[r].items()} for r in tab.index}, pvalues=sig)

# %% [markdown]
# **Business reading.** Only a minority of listings can be analysed this way (English reviews, ≥ 3 of them), so these results
# are *exploratory* – we apply a Bonferroni correction because we look at nine topics at once. Where a difference survives that
# correction it is a lead for the host (e.g. "overpriced" complaints), not proof.
