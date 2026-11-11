# %% [markdown]
# ## 15. Decision log, limitations and recommendations
#
# ### Decisions we made (and why)
# | # | Decision | Business / statistical reason |
# |---|---|---|
# | 1 | Drop listings without a price (15% of the file) | Two thirds cannot be booked; a benchmark is about bookable listings. Risk: slight survivor bias |
# | 2 | Remove prices < 50 HKD or above the 99.5th percentile | Data errors (a 0.73 HKD tent) and a few luxury outliers that dominate squared error; the tool targets typical hosts |
# | 3 | Treat **stay length as a core factor** and a segmentation (≥ 28 nights) | The price is a quote for a specific stay; monthly stays cost a fraction per night |
# | 4 | Add **quote-context controls** (quoted nights, weekend check-in) but **not lead time** | They are properties of the scraped quote that move price; lead time partly reflects demand (leakage) |
# | 5 | Model `log(price)` | Percentage errors, tamed skewness |
# | 6 | Split and cross-validate **by host**; pick the most representative host split by composition only | Prevents memorising professional hosts; avoids an unbalanced test set |
# | 7 | All preprocessing inside a `Pipeline`; test set used once; models selected by CV | No leakage, honest estimate |
# | 8 | Compare with the **comparable-listings baseline** that hosts use today | The business question is "better than what I do now?" |
# | 8b | Keep only two binary host flags (professional, local) | Continuous host-history variables let trees memorise professional hosts and did not help on unseen hosts (22.8% vs 22.2% typical error) |
# | 9 | Launch-time model excludes ratings, reviews, superhost | A new host does not have them; they are used in a separate model for existing listings |
# | 10 | Distances from OpenStreetMap, only HK heavy-rail MTR | Reproducible, cached, Shenzhen/Light Rail removed |
# | 11 | Interpret with a hedonic regression, clustered by host, and call results *associations* | Observational data; no causal claims |
# | 12 | Segment markets with KMeans **without price** as an input, k chosen by silhouette with a minimum segment size | Price differences between segments are findings, and segments are big enough to matter |
# | 13 | Report a band, and verify its coverage on unseen hosts | Avoids false precision |
#
# ### Limitations (also in the report)
# * The price is **an asking price quoted for one stay**, not a transaction price or revenue; seasonality beyond weekend/weekday is not modelled.
# * Listings without a price (mostly not bookable) are excluded; luxury listings above the 99.5th percentile are out of scope.
# * Demand is observed only through **reviews** (a proxy); the price-gap → demand result is an association and could be driven by
#   unobserved quality (photos, view, floor).
# * Licence status, which matters legally in Hong Kong, is empty in the data, so we cannot separate licensed from unlicensed supply.
# * Straight-line distances ignore hills and walking routes; MTR data come from a volunteer-maintained map.
# * Review text analysis is limited to English-like comments and keyword topics.
# * Data are a single snapshot (25 Sep 2026); a production tool needs periodic retraining.

# %%
RESULTS["meta"] = dict(random_state=RANDOM_STATE, python_note="scikit-learn / statsmodels / matplotlib; see notebook header",
                       model_best=BEST, k_segments=K_SEG)
save_results()
print("Saved outputs/results.json and", len(list(FIG.glob('*.png'))), "figures in outputs/figures/")
print(pd.Series(RESULTS["models"]["table"]).apply(lambda r: f"MAE {r['MAE']:.0f}  MedAPE {r['MedAPE']:.1%}  within20 {r['Within20']:.0%}").to_string())
