# %% [markdown]
# ## 14. Summary, decisions and limitations
# **The story in five sentences**
# 1. Hong Kong rentals are **two markets**, tourist stays and monthly rentals; stay length is the master variable of the price.
# 2. A machine-learning benchmark predicts a fair price clearly better than "look at comparable listings", on hosts it has never seen.
# 3. **Listing type, stay type and the operator** explain most of the price; **distance to the MTR does not** differentiate prices on its own.
# 4. Listings priced **far above** their benchmark get **fewer reviews**; discounts do not clearly buy more.
# 5. Every price comes with a **band** and a **"why this price?"** explanation whose numbers are computed by us and checked.
#
# **Decisions we made (and why)**
# | Decision | Reason |
# |---|---|
# | Drop listings without a price; remove prices < 50 HKD or above the 99.5th percentile | A benchmark is about bookable listings; data errors and a few luxury outliers would dominate the errors |
# | Treat **stay length as a core factor** and a segmentation (≥ 28 nights) | The price is a quote for a specific stay; monthly stays cost a fraction per night |
# | Add quote-context controls (nights, weekend check-in) but **not the lead time** | Lead time partly reflects demand (leakage) |
# | Model `log(price)` | Percentage errors, tamed skewness |
# | Split and cross-validate **by host**; test set used once; models chosen by cross-validation | Prevents memorising professional hosts; honest estimate |
# | Compare with the **comparable-listings baseline** | The business question is "better than what I do now?" |
# | Two host flags only; no ratings or reviews in the launch-time model | Richer host variables did not help on unseen hosts; a new host has no reviews |
# | Explain with permutation importance on unseen hosts and say *association* | Observational data, no causal claims |
# | Segment with KMeans **without price**, minimum segment size | Price differences between segments are findings; segments big enough to matter |
# | Report a band and verify its coverage; no amenity "uplift" ranking | Avoids false precision; amenities mostly mark the listing type |
# | Keep the code simple: basic pandas / NumPy / scikit-learn tools and SHAP for the explanations | The analysis must be readable and reproducible by the whole group |
#
# **Limitations**
# * The price is an **asking price quoted for one stay** (not a transaction price or revenue); seasonality beyond weekday/weekend is not modelled; listings without a price (mostly not
#   bookable) and luxury listings above the 99.5th percentile are out of scope.
# * Demand is observed only through **reviews** (a proxy); the price-gap → demand result is an association and could be driven by unobserved quality (photos, view, floor).
# * Licence status is empty in the data; straight-line distances ignore hills and walking routes; the MTR data come from a volunteer-maintained map; the data are a single snapshot (25 Sep 2026).

# %%
RESULTS["meta"] = dict(random_state=SEED, model_best=BEST, k_segments=K_SEG)
save_results()
print("Saved outputs/results.json and", len(list(FIG.glob("*.png"))), "figures in outputs/figures/")
