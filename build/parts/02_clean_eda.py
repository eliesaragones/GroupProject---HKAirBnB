# %% [markdown]
# ## 3. Data preparation – every removal is a documented business decision
# We only remove a listing when we can argue that it does not belong to the question *"what is a fair asking price for a bookable listing?"*.
# Every removal is counted, so that the report can show how much data each decision costs.

# %%
steps = [("Listings in the detailed file", len(L), 0)]           # log: (description, listings left, listings removed)
d = L.copy()

# 1) Listings without a price: first check why (most of them cannot be booked)
no_price = d.price_hkd.isna()
share_unbookable = (d.loc[no_price, "availability_365"] == 0).mean()
print(f"{no_price.sum():,} listings have no price; {pct(share_unbookable)} of them cannot be booked in the next year (availability_365 == 0)")
d = d[~no_price]                                                  # "~" means "not": keep the listings that have a price
steps.append(("Drop listings without a price quote (mostly not bookable)", len(d), int(no_price.sum())))

# 2) Listings without a minimum-nights policy (we could not tell whether they are short or monthly stays)
no_nights = d.minimum_nights.isna()
d = d[~no_nights]
steps.append(("Drop listings without minimum-nights policy", len(d), int(no_nights.sum())))

# 3) Implausible prices. Floor: a few HKD per night is a data error (e.g. a 0.73 HKD "tent" with a 300-night quote).
#    Cap: the 99.5th percentile; a handful of 20,000-150,000 HKD listings would dominate the errors and are not the typical host's problem.
PRICE_FLOOR, PRICE_CAP = 50.0, float(d.price_hkd.quantile(0.995))
too_low, too_high = d.price_hkd < PRICE_FLOOR, d.price_hkd > PRICE_CAP
d = d[~(too_low | too_high)]
steps.append((f"Drop implausible prices (< {PRICE_FLOOR:.0f} or > {PRICE_CAP:,.0f} HKD)", len(d), int(too_low.sum() + too_high.sum())))

# The cleaned table is called df (its row numbers are those of L, so that features computed on L can be matched later)
df = d.copy()
df["regime"] = np.where(df.minimum_nights >= 28, "Long stay (28+ nights)", "Short stay (<28 nights)")
df["long_stay"] = (df.minimum_nights >= 28).astype(int)
display(pd.DataFrame(steps, columns=["Step", "Listings left", "Removed"]))
print(f"Modelling sample: {len(df):,} listings from {df.host_id.nunique():,} hosts ({pct(len(df)/len(L))} of the file); median price {df.price_hkd.median():.0f} HKD")
RESULTS["cleaning"] = dict(steps=[dict(step=a, left=int(b), removed=int(c)) for a, b, c in steps], price_floor=PRICE_FLOOR, price_cap=PRICE_CAP,
                           share_unbookable_among_missing=float(share_unbookable), n_final=len(df), n_hosts=int(df.host_id.nunique()),
                           median_price=float(df.price_hkd.median()), mean_price=float(df.price_hkd.mean()))

# %% [markdown]
# ## 4. Understanding the market before modelling
# Three basic facts shape the rest of the analysis: where supply is concentrated, how many listings are in practice monthly rentals, and how much cheaper a long stay is per night.

# %%
# (a) listings per district and room type, (b) share of listings by minimum nights, (c) median price of short vs. monthly stays
rt3 = L.room_type.where(L.room_type.isin(["Private room", "Entire home/apt"]), "Hotel / shared room")
ct = pd.crosstab(L.neighbourhood_cleansed, rt3)[["Private room", "Entire home/apt", "Hotel / shared room"]]
by_district = ct.sum(axis=1).sort_values(ascending=False)
top2_share = by_district.head(2).sum() / len(L)
labels = ["1", "2", "3-6", "7-27", "28-31", "32+"]
mn_share = pd.cut(L.minimum_nights.dropna(), [0, 1, 2, 6, 27, 31, 10_000], labels=labels).value_counts(normalize=True).reindex(labels)
long_share = float((L.minimum_nights >= 28).mean())
med = df[df.room_type.isin(["Private room", "Entire home/apt"])].groupby(["room_type", "regime"]).price_hkd.median().unstack()
disc = 1 - med["Long stay (28+ nights)"] / med["Short stay (<28 nights)"]          # discount of a monthly stay

# Figure 1
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9))
ct.loc[by_district.head(8).index].iloc[::-1].plot.barh(stacked=True, ax=ax[0], color=[PAL["blue"], PAL["orange"], PAL["aqua"]], width=0.7)
ax[0].set_title(f"Two districts hold {pct(top2_share)} of listings", fontsize=8); ax[0].set_ylabel(""); ax[0].legend(fontsize=6, loc="lower right")
ax[1].bar(labels, mn_share.values, color=[PAL["blue"]] * 4 + [PAL["orange"]] * 2)
ax[1].set_title(f"{pct(long_share)} require 28+ nights", fontsize=8); ax[1].set_xlabel("Minimum nights required"); ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
med.T.plot.bar(ax=ax[2], color=[PAL["blue"], PAL["orange"]], width=0.75, rot=0)
ax[2].set_title(f"A monthly stay costs {pct(disc.min())}-{pct(disc.max())} less", fontsize=8); ax[2].set_xlabel(""); ax[2].set_ylabel("Median price per night (HKD)")
ax[2].set_xticklabels(["Short stay", "Long stay"]); ax[2].legend(fontsize=6)
fig.tight_layout()
savefig("fig1_market_overview", fig)

host_pro = float((L.calculated_host_listings_count >= 5).mean())   # share of listings that belong to hosts with 5+ listings
RESULTS["eda"] = dict(
    top2_share=float(top2_share), top2_names=list(by_district.head(2).index), long_stay_share=long_share,
    private_room_share=float((L.room_type == "Private room").mean()), entire_share=float((L.room_type == "Entire home/apt").mean()),
    median_price={f"{r}|{g}": float(med.loc[r, g]) for r in med.index for g in med.columns}, long_stay_discount={r: float(disc[r]) for r in disc.index},
    share_listings_from_hosts_5plus=host_pro, max_host_listings=int(L.calculated_host_listings_count.max()), n_hosts=int(L.host_id.nunique()),
    price_skew_raw=float(df.price_hkd.skew()), price_skew_log=float(np.log(df.price_hkd).skew()))
print(f"Hosts with 5+ listings own {pct(host_pro)} of all listings (largest host: {L.calculated_host_listings_count.max()}). "
      f"Skewness of the price, raw -> log: {df.price_hkd.skew():.1f} -> {np.log(df.price_hkd).skew():.2f}")

# %% [markdown]
# **Findings.**
# * The market is **two markets**: short stays (typically tourists; the median nightly price is about 1.7× that of a monthly stay for whole homes and 3× for private rooms)
#   and monthly rentals. *Stay policy is a first-class factor and the main segmentation of our analysis.*
# * Supply is dominated by **private rooms** and by two districts, so the benchmark must be especially good there and honest about thin cells (outlying districts).
# * Professional hosts (5+ listings) own most of the supply. This is a **leakage risk**: sister listings of one host look almost identical, so a random train/test split would
#   overstate accuracy. We therefore split **by host** (section 8).
# * Prices are very right-skewed (skewness ≈ 3, ≈ 0.4 after a log transform). We model **log-price**: errors become *percentage* errors, which is how a host thinks.
