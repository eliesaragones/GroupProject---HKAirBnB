# %% [markdown]
# ## 3. Data preparation – every removal is a documented business decision
#
# We only remove listings when we can argue that they do not belong to the question *"what is a fair asking price for a
# bookable listing?"*. Each step is counted so the report can show how much data each decision costs.

# %%
steps = [("Listings in the detailed file", len(L), 0)]
d = L.copy()

# (1) No price. Check WHY before dropping: two thirds are un-bookable; the rest simply have no quote for the scrape window.
no_price = d.price_hkd.isna()
share_unbookable = (d.loc[no_price, "availability_365"] == 0).mean()
print(f"{no_price.sum():,} listings have no price; {pct(share_unbookable)} of them have availability_365 == 0 (cannot be booked in the next year)")
d = d[~no_price]
steps.append(("Drop listings without a price quote (mostly not bookable)", len(d), int(no_price.sum())))

# (2) Missing stay policy (cannot be assigned to a stay regime)
miss_min = d.minimum_nights.isna()
d = d[~miss_min]
steps.append(("Drop listings without minimum-nights policy", len(d), int(miss_min.sum())))

# (3) Implausible prices. Floor: a few HKD per night (e.g. a 0.73 HKD 'tent' with a 300-night quote) are data errors.
#     Cap: the 99.5th percentile. The model is meant to benchmark *typical* listings; a handful of 20k-150k HKD/night luxury or
#     erroneous listings would dominate squared-error metrics and are not the host's pricing problem.
PRICE_FLOOR = 50.0
PRICE_CAP = float(d.price_hkd.quantile(0.995))
bad = (d.price_hkd < PRICE_FLOOR) | (d.price_hkd > PRICE_CAP)
n_low, n_high = int((d.price_hkd < PRICE_FLOOR).sum()), int((d.price_hkd > PRICE_CAP).sum())
d = d[~bad]
steps.append((f"Drop implausible prices (< {PRICE_FLOOR:.0f} or > {PRICE_CAP:,.0f} HKD)", len(d), n_low + n_high))

df = d.copy()                      # NOTE: original index is kept so features computed on the full file can be aligned
df["regime"] = np.where(df.minimum_nights >= 28, "Long stay (28+ nights)", "Short stay (<28 nights)")
df["long_stay"] = (df.minimum_nights >= 28).astype(int)

clean_tbl = pd.DataFrame(steps, columns=["Step", "Listings left", "Removed"])
display(clean_tbl)
print(f"Final modelling sample: {len(df):,} listings from {df.host_id.nunique():,} hosts "
      f"({pct(len(df)/len(L))} of the file). Price range {df.price_hkd.min():.0f}-{df.price_hkd.max():,.0f} HKD, median {df.price_hkd.median():.0f}")
RESULTS["cleaning"] = dict(steps=[dict(step=a, left=int(b), removed=int(c)) for a, b, c in steps],
                           price_floor=PRICE_FLOOR, price_cap=PRICE_CAP, share_unbookable_among_missing=float(share_unbookable),
                           n_final=len(df), n_hosts=int(df.host_id.nunique()), median_price=float(df.price_hkd.median()),
                           mean_price=float(df.price_hkd.mean()))

# %% [markdown]
# ## 4. Understanding the market before modelling
# Four facts shape everything that follows: where supply is concentrated, how many listings are really monthly rentals, how
# much cheaper a long stay is per night, and how guest demand has evolved.

# %%
rt3 = L.room_type.where(L.room_type.isin(["Private room", "Entire home/apt"]), "Hotel / shared room")
ct = pd.crosstab(L.neighbourhood_cleansed, rt3)[["Private room", "Entire home/apt", "Hotel / shared room"]]
top8 = ct.sum(axis=1).sort_values(ascending=False).head(8).index
ct8 = ct.loc[top8].iloc[::-1]
top2_share = ct.sum(axis=1).sort_values(ascending=False).head(2).sum() / len(L)

bins = [0, 1, 2, 6, 27, 31, 10_000]
labels = ["1", "2", "3-6", "7-27", "28-31", "32+"]
mn = pd.cut(L.minimum_nights.dropna(), bins=bins, labels=labels)
mn_share = mn.value_counts(normalize=True).reindex(labels)
long_share = float((L.minimum_nights >= 28).mean())

med = df[df.room_type.isin(["Private room", "Entire home/apt"])].groupby(["room_type", "regime"]).price_hkd.median().unstack()
disc = 1 - med["Long stay (28+ nights)"] / med["Short stay (<28 nights)"]

yr = R.date.dt.year.value_counts().sort_index()
yr = yr[yr.index >= 2013]

fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.5))
# (a) supply by district and room type
a = ax[0, 0]
left = np.zeros(len(ct8))
for col, c in zip(ct8.columns, [PAL["blue"], PAL["orange"], PAL["aqua"]]):
    a.barh(ct8.index, ct8[col], left=left, color=c, label=col, height=0.68, edgecolor=PAL["surface"], linewidth=1.2)
    left += ct8[col].values
a.set_title(f"Supply is concentrated: two districts hold {pct(top2_share)} of listings", fontsize=8.2)
a.set_xlabel("Number of listings"); a.grid(axis="y", visible=False)
a.legend(loc="lower right", fontsize=7)
# (b) minimum nights
b = ax[0, 1]
cols_b = [PAL["blue"]] * 4 + [PAL["orange"]] * 2
b.bar(labels, mn_share.values, color=cols_b, width=0.68)
for i, v in enumerate(mn_share.values):
    b.text(i, v + 0.008, pct(v), ha="center", fontsize=7.5, color=PAL["ink2"])
b.set_title(f"{pct(long_share)} of listings are in practice monthly rentals", fontsize=8.2)
b.set_xlabel("Minimum nights required by the host"); b.set_ylabel("Share of listings"); b.set_ylim(0, mn_share.max() * 1.3)
b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0)); b.grid(axis="x", visible=False)
b.text(4.5, mn_share.max() * 1.2, "28+ nights", ha="center", color=PAL["orange"], fontsize=7.5, fontweight="bold")
b.text(1.5, mn_share.max() * 1.2, "< 28 nights", ha="center", color=PAL["blue"], fontsize=7.5, fontweight="bold")
# (c) median price by regime
c_ = ax[1, 0]
x = np.arange(2); w = 0.36
for j, (reg, col) in enumerate([("Short stay (<28 nights)", PAL["blue"]), ("Long stay (28+ nights)", PAL["orange"])]):
    vals = [med.loc[r, reg] for r in ["Private room", "Entire home/apt"]]
    c_.bar(x + (j - 0.5) * w, vals, w * 0.92, color=col, label=reg)
    for xi, v in zip(x + (j - 0.5) * w, vals):
        c_.text(xi, v + 12, f"{v:,.0f}", ha="center", fontsize=7.5, color=PAL["ink2"])
c_.set_xticks(x); c_.set_xticklabels(["Private room", "Entire home/apt"])
c_.set_ylabel("Median price per night (HKD)"); c_.legend(loc="upper left", fontsize=7); c_.grid(axis="x", visible=False)
c_.set_ylim(0, med.values.max() * 1.25)
c_.set_title(f"A monthly stay costs {pct(disc.min())}-{pct(disc.max())} less per night", fontsize=8.2)
# (d) reviews per year
d_ = ax[1, 1]
full = yr[yr.index < R.date.max().year]
d_.plot(full.index, full.values, color=PAL["blue"], lw=2, marker="o", ms=3.5)
d_.plot([full.index[-1], yr.index[-1]], [full.values[-1], yr.values[-1]], color=PAL["blue"], lw=1.5, ls="--")
d_.plot(yr.index[-1], yr.values[-1], marker="o", ms=5, mfc="white", mec=PAL["blue"])
d_.text(yr.index[-1], yr.values[-1] * 0.8, "2026 (to Sep)", ha="right", fontsize=7)
d_.annotate("COVID-19", xy=(2020.8, yr.loc[2021]), xytext=(2019.3, 21500), fontsize=7.5, color=PAL["ink2"],
            arrowprops=dict(arrowstyle="-", color=PAL["muted"], lw=0.8))
d_.set_title("Guest demand proxy: reviews are back above 2019", fontsize=8.2)
d_.set_ylabel("Reviews posted per year"); d_.set_xlabel("")
d_.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: f"{v/1000:.0f}k" if v else "0"))
fig.tight_layout(h_pad=1.6, w_pad=1.4)
savefig("fig1_market_overview", fig)

host_pro = float((L.calculated_host_listings_count >= 5).mean())
RESULTS["eda"] = dict(
    top2_share=float(top2_share), top2_names=list(ct.sum(axis=1).sort_values(ascending=False).head(2).index),
    long_stay_share=long_share, private_room_share=float((L.room_type == "Private room").mean()),
    entire_share=float((L.room_type == "Entire home/apt").mean()),
    median_price={f"{r}|{g}": float(med.loc[r, g]) for r in med.index for g in med.columns},
    long_stay_discount={r: float(disc[r]) for r in disc.index},
    share_listings_from_hosts_5plus=host_pro, max_host_listings=int(L.calculated_host_listings_count.max()),
    n_hosts=int(L.host_id.nunique()), reviews_2019=int(yr.loc[2019]), reviews_2025=int(yr.loc[2025]),
    reviews_2021=int(yr.loc[2021]), superhost_share=float((L.host_is_superhost == "t").mean()),
    price_skew_raw=float(df.price_hkd.skew()), price_skew_log=float(np.log(df.price_hkd).skew()),
)
print(f"Hosts with 5+ listings own {pct(host_pro)} of all listings (largest host: {L.calculated_host_listings_count.max()} listings)")
print("Price skewness raw -> log:", round(df.price_hkd.skew(), 1), "->", round(np.log(df.price_hkd).skew(), 2))

# %% [markdown]
# **Business reading.**
# * The market is **two markets**: short stays (typically tourists; the median nightly price is about 1.7× that of a monthly stay for whole homes and
#   3× for private rooms) and monthly rentals. A single "average price" would be wrong for both; *stay policy is a first-class
#   factor and the main segmentation of our analysis.*
# * Supply is dominated by **private rooms in rental flats** and by two districts, so the benchmark must be especially good
#   in those cells – and honest about thin cells (outlying districts).
# * Professional hosts (5+ listings) own most of the supply. This is a **leakage risk** for machine learning: sister listings
#   of one host in the same building look almost identical, so a random train/test split would overstate accuracy. We therefore
#   split **by host** (section 9).
# * Prices are extremely right-skewed (skewness ≈ 3 → ≈ 0.4 after a log transform). We model **log-price**, which turns the
#   error into a *percentage* error – the natural way a host thinks ("I am 20% above market").
