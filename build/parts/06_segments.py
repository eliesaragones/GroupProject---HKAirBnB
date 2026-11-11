# %% [markdown]
# ## 8. Market segments – unsupervised learning (KMeans + PCA, Week 6)
#
# *Business question:* "Which kinds of listings really exist in this market, and do they behave differently on price?"
# If the market is made of a few distinct *segments*, a host should benchmark against their own segment, and an investor
# should choose a segment before choosing a district.
#
# *Design decisions.*
# * **Price is deliberately NOT an input.** Clustering on the characteristics of the offer (size, room type, stay policy,
#   centrality, operator type, amenities, local density) means that price differences between segments are a *finding*,
#   not something we put in.
# * Inputs are standardised (KMeans is distance-based) and skewed distances are log-transformed.
# * We choose the number of clusters *k* with the silhouette score, but only among **k = 4…7** and only if **every segment holds at
#   least 3% of listings**: the business needs enough segments to be actionable, but a segment of 28 listings is not a market.
# * PCA is used only to *draw* the clusters in two dimensions.

# %%
seg_in = pd.DataFrame({
    "accommodates": X_all.accommodates, "bedrooms": X_all.bedrooms.fillna(1), "baths": X_all.baths.fillna(1), "bath_shared": X_all.bath_shared,
    "is_entire": (X_all.room_type == "Entire home/apt").astype(int), "log_min_nights": X_all.log_min_nights,
    "log_dist_central": np.log1p(X_all.dist_central_km), "log_dist_mtr": np.log1p(X_all.dist_mtr_km),
    "host_listings": np.log1p(df.calculated_host_listings_count), "n_amenities": X_all.n_amenities, "listings_500m": X_all.listings_500m,
})
Z = StandardScaler().fit_transform(seg_in)
sil, min_share = {}, {}
for k in range(3, 9):
    km_ = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Z)
    sil[k] = silhouette_score(Z, km_.labels_, sample_size=3000, random_state=RANDOM_STATE)
    min_share[k] = np.bincount(km_.labels_).min() / len(Z)
# business constraint: every segment must be big enough to matter (>= 3% of listings)
eligible = [k for k in range(4, 8) if min_share[k] >= 0.03]
K_SEG = max(eligible, key=lambda k: sil[k])
print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
print("smallest segment share by k:", {k: round(v, 3) for k, v in min_share.items()}, "-> eligible", eligible, "-> chosen k =", K_SEG)
km = KMeans(n_clusters=K_SEG, n_init=20, random_state=RANDOM_STATE).fit(Z)
df["seg_id"] = km.labels_

prof = df.assign(entire=X_all.room_type.eq("Entire home/apt"), long=X_all.long_stay, acc=X_all.accommodates, bed=X_all.bedrooms,
                 dc=X_all.dist_central_km, shared=X_all.bath_shared, pro=X_all.pro_host, amen=X_all.n_amenities).groupby("seg_id").agg(
    n=("price_hkd", "size"), median_price=("price_hkd", "median"), entire=("entire", "mean"), long_stay=("long", "mean"),
    guests=("acc", "median"), bedrooms=("bed", "median"), dist_central=("dc", "median"), shared_bath=("shared", "mean"),
    pro_host=("pro", "mean"), amenities=("amen", "median"))

def label_segment(r):
    room = "Whole units" if r.entire > 0.6 else ("Private rooms" if r.entire < 0.25 else "Mixed units")
    stay = "monthly" if r.long_stay > 0.65 else ("short-stay" if r.long_stay < 0.35 else "mixed stay")
    mod = ""
    if r.guests >= 4: mod = ", groups"
    elif r.dist_central > 7: mod = ", outer areas"
    elif r.shared_bath > 0.5 and r.entire < 0.25: mod = ", shared bath"
    return f"{room}, {stay}{mod}"

prof["label"] = prof.apply(label_segment, axis=1)
dup = prof.label.duplicated(keep=False)
prof.loc[dup, "label"] = prof.loc[dup].apply(lambda r: r.label + f" (median {r.median_price:,.0f} HKD)", axis=1)
prof = prof.sort_values("median_price")
df["segment"] = df.seg_id.map(prof.label)
display(prof.round(2))

# --- figure 3: PCA map of segments + price distribution per segment ---
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
P = pca.transform(Z)
order = list(prof.index)
colors = {sid: CAT[i % len(CAT)] for i, sid in enumerate(order)}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.7), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = ax[0]
for sid in order:
    m = df.seg_id.values == sid
    a.scatter(P[m, 0], P[m, 1], s=5, alpha=0.5, color=colors[sid], linewidths=0)
for sid in order:
    m = df.seg_id.values == sid
    a.scatter(*np.median(P[m], axis=0), s=60, color=colors[sid], edgecolor="white", linewidth=1.5, zorder=3)
a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% of variance)"); a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
a.set_xlim(np.percentile(P[:, 0], [0.5, 99.5])); a.set_ylim(np.percentile(P[:, 1], [0.5, 99.5]))
a.set_title(f"{K_SEG} market segments (PCA view)", fontsize=8.2)
b = ax[1]
data = [df.loc[df.seg_id == sid, "price_hkd"].values for sid in order]
bp = b.boxplot(data, vert=False, patch_artist=True, widths=0.55, showfliers=False, whis=(10, 90),
               medianprops=dict(color=PAL["ink"], lw=1.4), whiskerprops=dict(color=PAL["muted"]), capprops=dict(color=PAL["muted"]))
for patch, sid in zip(bp["boxes"], order):
    patch.set_facecolor(colors[sid]); patch.set_edgecolor(PAL["surface"]); patch.set_alpha(0.95)
b.set_xscale("log"); b.set_xticks([100, 200, 500, 1000, 2000, 5000]); b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
short = {sid: textwrap.fill(prof.loc[sid, "label"], 26) for sid in order}
b.set_yticklabels([short[s] + f"\n(n={int(prof.loc[s,'n']):,})" for s in order], fontsize=6.6)
b.set_xlabel("Price per night, HKD (log scale)"); b.grid(axis="y", visible=False)
b.set_title("Segments differ strongly in price (box: middle 50%)", fontsize=8.2)
fig.tight_layout(w_pad=1.2)
savefig("fig3_segments", fig)

RESULTS["segments"] = dict(k=K_SEG, silhouette={int(k): float(v) for k, v in sil.items()},
                           profile=[dict(label=r.label, n=int(r.n), share=float(r.n / len(df)), median_price=float(r.median_price), entire=float(r.entire),
                                         long_stay=float(r.long_stay), guests=float(r.guests), bedrooms=float(r.bedrooms),
                                         dist_central=float(r.dist_central), pro_host=float(r.pro_host)) for _, r in prof.iterrows()],
                           pca_var=[float(v) for v in pca.explained_variance_ratio_])
