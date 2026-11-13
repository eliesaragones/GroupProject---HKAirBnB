# %% [markdown]
# ## 7. Market segments – unsupervised learning (KMeans + PCA, Week 6)
# *Business question:* which kinds of listings really exist in this market, and do they behave differently on price? If there are a few distinct *segments*, a host should
# benchmark against their own segment. **Price is deliberately NOT an input**: we cluster on the characteristics of the offer (size, room type, stay policy, centrality, operator
# type, amenities, local density), so that price differences between segments are a *finding*. The inputs are standardised (KMeans is distance-based). We choose the number of
# clusters *k* with the silhouette score, but only among **k = 4…7** and only if **every segment holds at least 3% of the listings** (a segment of 28 listings is not a market).

# %%
seg_in = pd.DataFrame({
    "accommodates": X_all.accommodates, "bedrooms": X_all.bedrooms.fillna(1), "baths": X_all.baths.fillna(1), "bath_shared": X_all.bath_shared,
    "is_entire": (X_all.room_type == "Entire home/apt").astype(int), "log_min_nights": X_all.log_min_nights,
    "log_dist_central": np.log1p(X_all.dist_central_km), "log_dist_mtr": np.log1p(X_all.dist_mtr_km),
    "host_listings": np.log1p(df.calculated_host_listings_count), "n_amenities": X_all.n_amenities, "listings_500m": X_all.listings_500m})
Z = StandardScaler().fit_transform(seg_in)                          # every column: mean 0, standard deviation 1

# Which k? Silhouette score (higher = better separated groups) and share of the smallest group, for k = 3..8
sil, smallest = {}, {}
for k in range(3, 9):
    labels_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z).labels_
    sil[k] = silhouette_score(Z, labels_k, sample_size=3000, random_state=SEED)
    smallest[k] = np.bincount(labels_k).min() / len(Z)
K_SEG = max([k for k in range(4, 8) if smallest[k] >= 0.03], key=lambda k: sil[k])     # best silhouette among the eligible k
print("silhouette:", {k: round(v, 3) for k, v in sil.items()}, "-> chosen k =", K_SEG)
df["seg_id"] = KMeans(n_clusters=K_SEG, n_init=20, random_state=SEED).fit(Z).labels_

# Describe each segment (one row per segment) and name it from its typical characteristics
prof = df.assign(entire=X_all.room_type.eq("Entire home/apt"), long=X_all.long_stay, acc=X_all.accommodates, bed=X_all.bedrooms,
                 dc=X_all.dist_central_km, shared=X_all.bath_shared, pro=X_all.pro_host).groupby("seg_id").agg(
    n=("price_hkd", "size"), median_price=("price_hkd", "median"), entire=("entire", "mean"), long_stay=("long", "mean"), guests=("acc", "median"),
    bedrooms=("bed", "median"), dist_central=("dc", "median"), shared_bath=("shared", "mean"), pro_host=("pro", "mean"))

def label_segment(r):
    room = "Whole units" if r.entire > 0.6 else ("Private rooms" if r.entire < 0.25 else "Mixed units")
    stay = "monthly" if r.long_stay > 0.65 else ("short-stay" if r.long_stay < 0.35 else "mixed stay")
    extra = ", groups" if r.guests >= 4 else (", outer areas" if r.dist_central > 7 else "")
    return f"{room}, {stay}{extra}"

prof["label"] = prof.apply(label_segment, axis=1)
prof = prof.sort_values("median_price")
df["segment"] = df.seg_id.map(prof.label)
display(prof.round(2))

# Figure 3: left = segments seen through PCA (11 columns reduced to 2), right = price distribution of each segment
P = PCA(n_components=2, random_state=SEED).fit_transform(Z)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
for i, sid in enumerate(prof.index):
    m = df.seg_id.values == sid
    ax[0].scatter(P[m, 0], P[m, 1], s=5, alpha=0.5, color=CAT[i], linewidths=0)
ax[0].set(xlim=np.percentile(P[:, 0], [0.5, 99.5]), ylim=np.percentile(P[:, 1], [0.5, 99.5]), xlabel="PC1", ylabel="PC2", title=f"{K_SEG} market segments (PCA view)")
bp = ax[1].boxplot([df.loc[df.seg_id == sid, "price_hkd"] for sid in prof.index], vert=False, patch_artist=True, showfliers=False, whis=(10, 90),
                   medianprops=dict(color=PAL["ink"]))
for patch, color in zip(bp["boxes"], CAT):
    patch.set_facecolor(color)
ax[1].set_xscale("log"); ax[1].set_xticks([100, 200, 500, 1000, 2000, 5000]); ax[1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax[1].set_yticklabels([textwrap.fill(prof.loc[s, "label"], 26) + f"\n(n={int(prof.loc[s, 'n']):,})" for s in prof.index], fontsize=6.6)
ax[1].set(xlabel="Price per night, HKD (log scale)", title="Segments differ strongly in price")
fig.tight_layout()
savefig("fig3_segments", fig)
RESULTS["segments"] = dict(k=K_SEG, silhouette={int(k): float(v) for k, v in sil.items()},
                           profile=[dict(label=r.label, n=int(r.n), share=float(r.n / len(df)), median_price=float(r.median_price), entire=float(r.entire),
                                         long_stay=float(r.long_stay), guests=float(r.guests), bedrooms=float(r.bedrooms),
                                         dist_central=float(r.dist_central), pro_host=float(r.pro_host)) for _, r in prof.iterrows()])

# %% [markdown]
# **Finding.** The market splits into a few clearly different segments whose **median prices differ by almost an order of magnitude**, although price was *not* used to build them.
# The biggest segment is single-guest rooms rented by the month, mostly by professional hosts; whole flats for groups sit at the other end. Practical rule: **compare a listing only with its own segment**.
