# %% [markdown]
# ## 7. Factors that only exist after the first guests: ratings and the "voice of the guest"
#
# Reviews are a second, richer data source (122k comments). They cannot be inputs to the *new-listing* benchmark (a new host
# has none) but they tell us (a) whether a listing priced above or below benchmark is **also perceived differently by
# guests**, and (b) whether an *existing* listing's price can be explained better once reputation is known.
#
# *Data-quality decisions for the text:*
# * Reviews are multilingual. We keep **English-like comments** (≥ 97% ASCII characters, ≥ 20 characters) for lexicon-based
#   analysis; the share we keep is reported, not hidden.
# * Sentiment uses **VADER** (a lexicon-based scorer designed for short reviews) – transparent and requires no training data.
# * Topic flags are regular expressions for what matters in Hong Kong's small-flat market: *location/transit, cleanliness,
#   dirt/smell, noise, small size, value for money, "overpriced" complaints and host service.*
# * A listing gets text features only if it has **≥ 3 English reviews** (otherwise the share would be noise).

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TOPICS = {
    "gv_location": r"location|located|\bwalk|\bmtr\b|station|convenient|close to|\bnear\b",
    "gv_clean": r"\bclean|spotless|\btidy\b|well[- ]kept",
    "gv_dirty": r"\bdirty|smell|\bmold|\bmould|cockroach|\bbugs?\b|stain|filthy|\bdust",
    "gv_noise": r"\bnois|\bloud\b|thin wall",
    "gv_small": r"\bsmall\b|\btiny\b|cramped|narrow|\bcompact\b|little space|shoebox",
    "gv_value": r"great value|good value|worth|affordable|\bcheap\b|bargain|value for money",
    "gv_pricey": r"expensive|overpriced|pricey|not worth|rip[- ]?off|too much",
    "gv_host": r"helpful|responsive|friendly|attentive|great host|communicat",
}
cache_rev = OUT / "review_features.csv"
if cache_rev.exists():
    GV = pd.read_csv(cache_rev, index_col=0)
else:
    t0 = time.time()
    R2 = R.dropna(subset=["comments"]).copy()
    R2["txt"] = R2.comments.str.replace(r"<[^>]+>", " ", regex=True)
    ascii_share = R2.txt.apply(lambda s: sum(ord(ch) < 128 for ch in s) / max(len(s), 1))
    R2["english"] = (ascii_share >= 0.97) & (R2.txt.str.len() >= 20)
    E = R2[R2.english].copy()
    low = E.txt.str.lower()
    for c, rx in TOPICS.items():
        E[c] = low.str.contains(rx, regex=True).astype(int)
    sia = SentimentIntensityAnalyzer()
    E["gv_sentiment"] = E.txt.apply(lambda s: sia.polarity_scores(s)["compound"])
    agg = E.groupby("listing_id").agg(n_en=("txt", "size"), gv_sentiment=("gv_sentiment", "mean"), **{c: (c, "mean") for c in TOPICS})
    agg = agg[agg.n_en >= 3]
    GV = agg
    GV.to_csv(cache_rev)
    print(f"processed {len(R2):,} reviews in {time.time()-t0:.0f}s")
GV_COLS = ["gv_sentiment"] + list(TOPICS)
R_en_share = float((R.comments.fillna("").apply(lambda s: sum(ord(ch) < 128 for ch in s) / max(len(s), 1)) >= 0.97).mean())
print(f"English-like share of review comments: {pct(R_en_share)}; listings with >=3 English reviews: {len(GV):,}")

def experience(Lsrc):
    """'Existing listing' factors: ratings, review activity and guest-voice text features."""
    E_ = pd.DataFrame(index=Lsrc.index)
    for c in ["accuracy", "cleanliness", "checkin", "communication", "location", "value", "rating"]:
        E_[f"rs_{c}"] = Lsrc[f"review_scores_{c}"]
    E_["log_n_reviews"] = np.log1p(Lsrc.number_of_reviews)
    E_["reviews_ltm"] = Lsrc.number_of_reviews_ltm
    E_["reviews_per_month"] = Lsrc.reviews_per_month
    E_["days_since_last_review"] = (SCRAPE_DATE - Lsrc.last_review).dt.days
    E_["superhost"] = (Lsrc.host_is_superhost == "t").astype(int)
    E_["has_reviews"] = (Lsrc.number_of_reviews > 0).astype(int)
    gv = GV.reindex(Lsrc.id.values)
    gv.index = Lsrc.index
    for c in GV_COLS:
        E_[c] = gv[c]
    return E_

EXP_FAMILIES = {
    "Reviews & ratings": [f"rs_{c}" for c in ["accuracy", "cleanliness", "checkin", "communication", "location", "value", "rating"]]
                         + ["log_n_reviews", "reviews_ltm", "reviews_per_month", "days_since_last_review", "superhost", "has_reviews"],
    "Guest voice (review text)": GV_COLS,
}
EXP_ALL = experience(df)
EXP_COLS = [c for v in EXP_FAMILIES.values() for c in v]
print(f"Listings with rating data: {pct(EXP_ALL.rs_rating.notna().mean())};  with guest-voice text features: {pct(EXP_ALL.gv_sentiment.notna().mean())}")
RESULTS["reviews"] = dict(share_english=R_en_share, n_listings_gv=int(len(GV)), share_with_rating=float(EXP_ALL.rs_rating.notna().mean()),
                          share_with_gv=float(EXP_ALL.gv_sentiment.notna().mean()), n_reviews=int(len(R)))
