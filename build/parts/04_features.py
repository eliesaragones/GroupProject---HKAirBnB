# %% [markdown]
# ## 6. Useful factors: what could explain a listing's price, and why?
#
# We organise candidate factors into **families**, each tied to a business rationale. Grouping them matters: later we measure
# how much each *family* adds (ablation) and how much accuracy is lost when it is scrambled (permutation importance), which is
# a far more useful business statement than "feature 37 has importance 0.02".
#
# | Family | Factors | Business rationale (hedonic pricing: a listing is a bundle of attributes guests value) |
# |---|---|---|
# | **Property & capacity** | room type, property type, guests, bedrooms, beds, baths, shared bathroom | Space and privacy are what guests buy |
# | **Stay policy & quote** | long-stay (≥ 28 nights), log minimum nights, nights in the quoted stay, weekend check-in | Determines the customer segment (tourists vs. monthly renters). The scraped price is a *quote for a specific stay*, and Fri/Sat check-ins and longer stays are priced differently; the quote variables are **measurement controls**, set to a standard stay when benchmarking |
# | **Location** | district, region, latitude/longitude | Neighbourhood prestige and demand |
# | **Transit & centrality** | distance to nearest MTR, MTR stations within 1 km, distance to Central / TST / nearest of 4 hubs | Hong Kong is a transit city |
# | **Local competition** | listings within 500 m and 1 km | Dense areas = more substitutes = price pressure (or a proven tourist area) |
# | **Amenities** | 30 most common amenity flags + total count | Convenience features guests trade money for |
# | **Host profile** | professional host (5+ listings), host lives in Hong Kong | Professional operators price differently from occasional hosts. *Decision:* we first built six host variables (portfolio size, tenure, verification, profile text…) but kept only these two flags: in host-grouped cross-validation the richer set did **not** improve accuracy (typical error 22.8% vs 22.2% without), because continuous host-history variables let a tree recognise individual professional hosts, which does not generalise to hosts it has never seen |
# | **Listing text** | keyword flags in title/description, description length | How the listing is *marketed* (luxury, view, MTR…) |
#
# *Deliberately excluded:* the quote's **lead time** (days until the quoted check-in) – the next free date is partly an *outcome* of demand (busy listings are booked up), so using it would leak demand into price.
#
# Two further families exist only **after** a listing has guests – `Reviews & ratings` and `Guest voice` (what guests write).
# A new host does not have them, so they are **excluded from the main "new-listing benchmark"** and used only for a second
# model for *existing* listings (section 14). This separation is deliberate: it keeps the benchmark honest for its main user.

# %%
# --- grouping decisions (made on the modelling sample only, using counts - never the price) ---
ptype_counts = df.property_type.value_counts()
TOP_PTYPES = ptype_counts[ptype_counts >= 50].index.tolist()
dist_counts = df.neighbourhood_cleansed.value_counts()
KEEP_DISTRICTS = dist_counts[dist_counts >= 60].index.tolist()
REGION = {**{d: "Hong Kong Island" for d in ["Central & Western", "Wan Chai", "Eastern", "Southern"]},
          **{d: "Kowloon" for d in ["Yau Tsim Mong", "Sham Shui Po", "Kowloon City", "Wong Tai Sin", "Kwun Tong"]}}
print(f"{len(TOP_PTYPES)} property types kept ({pct(df.property_type.isin(TOP_PTYPES).mean())} of listings), rest -> 'Other'")
print(f"{len(KEEP_DISTRICTS)} districts kept ({pct(df.neighbourhood_cleansed.isin(KEEP_DISTRICTS).mean())} of listings), rest -> 'Other districts'")

# --- amenities: parse the JSON list and flag the most common ones ---
def parse_list(s):
    try: return json.loads(s)
    except Exception: return []

amen_lists = L.amenities.apply(parse_list)
amen_prev = pd.Series([a for l in amen_lists.loc[df.index] for a in l]).value_counts() / len(df)
AMEN = amen_prev[(amen_prev >= 0.05) & (amen_prev <= 0.97)].head(30).index.tolist()
slug = lambda s: "am_" + re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
AMEN_COLS = {a: slug(a) for a in AMEN}
print(f"{len(amen_prev)} distinct amenities; {len(AMEN)} flagged (prevalence 5-97%): {AMEN[:8]} ...")

TEXT_FLAGS = {  # marketing keywords in title + description
    "tx_transit": r"\bmtr\b|\bstation\b|\bsubway\b|\bmetro\b",
    "tx_luxury": r"luxur|premium|high-end|deluxe|boutique",
    "tx_view": r"\bview|harbou?r|\bsea\b|skyline|rooftop",
    "tx_studio": r"\bstudio\b",
    "tx_renovated": r"brand new|newly|renovat|modern",
    "tx_family": r"\bfamily|\bkids\b|\bchildren\b",
    "tx_monthly": r"monthly|long[- ]?term|per month|\bmonth\b",
    "tx_hostel": r"hostel|dorm|bunk|capsule",
    "tx_tiny": r"\btiny\b|\bcompact\b|\bcozy\b|\bcosy\b|\bsmall\b",
}

def engineer(Lsrc):
    """Builds the factor table for any subset of listings (index preserved)."""
    X = pd.DataFrame(index=Lsrc.index)
    # property & capacity
    X["room_type"] = Lsrc.room_type
    X["property_type_grp"] = Lsrc.property_type.where(Lsrc.property_type.isin(TOP_PTYPES), "Other")
    X["accommodates"] = Lsrc.accommodates
    X["bedrooms"] = Lsrc.bedrooms
    X["studio"] = (Lsrc.bedrooms == 0).astype(int)
    X["beds"] = Lsrc.beds
    txt = Lsrc.bathrooms_text.fillna("").str.lower()
    X["baths"] = txt.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    X.loc[txt.str.contains("half") & X["baths"].isna(), "baths"] = 0.5
    X["bath_shared"] = txt.str.contains("shared").astype(int)
    # stay policy
    X["long_stay"] = (Lsrc.minimum_nights >= 28).astype(int)
    X["log_min_nights"] = np.log(Lsrc.minimum_nights.clip(lower=1))
    # quote context (properties of the scraped price quote, not of the listing)
    q_in, q_out = pd.to_datetime(Lsrc.price_quote_checkin_date), pd.to_datetime(Lsrc.price_quote_checkout_date)
    q_nights = (q_out - q_in).dt.days
    X["q_log_nights"] = np.log(q_nights.where(q_nights >= 1, Lsrc.minimum_nights).clip(lower=1))
    X["q_weekend"] = q_in.dt.dayofweek.isin([4, 5]).astype(int)       # Friday / Saturday check-in
    # location
    X["district_grp"] = Lsrc.neighbourhood_cleansed.where(Lsrc.neighbourhood_cleansed.isin(KEEP_DISTRICTS), "Other districts")
    X["region"] = Lsrc.neighbourhood_cleansed.map(REGION).fillna("New Territories & Islands")
    X["lat"], X["lon"] = Lsrc.latitude, Lsrc.longitude
    # transit & centrality / competition (computed on the full file in section 5)
    for c in ["dist_mtr_km", "n_mtr_1km", "dist_central_km", "dist_tst_km", "dist_hub_min_km", "listings_500m", "listings_1km"]:
        X[c] = GEO.loc[Lsrc.index, c]
    # amenities
    al = amen_lists.loc[Lsrc.index]
    for a, c in AMEN_COLS.items():
        X[c] = al.apply(lambda l, a=a: int(a in l))
    X["n_amenities"] = al.apply(len)
    # host profile
    X["pro_host"] = (Lsrc.calculated_host_listings_count >= 5).astype(int)
    X["host_local"] = Lsrc.host_location.fillna("").str.contains("Hong Kong").astype(int)
    # listing text
    blob = (Lsrc.name.fillna("") + " " + Lsrc.description.fillna("")).str.lower()
    for c, rx in TEXT_FLAGS.items():
        X[c] = blob.str.contains(rx, regex=True).astype(int)
    X["desc_words_log"] = np.log1p(Lsrc.description.fillna("").str.split().str.len())
    return X

X_all = engineer(df)

CAT_COLS = ["room_type", "property_type_grp", "district_grp", "region"]
FAMILIES = {
    "Property & capacity": ["room_type", "property_type_grp", "accommodates", "bedrooms", "studio", "beds", "baths", "bath_shared"],
    "Stay policy & quote": ["long_stay", "log_min_nights", "q_log_nights", "q_weekend"],
    "Location": ["district_grp", "region", "lat", "lon"],
    "Transit & centrality": ["dist_mtr_km", "n_mtr_1km", "dist_central_km", "dist_tst_km", "dist_hub_min_km"],
    "Local competition": ["listings_500m", "listings_1km"],
    "Amenities": list(AMEN_COLS.values()) + ["n_amenities"],
    "Host profile": ["pro_host", "host_local"],
    "Listing text": list(TEXT_FLAGS) + ["desc_words_log"],
}
NEW_COLS = [c for fam in FAMILIES.values() for c in fam]
assert set(NEW_COLS) == set(X_all.columns), set(X_all.columns) ^ set(NEW_COLS)
print({k: len(v) for k, v in FAMILIES.items()}, "-> total", len(NEW_COLS), "launch-time factors")
RESULTS["features"] = dict(families={k: len(v) for k, v in FAMILIES.items()}, n_total=len(NEW_COLS), amenities_flagged=AMEN,
                           top_property_types=TOP_PTYPES, kept_districts=KEEP_DISTRICTS)
display(X_all[["accommodates", "bedrooms", "baths", "bath_shared", "dist_mtr_km", "dist_central_km", "listings_500m", "n_amenities", "pro_host"]].describe().T[["mean", "50%", "min", "max"]].round(2))
print(f"Weekend check-in share of quotes: short stays {pct(X_all.loc[X_all.long_stay==0,'q_weekend'].mean())}, long stays {pct(X_all.loc[X_all.long_stay==1,'q_weekend'].mean())}")
print("Missing values in launch-time factors:", X_all[NEW_COLS].isna().sum()[lambda s: s > 0].to_dict(), "(imputed with the training median inside the pipeline)")
