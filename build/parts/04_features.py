# %% [markdown]
# ## 6. Useful factors: what could explain a listing's price, and why?
# A model needs a table with one row per listing and one column per *factor* that might influence the price; this is where the business knowledge enters.
# We organise the factors into **families**, each with a business rationale (hedonic pricing: a listing is a bundle of attributes that guests value). Grouping matters because
# later we measure how much each *family* adds, which is a far more useful business statement than "feature 37 has importance 0.02".
#
# | Family | Factors | Business rationale |
# |---|---|---|
# | **Property & capacity** | room type, property type, guests, bedrooms, beds, baths, shared bathroom | Space and privacy are what guests buy |
# | **Stay policy & quote** | long stay (≥ 28 nights), log minimum nights, nights of the quoted stay, weekend check-in | Defines the customer (tourist vs. monthly renter). The scraped price is a *quote for a specific stay*, so the quote variables are **measurement controls**, set to a standard stay when benchmarking |
# | **Location** | district, region, latitude/longitude | Neighbourhood prestige and demand |
# | **Transit & centrality** | distance to the nearest MTR, MTR stations within 1 km, distance to Central / TST / nearest hub | Hong Kong is a transit city |
# | **Local competition** | listings within 500 m and 1 km | More substitutes = price pressure (or a proven tourist area) |
# | **Amenities** | 30 most common amenity flags + total count | Convenience that guests pay for |
# | **Host profile** | professional host (5+ listings), host lives in Hong Kong | Professional operators price differently. Richer host variables (portfolio size, tenure...) did **not** help on unseen hosts, because they let a model recognise individual hosts |
#
# *Deliberately excluded:* the quote's **lead time** (days until the quoted check-in) is partly an *outcome* of demand (busy listings are booked up), so it would leak demand into price.
# **Ratings and review activity** exist only after a listing has guests; a new host has none, so they are **not** used by the benchmark for new listings (only in a separate model, section 11).

# %%
# Simplify categories with too many values (rare categories only add noise); we only use counts here, never the price
ptype_counts = df.property_type.value_counts()
TOP_PTYPES = ptype_counts[ptype_counts >= 50].index.tolist()          # property types with at least 50 listings; the rest become "Other"
dist_counts = df.neighbourhood_cleansed.value_counts()
KEEP_DISTRICTS = dist_counts[dist_counts >= 60].index.tolist()        # districts with at least 60 listings; the rest become "Other districts"
REGION = {d: "Hong Kong Island" for d in ["Central & Western", "Wan Chai", "Eastern", "Southern"]}
REGION.update({d: "Kowloon" for d in ["Yau Tsim Mong", "Sham Shui Po", "Kowloon City", "Wong Tai Sin", "Kwun Tong"]})   # all others: New Territories & Islands

# Amenities are stored as one text per listing that looks like a Python list: '["Wifi", "Kitchen", ...]'
def parse_list(s):
    try:
        return json.loads(s)
    except Exception:
        return []

amen_lists = L.amenities.apply(parse_list)
amen_prev = pd.Series([a for lst in amen_lists.loc[df.index] for a in lst]).value_counts() / len(df)      # share of listings that offer each amenity
AMEN = amen_prev[(amen_prev >= 0.05) & (amen_prev <= 0.97)].head(30).index.tolist()                       # 30 most common, neither universal nor rare
AMEN_COLS = {a: "am_" + re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_") for a in AMEN}                   # amenity name -> column name
print(f"{len(TOP_PTYPES)} property types and {len(KEEP_DISTRICTS)} districts kept; {len(amen_prev)} distinct amenities, {len(AMEN)} flagged: {AMEN[:6]} ...")

# %%
# The factor table: one small block per family
X_all = pd.DataFrame(index=df.index)
X_all["room_type"] = df.room_type
X_all["property_type_grp"] = df.property_type.where(df.property_type.isin(TOP_PTYPES), "Other")
X_all["accommodates"], X_all["bedrooms"], X_all["beds"] = df.accommodates, df.bedrooms, df.beds
X_all["studio"] = (df.bedrooms == 0).astype(int)                                       # 0 bedrooms = studio
bath_text = df.bathrooms_text.fillna("").str.lower()                                   # e.g. "1.5 baths", "1 shared bath", "Shared half-bath"
X_all["baths"] = bath_text.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)            # the number in the text (regular expression)
X_all.loc[bath_text.str.contains("half") & X_all.baths.isna(), "baths"] = 0.5          # a "half-bath" without number counts as 0.5
X_all["bath_shared"] = bath_text.str.contains("shared").astype(int)

X_all["long_stay"] = df.long_stay
X_all["log_min_nights"] = np.log(df.minimum_nights.clip(lower=1))
q_in, q_out = pd.to_datetime(df.price_quote_checkin_date), pd.to_datetime(df.price_quote_checkout_date)   # the quoted stay
q_nights = (q_out - q_in).dt.days
X_all["q_log_nights"] = np.log(q_nights.where(q_nights >= 1, df.minimum_nights).clip(lower=1))            # if unknown, use the minimum nights
X_all["q_weekend"] = q_in.dt.dayofweek.isin([4, 5]).astype(int)                        # Friday or Saturday check-in

X_all["district_grp"] = df.neighbourhood_cleansed.where(df.neighbourhood_cleansed.isin(KEEP_DISTRICTS), "Other districts")
X_all["region"] = df.neighbourhood_cleansed.map(REGION).fillna("New Territories & Islands")
X_all["lat"], X_all["lon"] = df.latitude, df.longitude

transit_cols = ["dist_mtr_km", "n_mtr_1km", "dist_central_km", "dist_tst_km", "dist_hub_min_km", "listings_500m", "listings_1km"]   # computed in section 5
X_all[transit_cols] = df[transit_cols]
for amenity, col in AMEN_COLS.items():
    X_all[col] = amen_lists.loc[df.index].apply(lambda lst: int(amenity in lst))       # 1 if the listing offers the amenity
X_all["n_amenities"] = amen_lists.loc[df.index].apply(len)

X_all["pro_host"] = (df.calculated_host_listings_count >= 5).astype(int)               # professional operator: 5+ listings
X_all["host_local"] = df.host_location.fillna("").str.contains("Hong Kong").astype(int)

CAT_COLS = ["room_type", "property_type_grp", "district_grp", "region"]               # columns that contain categories (text)
FAMILIES = {
    "Property & capacity": ["room_type", "property_type_grp", "accommodates", "bedrooms", "studio", "beds", "baths", "bath_shared"],
    "Stay policy & quote": ["long_stay", "log_min_nights", "q_log_nights", "q_weekend"],
    "Location": ["district_grp", "region", "lat", "lon"],
    "Transit & centrality": ["dist_mtr_km", "n_mtr_1km", "dist_central_km", "dist_tst_km", "dist_hub_min_km"],
    "Local competition": ["listings_500m", "listings_1km"],
    "Amenities": list(AMEN_COLS.values()) + ["n_amenities"],
    "Host profile": ["pro_host", "host_local"]}
NEW_COLS = [c for cols in FAMILIES.values() for c in cols]                            # all launch-time factors
assert set(NEW_COLS) == set(X_all.columns)                                            # check: the families cover exactly the columns of the table
RESULTS["features"] = dict(families={k: len(v) for k, v in FAMILIES.items()}, n_total=len(NEW_COLS), amenities_flagged=AMEN,
                           top_property_types=TOP_PTYPES, kept_districts=KEEP_DISTRICTS)
print({k: len(v) for k, v in FAMILIES.items()}, "-> total", len(NEW_COLS), "launch-time factors")
print("Missing values:", X_all.isna().sum()[lambda s: s > 0].to_dict(), "(filled with the training median before modelling)")

# Experience factors (only for listings that already have guests): ratings and review activity
RATING_COLS = ["accuracy", "cleanliness", "checkin", "communication", "location", "value", "rating"]
EXP_ALL = pd.DataFrame({f"rs_{c}": df[f"review_scores_{c}"] for c in RATING_COLS}, index=df.index)
EXP_ALL["log_n_reviews"] = np.log1p(df.number_of_reviews)
EXP_ALL["reviews_ltm"] = df.number_of_reviews_ltm                                     # reviews in the last 12 months
EXP_ALL["reviews_per_month"] = df.reviews_per_month
EXP_ALL["days_since_last_review"] = (SCRAPE_DATE - df.last_review).dt.days
EXP_ALL["superhost"] = (df.host_is_superhost == "t").astype(int)
EXP_ALL["has_reviews"] = (df.number_of_reviews > 0).astype(int)
EXP_FAMILIES = {"Reviews & ratings": list(EXP_ALL.columns)}
