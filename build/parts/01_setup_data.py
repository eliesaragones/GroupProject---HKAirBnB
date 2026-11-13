# %% [markdown]
# ## 2. Setup and data loading
# We load the libraries, fix a random seed and the project folders, and read the three Inside Airbnb files (detailed listings, summary
# listings and reviews). Then we audit the data: size, empty columns and how the price is stored.

# %%
# Libraries: pandas (tables), NumPy (arrays), matplotlib (charts), requests (web APIs), scikit-learn (machine learning), statsmodels (regression)
import os, re, json, glob, time, textwrap, warnings, contextlib, io
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
try:   # optional: use the operating-system certificates for HTTPS (needed behind some antivirus programs)
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
from sklearn.base import clone
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
with contextlib.redirect_stderr(io.StringIO()):     # hides a harmless joblib message on Windows computers without the "wmic" tool
    from joblib.externals.loky import cpu_count
    cpu_count(only_physical_cores=True)
SEED = 42                                           # fixed random seed, so that the results are reproducible
np.random.seed(SEED)

# %%
# Project folders (the `if` makes the notebook work also when it is opened from a sub-folder)
ROOT = Path.cwd()
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RESULTS = {}                                        # every number quoted in the report is collected here and saved in outputs/results.json
def save_results():
    with open(OUT / "results.json", "w", encoding="utf8") as f:
        json.dump(RESULTS, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# Common chart style: one colour-blind-safe palette
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948",
           ink="#0b0b0b", ink2="#52514e", muted="#898781", surface="#fcfcfb")
CAT = [PAL[c] for c in ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]]
mpl.rcParams.update({"axes.facecolor": PAL["surface"], "font.size": 8, "axes.titlesize": 8.8, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e1e0d9", "axes.axisbelow": True, "legend.frameon": False})

def savefig(name, fig):
    """Save the figure in outputs/figures (the report uses these files) and show it."""
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=220, facecolor="white")
    plt.show()

def pct(x, d=0):
    """pct(0.256) gives '26%'."""
    return f"{100*x:.{d}f}%"

# %%
# Load the files. L = listings (one row per listing), S = summary file (only used for a check), R = reviews (one row per review)
L = pd.read_csv(sorted(glob.glob(str(DATA / "listings*.csv.gz")))[0])
S = pd.read_csv(DATA / "listings.csv")
R = pd.read_csv(DATA / "reviews.csv.gz")
SCRAPE_DATE = pd.to_datetime(L.last_scraped.iloc[0])

# The price is stored as text such as "$1,024.33": remove "$" and "," and convert to a number (errors="coerce": unreadable -> NaN)
L["price_hkd"] = pd.to_numeric(L.price.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
for c in ["first_review", "last_review"]:
    L[c] = pd.to_datetime(L[c])
R["date"] = pd.to_datetime(R["date"])

# First audit of the data
empty_cols = L.columns[L.isna().all()].tolist()
same_price = (S.set_index("id").price.round(2) == L.set_index("id").price_hkd.round(2))[L.set_index("id").price_hkd.notna()]
print(f"Detailed listings: {L.shape[0]:,} rows x {L.shape[1]} columns (scraped {SCRAPE_DATE.date()}) | Summary: {S.shape[0]:,} x {S.shape[1]} | Reviews: {R.shape[0]:,}")
print(f"Completely empty columns: {len(empty_cols)}, e.g. {empty_cols[:6]}")
print(f"The summary price equals the detailed price for only {pct(same_price.mean())} of the listings -> we use the detailed file")
RESULTS["data"] = dict(n_listings=len(L), n_columns=L.shape[1], n_reviews=len(R), n_listings_with_reviews=int(R.listing_id.nunique()),
                       scrape_date=str(SCRAPE_DATE.date()), review_first=str(R.date.min().date()), review_last=str(R.date.max().date()),
                       n_empty_columns=len(empty_cols), license_missing=bool(L.license.isna().all()), price_missing_pct=float(L.price_hkd.isna().mean()))
pd.DataFrame({"missing_%": (L[["price_hkd", "minimum_nights", "bedrooms", "beds", "review_scores_rating"]].isna().mean() * 100).round(1)}).T

# %% [markdown]
# **What the audit tells us:**
# * `price` is the **price quote for a specific stay** (the `price_quote_*` columns give the check-in/out dates) and the length of that stay is almost
#   identical to the host's `minimum_nights`. A "price per night" is therefore *not comparable* between a 1-night and a 30-night stay: **stay length must be a core factor**.
# * 15% of the listings have **no price** (we check below that most of them cannot be booked).
# * The `license` column is empty, so legal status **cannot** be analysed; several other columns are 100% empty and ignored.
