# %% [markdown]
# ## 2. Setup and data loading

# %%
import os, warnings, json, re, glob, time, textwrap
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # silences a harmless joblib warning on Windows
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import BallTree
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path.cwd()
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RESULTS = {}          # every number quoted in the report is stored here and written to outputs/results.json

def save_results():
    def _clean(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, (pd.Timestamp,)): return str(o.date())
        return str(o)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=_clean), encoding="utf8")

# ---- visual identity: validated colour-blind-safe categorical palette, one-hue sequential ramp ----
PAL = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
           green="#008300", violet="#6250d6", red="#e34948",
           ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = [PAL[c] for c in ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]]

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": PAL["surface"], "savefig.facecolor": "white",
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "text.color": PAL["ink2"],
    "axes.titlesize": 8.8, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": PAL["ink"],
    "axes.labelsize": 8, "axes.labelcolor": PAL["ink2"], "axes.edgecolor": PAL["axis"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": PAL["grid"],
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": PAL["muted"], "ytick.color": PAL["muted"],
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.dpi": 110, "savefig.dpi": 220,
})

def savefig(name, fig=None):
    fig = fig or plt.gcf()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.show()

def pct(x, d=0): return f"{100*x:.{d}f}%"

# %%
f_det = sorted(glob.glob(str(DATA / "listings*.csv.gz")))[0]          # detailed listings file (any '(1)' suffix is fine)
L = pd.read_csv(f_det)
S = pd.read_csv(DATA / "listings.csv")                                   # summary file (quick checks only)
R = pd.read_csv(DATA / "reviews.csv.gz")
SCRAPE_DATE = pd.to_datetime(L.last_scraped.iloc[0])

L["price_hkd"] = pd.to_numeric(L.price.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
for c in ["first_review", "last_review"]:
    L[c] = pd.to_datetime(L[c])
R["date"] = pd.to_datetime(R["date"])

empty_cols = L.columns[L.isna().all()].tolist()
print(f"Detailed listings : {L.shape[0]:,} rows x {L.shape[1]} columns   (scraped {SCRAPE_DATE.date()})")
print(f"Summary listings  : {S.shape[0]:,} rows x {S.shape[1]} columns")
print(f"Reviews           : {R.shape[0]:,} rows, {R.listing_id.nunique():,} listings, {R.date.min().date()} -> {R.date.max().date()}")
print(f"Completely empty columns in the detailed file: {len(empty_cols)}  e.g. {empty_cols[:8]}")
print("Same listing ids in summary and detailed file:", set(S.id) == set(L.id))
print(f"Summary price == detailed price for only {pct((S.set_index('id').price.round(2) == L.set_index('id').price_hkd.round(2))[L.set_index('id').price_hkd.notna()].mean())} of listings -> we use the detailed file only")

RESULTS["data"] = dict(
    n_listings=len(L), n_columns=L.shape[1], n_reviews=len(R), n_listings_with_reviews=int(R.listing_id.nunique()),
    scrape_date=str(SCRAPE_DATE.date()), review_first=str(R.date.min().date()), review_last=str(R.date.max().date()),
    n_empty_columns=len(empty_cols), license_missing=bool(L.license.isna().all()),
    price_missing_pct=float(L.price_hkd.isna().mean()),
)
display_cols = ["price_hkd", "minimum_nights", "bedrooms", "beds", "review_scores_rating", "amenities", "description"]
pd.DataFrame({"missing_%": (L[display_cols].isna().mean() * 100).round(1)}).T

# %% [markdown]
# **What the audit tells us (and what we decide):**
# * The `price` field is a **price quote for a specific stay** (the `price_quote_*` columns show check-in/out dates), and the
#   length of that stay is almost identical to the host's `minimum_nights`. A "price per night" is therefore *not comparable*
#   between a 1-night and a 30-night stay – we must treat **stay length as a core factor**, not as noise.
# * 15% of listings have **no price** – we check below that these are essentially listings that cannot be booked.
# * The `license` column is entirely empty, so legal status **cannot** be analysed from these data (we do not pretend to).
# * Several columns are 100% empty (host response rate, etc.) and are ignored.
