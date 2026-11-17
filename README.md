# What should a Hong Kong host charge? – ML pricing benchmark for short- and mid-term rentals

IS3240 Advanced Business Programming in Python – Group Project (CityU).

**Group: Exchange Pro 2** – Pol Montanera (SID 40181821), Guillem Rovira (SID 40181810), Elies Aragonès (SID 40181833), Mark Hristovski (SID 40178816).

We build a machine-learning **pricing benchmark** for Hong Kong Airbnb-type listings: a predicted price with an 80% band
from information known before the first guest, an explanation of what drives price (stay length, listing type, district,
host type), a price audit that flags listings far from their benchmark, colour maps of where whole flats are expensive or cheap, plain-language
explanations of each predicted price (SHAP values + an OpenAI language model with a consistency check), and a test of whether the price gap is associated
with weaker demand.

## What you need to run the notebook

The notebook (`Code.ipynb`) is delivered **already executed**, so all results, tables and figures can be read without running anything.
To run it again you need three things:

### 1. The `data/` folder (not included because of its size)
The raw data files are **not part of the delivered files** because of their size (about 19 MB). **We can send them on request.**
They are public: Inside Airbnb, Hong Kong, snapshot of 25 September 2026 (<http://insideairbnb.com/get-the-data/>). Create a folder `data/` next to the notebook with:

| File | What it is | Needed? |
|---|---|---|
| `listings.csv.gz` (any file matching `listings*.csv.gz`) | Detailed listings (6,614 rows × 91 columns) | **Yes** |
| `listings.csv` | Summary listings (only used for one consistency check) | **Yes** |
| `mtr_stations.csv` | 98 MTR stations from OpenStreetMap | No: if missing, the notebook downloads it from the Overpass API (needs internet) |
| `reviews.csv.gz` | Individual reviews | No: it is not used |

### 2. An OpenAI key in a `.env` file (only for the "Why this price?" explanations)
Section 13 of the notebook asks the OpenAI API (model `gpt-4o-mini`) to put the SHAP numbers into plain sentences.

1. Copy `.env.example` to `.env` (same folder as the notebook).
2. Open `.env` and write your key: `OPENAI_API_KEY=sk-...` (alternatively, set it as the environment variable `OPENAI_API_KEY`).
3. `.env` is git-ignored: **never commit or share a key**.

**Without a key the notebook still runs end to end**: it uses a fixed template instead of the language model, so the explanations are less fluent but contain the same numbers.
The wording written by OpenAI can differ slightly between runs; the version in the delivered notebook and report comes from one real run with the API.

### 3. Python and the libraries
Tested with Python 3.11. Install the libraries with `pip install -r requirements.txt`.

## How to run
1. Put the files in `data/` (see above) and, optionally, create `.env` with the key.
2. `pip install -r requirements.txt`
3. Open `Code.ipynb` (Jupyter or VS Code) and choose *Restart & Run All*. It takes about 4–8 minutes.
   Results are written to `outputs/` (figures, `results.json` and `listing_price_audit.csv`).
4. Optional – rebuild the deliverables from source: `python build/make_notebook.py` (rebuilds and re-executes the notebook from `build/parts/*.py`)
   and then `python build/make_report.py` (regenerates the Word report from `outputs/results.json`).

## Repository contents
| Path | What it is |
|---|---|
| `Code.ipynb` | The full analysis (already executed, outputs included) |
| `Report.docx` | The report (max. 10 pages + appendix) |
| `data/` | Input data (see above; not included in the delivery) |
| `outputs/` | Figures, `results.json` (every number quoted in the report) and `listing_price_audit.csv` |
| `build/` | Source of the notebook (`parts/*.py`) and scripts that build the notebook and the report |
| `requirements.txt` | Library versions used |
| `.env.example` | Template of the `.env` file for the OpenAI key |

## Data sources
* Inside Airbnb – Hong Kong: <http://insideairbnb.com> (`listings.csv.gz` and `listings.csv`).
* OpenStreetMap contributors, railway stations via the Overpass API.
