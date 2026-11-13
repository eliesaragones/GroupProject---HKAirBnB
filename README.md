# What should a Hong Kong host charge? – ML pricing benchmark for short- and mid-term rentals

IS3240 Advanced Business Programming in Python – Group Project (CityU).

We build a machine-learning **pricing benchmark** for Hong Kong Airbnb-type listings: a predicted price with an 80% band
from information known before the first guest, an explanation of what drives price (stay length, listing type, district,
host type), a price audit that flags listings far from their benchmark, colour maps of where whole flats are expensive or cheap, plain-language
explanations of each predicted price (SHAP values + an optional OpenAI language model with a check), and a test of whether the price gap is associated
with weaker demand.

## Repository contents
| Path | What it is |
|---|---|
| `Code.ipynb` | The full analysis (already executed, outputs included) |
| `Report.docx` | The report (max. 10 pages + appendix) |
| `data/` | Inside Airbnb Hong Kong listings and reviews (scraped 25 Sep 2026) and `mtr_stations.csv` (MTR stations from OpenStreetMap) |
| `outputs/` | Figures, `results.json` (every number quoted in the report) and `listing_price_audit.csv` |
| `build/` | Source of the notebook (`parts/*.py`) and scripts that build the notebook and the report |
| `requirements.txt` | Library versions used |

## How to run
1. `pip install -r requirements.txt`
2. Open `Code.ipynb` and choose *Restart & Run All* (about 4–8 minutes). It reads the files in `data/`; the MTR stations are cached
   in `data/mtr_stations.csv`, so no internet is needed (if the cache is missing, they are downloaded from the Overpass API).
3. Optional – plain-language "why this price?" explanations with an LLM: put your OpenAI key in the environment variable `OPENAI_API_KEY`
   (or copy `.env.example` to `.env`, which is git-ignored). Without a key the notebook uses a deterministic template and still runs end to end.
   Never commit a key.
4. Optional – rebuild the deliverables from source: `python build/make_notebook.py` (re-executes the notebook) and then
   `python build/make_report.py` (regenerates the report from `outputs/results.json`); add `--tex` to write `Report.tex` instead (compile with `xelatex Report.tex`, twice).

## Data sources
* Inside Airbnb – Hong Kong: <http://insideairbnb.com> (`listings.csv.gz`, `listings.csv`, `reviews.csv.gz`).
* OpenStreetMap contributors, railway stations via the Overpass API.
