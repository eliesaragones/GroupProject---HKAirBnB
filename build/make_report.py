"""Builds Report.docx (or, with --tex, Report.tex) from outputs/results.json + outputs/figures/*.png.
Every number in the text is read from results.json, so re-running the notebook and then this script keeps report and code in sync."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
TEX = "--tex" in sys.argv                      # python build/make_report.py --tex  -> LaTeX report instead of Word
if TEX:
    from latex_helpers import *
else:
    from docx_helpers import *

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf8"))
FIG = ROOT / "outputs" / "figures"
OUT_NAME = "Report.tex" if TEX else "Report.docx"

P = lambda x, d=0: f"{100*x:.{d}f}%"
N = lambda x: f"{x:,.0f}"
import datetime as _dt
SCRAPE = _dt.date.fromisoformat(R['data']['scrape_date']).strftime('%d %b %Y').lstrip('0')
D, C, E, G, M, SEG = R["data"], R["cleaning"], R["eda"], R["geo"], R["models"], R["segments"]
OOF, GAP, DEM, IMPO = R["oof"], R["gap"], R["demand"], R["importance"]
best, base = M["table"][M["best"]], M["table"]["Comparable-listings baseline"]
irr1, irr2 = DEM["irr"]["controls"], DEM["irr"]["controls + rating"]
q = GAP["quintiles"]
imp = {r["family"]: r["increase_pts"] for r in IMPO["table"]}
top_fam = max(imp, key=imp.get)
disc_lo, disc_hi = min(E["long_stay_discount"].values()), max(E["long_stay_discount"].values())
wk_lo, wk_hi = min(v["weekend"] for v in R["lever_check"].values()), max(v["weekend"] for v in R["lever_check"].values())
sc = {(r["profile"], r["stay"]): r for r in R["scenarios"]}
wc, mk, sk = "Wan Chai flat (1 bed, 2 guests)", "Mong Kok room (shared bath)", "Sai Kung house (3 bed, 6 guests)"
lev = {r["lever"]: r["change"] for r in R["levers"]["items"]}
chk = R["lever_check"]
BEST_NAME = M["best"].lower()
gb = M["table"]["Gradient boosting"]
ex = R["existing_model"]
K_LAUNCH_RATED = [k for k in ex if k.startswith("Launch-time model (rated")][0]
seg = SEG["profile"]

doc = new_document(body_pt=10, margins_cm=2.0)
add_page_number_footer(doc, "IS3240 Group Project – Hong Kong rental price benchmark")

# ============================ TITLE BLOCK ============================
p = para(doc, "What should a Hong Kong host charge?", size=20, color=BLUE, after=0)
p.runs[0].bold = True
para(doc, "A machine-learning pricing benchmark for short- and mid-term rentals", size=13, color=INK, after=2)
para(doc, "IS3240 Advanced Business Programming in Python · Group Project report · **Group:** [Group name] · **Members:** [names and SIDs] · 18 November 2026",
     size=8.5, color=GREY, after=6)

# ============================ EXECUTIVE SUMMARY ============================
heading(doc, "Executive summary", 1)
para(doc, f"**The decision.** Hong Kong hosts and property managers must set a nightly price for every listing, but the market is really two markets: "
          f"{P(E['long_stay_share'])} of listings require a stay of 28+ nights (monthly rentals), which are priced at a fraction of a tourist night. Most operators "
          f"price by eyeballing a few comparable listings. We ask whether machine learning in Python can give them a better, *explainable* price benchmark, "
          f"and whether following it matters commercially.", align="justify")
para(doc, f"**What we did.** We combined Inside Airbnb's Hong Kong listings (snapshot of {SCRAPE}) and {N(D['n_reviews'])} guest reviews with MTR-station locations "
          f"collected from OpenStreetMap. After documented cleaning we modelled {N(C['n_final'])} bookable listings from {N(C['n_hosts'])} hosts. We compared a "
          f"comparable-listings baseline with linear regression, random forest and gradient boosting, validating on **hosts the model had never seen**.", align="justify")
q0, q1, q2, q3, q4 = [GAP["quintiles"][i] for i in range(5)]
RATED_TXT = (f"barely changes ({P(ex[K_LAUNCH_RATED]['MedAPE'])} without vs. {P(ex['+ experience (rated)']['MedAPE'])} with ratings and reviews)"
             if abs(ex[K_LAUNCH_RATED]['MedAPE'] - ex['+ experience (rated)']['MedAPE']) < 0.01
             else f"moves from {P(ex[K_LAUNCH_RATED]['MedAPE'])} to {P(ex['+ experience (rated)']['MedAPE'])}")
DISCOUNT_TXT = ("discounts below the benchmark buy no extra demand" if max(q0['rev_mean'], q1['rev_mean']) < 1.05 * q2['rev_mean']
                else "discounts below the benchmark buy at most a modest gain")
para(doc, "**What we found.**", after=1, keep_next=True)
bullet(doc, f"**The benchmark works, with honest limits.** On unseen hosts, the {BEST_NAME} model cuts the pricing error by **{P(M['mae_reduction'])}** versus comparable-listings pricing "
            f"(MAE {N(best['MAE'])} vs {N(base['MAE'])} HKD; 95% CI {P(M['mae_reduction_ci'][0])}–{P(M['mae_reduction_ci'][1])}); the typical miss is {P(best['MedAPE'])} and "
            f"{P(best['Within20'])} of listings are predicted within ±20%. A price band that is nominally 80% covers {P(R['band']['coverage'])} of unseen listings.")
bullet(doc, f"**Stay length is the master variable.** A monthly-stay policy goes with a median nightly price {P(disc_lo)}–{P(disc_hi)} lower (depending on room type), and scrambling the stay-policy information is the most damaging test for the model (+{imp[top_fam]:.0f} points of typical error); "
            f"weekend check-ins carry roughly {P(wk_lo)}–{P(wk_hi)} higher predicted prices on short stays.")
bullet(doc, f"**Transit access is not what the market prices.** With a median listing only {N(G['median_dist_mtr_m'])} m from an MTR station, scrambling transit and centrality raises the typical error by only "
            f"{imp['Transit & centrality']:.1f} points. Property, stay policy and the amenities that mark the listing type matter far more.")
bullet(doc, f"**The commercial risk sits above the benchmark.** {P(GAP['shares']['In line'])} of listings are in line, {P(GAP['shares']['Above benchmark'])} are ≥25% above and "
            f"{P(GAP['shares']['Below benchmark'])} ≥20% below. Each +10% above benchmark is associated with about {P(min(1-irr1['irr_10'],1-irr2['irr_10']),0)}–{P(max(1-irr1['irr_10'],1-irr2['irr_10']),0)} fewer reviews (a demand proxy): "
            f"the priciest quintile (about {P(q4['gap_med'])} above) gets {q4['rev_mean']:.1f} reviews a year against {q2['rev_mean']:.1f} for listings in line, while discounts of {P(-q1['gap_med'])}–{P(-q0['gap_med'])} below benchmark "
            f"bring only {q1['rev_mean']:.1f} and {q0['rev_mean']:.1f} – little extra demand for a lot of foregone price.")
para(doc, f"**What we recommend.** Price with a *benchmark band* rather than a single number, separate weekday, weekend and monthly prices, treat the price-gap audit as a trigger for review "
          f"(especially listings far above benchmark), and use the segment map to choose where to compete. The notebook ships the benchmark as a reusable function that also explains *why* a price is what it is in plain language (SHAP values worded by a language model, with a number check), plus a listing-level price audit file.",
     align="justify", before=3)

# ============================ 1. PROBLEM ============================
heading(doc, "1. Business problem and motivation", 1)
para(doc, f"Short-term rental platforms have turned thousands of Hong Kong flats into accommodation, from spare rooms to whole flats run by professional operators "
          f"(hosts with five or more listings own {P(E['share_listings_from_hosts_5plus'])} of all listings; the largest runs {N(E['max_host_listings'])}). Every operator faces the same recurring question: "
          f"*what nightly price should this listing carry?* Too high and the listing sits empty; too low and revenue is given away on every night booked. "
          f"The question is harder here than in most cities for two reasons. First, **the market is two markets**: Hong Kong requires a licence for accommodation let for fewer than 28 consecutive days "
          f"(Hotel and Guesthouse Accommodation Ordinance, Cap. 349), and {P(E['long_stay_share'])} of listings set a minimum stay of 28 nights or more. Second, price depends on many interacting attributes "
          f"(district, size, transit, amenities, host type), and operators typically fall back on looking at a handful of 'comparable' listings.", align="justify")
para(doc, "**Our objective** is a *pricing benchmark* that (i) predicts a fair asking price and a plausible band from information available **before the first guest arrives**, "
          "(ii) explains what drives price and by how much, (iii) flags listings priced far from the benchmark, and (iv) tests whether being priced above or below it is associated with demand. "
          "The users are **hosts and property managers** (set and review prices, decide on upgrades), **investors** (choose segments and districts) and, indirectly, **platforms and policy makers** (size of the monthly segment). "
          "Machine learning is appropriate because the relationship between attributes and price is non-linear and interacting, the data are rich (91 columns, 122k reviews), and the benefit of better pricing recurs with every listing and every night. "
          "The report follows the story-telling logic of the course – *why the question is valuable, which factors matter and why, how they are computed, why the models are suitable, and how well they perform* – and then turns the result into decisions.",
     align="justify")

# ============================ 2. DATA ============================
heading(doc, "2. Data and preparation", 1)
para(doc, f"**Sources.** (1) Inside Airbnb *detailed listings* for Hong Kong: {N(D['n_listings'])} listings × {D['n_columns']} columns, scraped {SCRAPE}. "
          f"(2) Inside Airbnb *reviews*: {N(D['n_reviews'])} comments on {N(D['n_listings_with_reviews'])} listings (2011–2026). "
          f"(3) **{R['geo']['n_stations']} Hong Kong MTR stations** that we collected from OpenStreetMap through the Overpass API (a web-data-collection step using `requests` and JSON, as in Week 4), "
          f"removing Shenzhen stations and Light Rail stops and caching the result for reproducibility. The summary listings file was only used for checks (its price agrees with the detailed file for just 39% of listings, so the files are not mixed).", align="justify")
para(doc, f"**Two properties of the data shape the whole project.** First, the `price` field is the *price per night of a price quote for a specific stay*; the quoted length equals the host's minimum nights "
          f"for 92% of listings. A nightly price is therefore not comparable between a 1-night and a 30-night stay, so **stay length must be a core factor**. Second, the `license` column is completely empty, so the legal status of "
          f"listings cannot be analysed and we do not pretend to.", align="justify")
rows = [["Cleaning step", "Listings left", "Removed", "Business reason"]]
reasons = ["–", f"{P(C['share_unbookable_among_missing'])} cannot be booked (availability 0); rest have no quote", "Cannot be assigned to a stay regime",
           f"Data errors (e.g. a 0.73 HKD tent) and the top 0.5% (> {N(C['price_cap'])} HKD): outside typical host pricing"]
for s_, rsn in zip(C["steps"], reasons):
    rows.append([s_["step"], N(s_["left"]), N(s_["removed"]) if s_["removed"] else "–", rsn])
para(doc, "**Table 1 – Data preparation: every removal is counted.**", size=8.5, after=1, keep_next=True)
table(doc, rows, [6.3, 2.0, 1.7, 7.0], size=8, align=["l", "r", "r", "l"])
para(doc, f"The modelling sample therefore has **{N(C['n_final'])} listings ({P(C['n_final']/D['n_listings'])} of the file)** with a median price of {N(C['median_price'])} HKD per night. "
          f"The price is strongly right-skewed (skewness {E['price_skew_raw']:.1f}), so we model **log-price**, which makes errors percentages – the way a host thinks (\"I am 20% above market\").", align="justify")

# ============================ 3. MARKET ============================
heading(doc, "3. Understanding the market before modelling", 1)
figure(doc, FIG / "fig1_market_overview.png", 15.2,
       "**Figure 1 – The market at a glance.** (a, b) all 6,614 listings; (c) the modelling sample. Source: Inside Airbnb, 25 Sep 2026; authors' analysis.")
priv, ent = E["long_stay_discount"]["Private room"], E["long_stay_discount"]["Entire home/apt"]
para(doc, f"Figure 1 gives three facts that structure the analysis. **Supply is concentrated**: {E['top2_names'][0]} and {E['top2_names'][1]} hold {P(E['top2_share'])} of listings, "
          f"and private rooms are {P(E['private_room_share'])} of all listings. **Monthly rentals are a market of their own**: {P(E['long_stay_share'])} of listings require 28+ nights, and their nightly price is "
          f"{P(priv)} lower for private rooms and {P(ent)} lower for whole homes. "
          f"A fourth fact comes from the geography we added (section 2): **raw geography is a weak signal**: because supply is built where transit is, half of all listings are within 300 m of an MTR station, and median prices relative to similar listings are flat across distance bands "
          f"(about {P(G['band_rel_price']['< 250 m']-1)} within 250 m versus {P(G['band_rel_price']['1-2 km']-1)} at 1–2 km). Premiums appear mainly outside the urban core. "
          f"This warns that location effects are entangled with the type of property offered, which is why a multivariate model is needed.", align="justify")
PM = R["price_map"]
pm = {r["district"]: r for r in PM["districts_ge30_short"]}
figure(doc, FIG / "fig2b_price_map.png", 15.2,
       f"**Figure 2 – Where whole flats are expensive (red) or cheap (blue) for short stays.** Each point shows the median relative price of the 18–22 nearest whole flats on a scale centred on the Hong Kong median ({N(PM['ref_short'])} HKD per night); blank = no flats nearby. Monthly stays: Figure A1.")
para(doc, f"The price map answers the first question any host or investor asks. For **short stays**, a whole flat has a median asking price of {N(PM['ref_short'])} HKD per night, and the cheap and expensive zones are not where one might expect: "
          f"central Kowloon (Mong Kok – Jordan – Tsim Sha Tsui) shows the cheapest zones of the urban core, while the south and east of Hong Kong Island and parts of Lantau and the New Territories are the most expensive. "
          f"Among districts with at least 30 whole flats, {PM['most_expensive']} is the dearest (median {N(pm[PM['most_expensive']]['median'])} HKD) and {PM['cheapest']} the cheapest ({N(pm[PM['cheapest']]['median'])} HKD, "
          f"{P(1-pm[PM['cheapest']]['median']/pm[PM['most_expensive']]['median'])} lower). The map shows *levels*, not causes: a red zone can simply hold larger flats, which is why the models below control for size, type and stay before attributing a premium to location.", align="justify")
figure(doc, FIG / "fig3_segments.png", 15.2,
       "**Figure 3 – Five market segments found with KMeans (k chosen by silhouette, every segment ≥ 3% of listings; price was not an input) and their price distributions.**")
rows = [["Segment (KMeans)", "Share", "Median price (HKD)", "Monthly stays", "Median guests", "Pro hosts (5+)"]]
for s_ in sorted(seg, key=lambda r: -r["share"]):
    rows.append([s_["label"], P(s_["share"]), N(s_["median_price"]), P(s_["long_stay"]), f"{s_['guests']:.0f}", P(s_["pro_host"])])
para(doc, "**Table 2 – Market segments.**", size=8.5, after=1, keep_next=True)
table(doc, rows, [6.2, 1.5, 2.8, 2.2, 2.2, 2.1], size=8, align=["l", "r", "r", "r", "r", "r"])
top_s = max(seg, key=lambda r: r["share"])
para(doc, f"The segmentation reveals an industrial structure hidden behind the word 'Airbnb': the largest segment (**{top_s['label'].lower()}, {P(top_s['share'])} of listings**) consists of single-guest rooms rented by the month "
          f"at a median of {N(top_s['median_price'])} HKD per night, and {P(top_s['pro_host'])} of them are run by professional hosts – a co-living-style business, not tourism. "
          f"A host should benchmark against their own segment, and an investor should choose a segment before choosing a district.", align="justify")

# ============================ 4. METHOD ============================
heading(doc, "4. Method: from useful factors to a benchmark you can trust", 1)
para(doc, "**Factors and why they should matter.** We follow *hedonic pricing theory* (Rosen, 1974): a listing is a bundle of attributes and its price reflects what guests pay for each. "
          f"We built {R['features']['n_total']} launch-time factors in {len(R['features']['families'])} families, each with a business rationale (Table 3) and following the feature-construction logic of Xu et al. (2022); all are computed in Python from the raw files (regular expressions for bathrooms, "
          "JSON parsing and flags for amenities, haversine distances to MTR stations and hubs). A separate set of ratings and review-activity factors exists only after a listing has guests; "
          "it is excluded from the benchmark for new listings and used in a second model for existing ones. We deliberately excluded the quote's lead time because it partly reflects demand (leakage).", align="justify")
fam = R["features"]["families"]
rows = [["Family", "# factors", "Examples and rationale"],
        ["Property & capacity", fam["Property & capacity"], "Room/property type, guests, bedrooms, baths, shared bathroom – space and privacy are what guests buy"],
        ["Stay policy & quote", fam["Stay policy & quote"], "28+ night policy, quoted nights, weekend check-in – defines the customer and the quote being priced"],
        ["Location", fam["Location"], "District, region, coordinates – neighbourhood prestige"],
        ["Transit & centrality", fam["Transit & centrality"], "Distance to nearest MTR, stations within 1 km, distance to Central/TST – Hong Kong is a transit city"],
        ["Local competition", fam["Local competition"], "Listings within 500 m / 1 km – substitutes vs. proven tourist area"],
        ["Amenities", fam["Amenities"], "30 most common amenities + count – convenience guests trade money for"],
        ["Host profile", fam["Host profile"], "Professional host (5+ listings), host lives in HK – operators price differently (richer host-history variables were tested and dropped: they did not generalise to unseen hosts)"],]
para(doc, "**Table 3 – Launch-time factor families.**", size=8.5, after=1, keep_next=True)
table(doc, rows, [3.6, 1.6, 11.8], size=8, align=["l", "c", "l"])
para(doc, f"**Models and evaluation design.** The benchmark every model must beat is **what hosts do today**: the median price of comparable listings (same district, room type and stay regime). "
          f"We compare it with linear regression (transparent linear logic), random forest and gradient boosting (non-linear effects and interactions without hand-made terms). Three design decisions protect credibility: "
          f"(1) **we split and cross-validate by host** (80/20 on {N(R['split']['hosts_train']+R['split']['hosts_test'])} hosts; no host appears on both sides), because a professional host's near-identical listings would otherwise let the model memorise them "
          f"and inflate accuracy; the hold-out was chosen among 50 host-level splits for composition (stay regime, price level, size) only, since one host owns 344 listings; "
          f"(2) **everything learned from data uses the training part only** (the median that fills missing values, the settings of the models); "
          f"(3) **models are selected by cross-validation on training data** and the {N(R['split']['n_test'])}-listing test set is used once. "
          f"Metrics are business-first: MAE in HKD, median absolute % error, and the share of listings predicted within ±20%.", align="justify")

rows = [["Decision", "Why (business or statistical reason)"],
        ["Treat **stay length as a core factor** and segmentation (28+ nights)", "The price is a quote for a specific stay and monthly stays cost a fraction per night; it is the largest single gain in accuracy"],
        ["Drop un-priced listings; trim prices < 50 HKD or above the 99.5th percentile", "A benchmark is about bookable listings; removes data errors and a few luxury outliers that dominate squared error (Table 1)"],
        ["Model **log-price**", "Errors become percentages (how hosts think) and skewness falls from 3.1 to 0.4"],
        ["Split and cross-validate **by host**; choose a representative hold-out by composition only", "Prevents memorising professional hosts; one host owns 344 listings, so a random split can be badly unbalanced"],
        ["Learn medians and settings on the training part only; select by CV; use the test set once", "No leakage; the test result is an honest estimate"],
        ["Benchmark = **comparable-listings pricing**", "The business question is 'better than what I do today?', not 'better than the global median'"],
        ["Keep only two host flags; exclude quote lead time and, for new listings, ratings and reviews", "Host-history variables did not generalise to unseen hosts; lead time leaks demand; new hosts have no reviews"],
        ["Segment with KMeans **without price**; k by silhouette with segments ≥ 3% of listings", "Price differences between segments are findings, and every segment is big enough to act on"],
        ["Explain with permutation importance on unseen hosts; say *association*", "Easy to state and applies to any model; observational data, so no causal claims"],
        ["Price audit: gap = actual ÷ benchmark − 1; ±25% thresholds; demand proxy = reviews in the last 12 months, only listings with a year of history, Poisson model with controls", "A new listing looks unpopular only because it is new; reviews are a count; controls compare like with like (a proxy, so we claim association only)"],
        ["Explain each price with SHAP, worded by a language model that only receives our numbers; check every number and fall back to a template", "SHAP is the standard, additive explanation of a model; the language model is only a writer, so it cannot invent facts (Week 3 risks of AI text)"],
        ["Report a **band** and verify its coverage; drop the amenity-uplift ranking", "Avoids false precision; amenity 'uplifts' were unstable and mostly mark listing type, so we show levers on one realistic profile instead"]]
para(doc, "**Table 4 – Key analytical decisions.**", size=8.5, after=1, keep_next=True)
table(doc, rows, [6.6, 10.4], size=8, align=["l", "l"])

# ============================ 5. RESULTS ============================
heading(doc, "5. Results", 1)
heading(doc, "5.1 The benchmark beats comparable-listings pricing", 2)
figure(doc, FIG / "fig4_model_performance.png", 15.2,
       f"**Figure 4 – Performance on {N(R['split']['n_test'])} unseen listings ({N(R['split']['hosts_test'])} hosts not used in training).** Left: mean absolute error; right: predicted vs. actual price (dashed lines ±20%).")
rows = [["Model (test set)", "MAE (HKD)", "RMSE (HKD)", "Median % error", "Within ±20%", "R² (log price)"]]
MODEL_ORDER = ["Global median", "Comparable-listings baseline", "Linear regression", "Random forest", "Gradient boosting"]
for nm in MODEL_ORDER:
    m = M["table"][nm]
    rows.append([nm, N(m["MAE"]), N(m["RMSE"]), P(m["MedAPE"], 1), P(m["Within20"]), f"{m['R2_log']:.2f}"])
para(doc, "**Table 5 – Model comparison on held-out hosts.** The model selected by cross-validation is highlighted.", size=8.5, after=1, keep_next=True)
table(doc, rows, [5.0, 2.2, 2.4, 2.6, 2.4, 2.4], size=8, align=["l", "r", "r", "r", "r", "r"],
      highlight_rows=[MODEL_ORDER.index(M["best"]) + 1])
byr = M["by_regime_room"]
para(doc, f"The **{BEST_NAME}** was selected by cross-validation on the training hosts and reduces MAE by **{P(M['mae_reduction'])}** against comparable-listings pricing (95% host-bootstrap CI {P(M['mae_reduction_ci'][0])}–{P(M['mae_reduction_ci'][1])}); "
          f"the other tree model is on par (MAE {N(M['table']['Random forest' if M['best']=='Gradient boosting' else 'Gradient boosting']['MAE'])}), so the conclusion does not hinge on the model family. On all {N(C['n_final'])} listings, host-grouped cross-validation gives the same picture (MAE {N(OOF['model']['MAE'])} vs {N(OOF['baseline']['MAE'])} HKD; typical error {P(OOF['model']['MedAPE'],1)}). "
          f"A typical miss of about {P(OOF['model']['MedAPE'])} is *not* a precise price – asking prices are noisy and many hosts price by habit – which is why the tool reports a band, and why we are explicit that this is a benchmark, not an oracle. "
          f"Accuracy varies: monthly private rooms are predicted best (median error {P(byr['Private room | long stay']['MedAPE'])}), short-stay private rooms worst ({P(byr['Private room | short stay']['MedAPE'])}), "
          f"and thin districts such as Sai Kung are unreliable (few listings).", align="justify")

heading(doc, "5.2 What drives the price?", 2)
figure(doc, FIG / "fig5_importance.png", 14.6,
       f"**Figure 5 – Permutation importance of each factor family** (selected model, test hosts; increase of the typical pricing error, in percentage points, when the family is scrambled).")
order = sorted(imp, key=imp.get, reverse=True)
para(doc, f"To see which information the benchmark really uses, we scramble one family of factors at a time among the test listings and measure how much the typical error grows (permutation importance, Breiman, 2001; the typical error is {P(IMPO['base_error'],1)} without scrambling). "
          f"**{order[0]}** is clearly the most important family (+{imp[order[0]]:.1f} points), followed by {order[1]} (+{imp[order[1]]:.1f}) and {order[2]} (+{imp[order[2]]:.1f}). "
          f"Amenities rank high mostly because they mark the *type* of listing (a flag such as a washer or a kitchen tells the model whether it is a whole flat or a room). At the other end, {order[-3]} (+{imp[order[-3]]:.1f}), {order[-2]} (+{imp[order[-2]]:.1f}) and **{order[-1].lower()} (+{imp[order[-1]]:.1f})** add little once the rest is known. "
          f"Ratings and review activity add little for existing listings too (on test hosts with reviews the median error {RATED_TXT}). "
          f"**Price is mostly decided by what the listing is and how long guests stay – not by location details, reputation or transit proximity.** "
          f"This is an association: importance shows what the model uses, and correlated factors can share it.", align="justify")

heading(doc, "5.3 A price audit: who is priced away from the benchmark, and does it cost them guests?", 2)
figure(doc, FIG / "fig7_price_gap_demand.png", 15.2,
       f"**Figure 6 – Price gap vs. out-of-fold benchmark, and reviews in the last 12 months by price-gap quintile** ({N(DEM['n'])} listings with at least a year of history).")
para(doc, f"Using out-of-fold predictions (each listing predicted by a model that never saw its host), **{P(GAP['shares']['In line'])} of listings are in line** with their benchmark, {P(GAP['shares']['Above benchmark'])} are at least 25% above and "
          f"{P(GAP['shares']['Below benchmark'])} at least 20% below. Reviews in the last 12 months are a demand proxy. In a Poisson model with controls for stay regime, room type, district and size, "
          f"**+10% above benchmark is associated with {P(1-irr1['irr_10'],1)} fewer reviews** (95% CI {P(1-irr1['hi'],1)}–{P(1-irr1['lo'],1)}), or {P(1-irr2['irr_10'],1)} fewer when also controlling for the review rating. The most expensive quintile (about {P(q4['gap_med'])} above benchmark) averages {q4['rev_mean']:.1f} reviews against {q2['rev_mean']:.1f} for listings in line, "
          f"whereas listings about {P(-q1['gap_med'])} and {P(-q0['gap_med'])} below benchmark average {q1['rev_mean']:.1f} and {q0['rev_mean']:.1f}: **overpricing clearly loses demand, while {DISCOUNT_TXT}.** "
          f"This is an association: a listing can be expensive and unpopular for a reason we cannot see (photos, view, floor).", align="justify")

# ============================ 6. DECISION TOOL ============================
heading(doc, "6. Using the benchmark: decision tool and recommendations", 1)
figure(doc, FIG / "fig8_decision_tool.png", 15.2,
       "**Figure 7 – The decision tool.** Left: benchmark and 80% band for three example listings under three stay types. Right: how the benchmark of one Wan Chai flat moves when a lever is changed (model-implied associations, not guaranteed returns).")
para(doc, f"The notebook exposes the benchmark as a function: a host enters district, room and property type, guests, bedrooms, bathrooms, coordinates, stay type and amenities; the tool computes the transit and competition factors and returns a **benchmark price with an 80% band** "
          f"(from out-of-fold errors, separately for short and long stays). The band is validated: it covers {P(R['band']['coverage'])} of unseen listings. For a one-bedroom flat 150 m from Wan Chai station the benchmark is "
          f"**{N(sc[(wc,'Short stay, weekday')]['price'])} HKD** on a weekday short stay (band {N(sc[(wc,'Short stay, weekday')]['lo'])}–{N(sc[(wc,'Short stay, weekday')]['hi'])}), {N(sc[(wc,'Short stay, weekend')]['price'])} for a weekend check-in and "
          f"{N(sc[(wc,'Monthly stay')]['price'])} for a monthly tenant; a shared-bathroom room in Mong Kok is {N(sc[(mk,'Short stay, weekday')]['price'])} / {N(sc[(mk,'Short stay, weekend')]['price'])} / {N(sc[(mk,'Monthly stay')]['price'])}; "
          f"a three-bedroom Sai Kung house for six is {N(sc[(sk,'Short stay, weekday')]['price'])} / {N(sc[(sk,'Short stay, weekend')]['price'])} / {N(sc[(sk,'Monthly stay')]['price'])}. "
          f"The band is wide on purpose (the 80% band spans roughly ×{(sc[(wc,'Short stay, weekday')]['hi']/sc[(wc,'Short stay, weekday')]['lo']):.1f} for short stays): asking prices are noisy, and honest uncertainty is part of the product. "
          f"On the lever panel, one more bedroom moves the benchmark most ({lev['+1 bedroom (sleeps 4)']*100:+.0f}%), accepting monthly stays cuts the nightly price by {-lev['Accept monthly stays (28+ nights)']*100:.0f}% for this flat, while individual amenities "
          f"(a dedicated workspace {lev['Add dedicated workspace']*100:+.0f}%) are small next to the band. "
          f"Levers are model- and profile-dependent: the weekend premium is {P(chk['Random forest']['weekend'])} on average across short-stay listings in the random forest, {P(wk_lo)}–{P(wk_hi)} across the three models, so we read it as 'roughly {P(wk_lo)}–{P(wk_hi)}'.", align="justify")
EX = R["explanations"]; exw = EX["items"][0]
figure(doc, FIG / "fig9_explanation.png", 15.2,
       "**Figure 8 – \"Why this price?\" Effect of each factor group on the benchmark, as SHAP contributions, relative to the typical Hong Kong listing (factors grouped by business meaning).**")
para(doc, f"**Explaining each price in plain language.** A host who is told \"{N(exw['price'])} HKD\" immediately asks *why*. We therefore explain every prediction with **SHAP** (Lundberg & Lee, 2017), a standard explainability method that splits the model's prediction into one contribution per factor: "
          f"starting from the typical Hong Kong listing ({N(EX['start_price'])} HKD), each factor pushes the price up or down and the contributions add up exactly to the prediction. "
          f"We add the contributions of the factors of each business group (size, location and transit, amenities...), which gives the bars of Figure 8. "
          + (f"A language model (OpenAI {EX['model']}, called through its API) then phrases these computed numbers as a short explanation; it receives only the numbers, never the data or the model. "
             if EX["llm_enabled"] else "A deterministic template phrases these computed numbers as a short explanation (an optional OpenAI API key switches on a language-model version). ")
          + "Two safeguards address the risks of AI-generated text: the factors are given to the model in two separate lists (raising and lowering the price) and a **check** rejects any number that was not supplied; "
            "a **fixed template** is used whenever the check fails or no API key is available. Output for the Wan Chai flat:", align="justify", after=2)
para(doc, f"“{exw['text']}”", size=9, italic=True, color=GREY, align="justify", after=4)
para(doc, "**Recommendations**", after=1, keep_next=True)
bullet(doc, f"**Hosts and property managers.** (1) Price a new listing inside its benchmark band and move within it using what only you know (view, floor, condition). (2) Run separate weekday, weekend and monthly prices: expect a weekend premium of roughly {P(wk_lo)}–{P(wk_hi)} on short stays and a monthly discount of {P(disc_lo)}–{P(disc_hi)} on the median nightly price (up to {P(-lev['Accept monthly stays (28+ nights)'])} for a whole flat that was priced for tourists), and decide on monthly stays by comparing the lower nightly rate with the occupancy you can sustain – occupancy is not observable here. "
            f"(3) Use the price audit (`outputs/listing_price_audit.csv`) as a trigger: review first any listing more than 25% above benchmark, because that is where demand falls; for listings far below benchmark, test gradual increases, since the deepest discounts show no extra demand. "
            f"(4) Spend on capacity and listing type rather than on small amenities: they dominate the model.")
bullet(doc, f"**Investors.** Choose the segment first: whole units for groups (median {N(max(seg, key=lambda r: r['median_price'])['median_price'])} HKD) and short-stay rooms behave very differently from the monthly-room segment (median {N(min(seg, key=lambda r: r['median_price'])['median_price'])} HKD), "
            f"and professional operators dominate the latter. District matters, but transit distance is not priced separately.")
bullet(doc, f"**Platforms and policy makers.** {P(E['long_stay_share'])} of listings are in practice monthly rentals. Because the licence field is empty, regulators and platforms cannot see licensed versus unlicensed supply in public data; publishing it would make the market auditable.")

# ============================ 7. LIMITATIONS ============================
heading(doc, "7. Limitations and ethical considerations", 1)
bullet(doc, "**Asking price, one snapshot.** The price is a quote for one stay on one date, not a transaction or revenue; seasonality beyond weekend/weekday is not modelled, and listings without a price (mostly unbookable) and above the 99.5th percentile are out of scope.")
bullet(doc, "**Association, not causation.** Premiums and the price-gap→demand result are observational; unobserved quality (photos, view, floor) may drive both price and demand. Reviews are only a proxy for bookings.")
bullet(doc, "**Data gaps and proxies.** Licence status is empty; straight-line distances ignore hills and routes; MTR data come from a volunteer map; thin districts have few listings.")
bullet(doc, "**Explanations are model-based.** Permutation importance and SHAP values describe what the model uses, not what causes the price, and correlated factors share importance; the language-model wording is checked against the supplied numbers but remains generated text, so the template fallback exists and the explanation should be read as support, not advice.")
bullet(doc, "**Ethics and privacy.** We use public, aggregated listing data released by Inside Airbnb; no host names, photos or reviewer identities are used or reproduced, and the audit file contains only listing ids. A price benchmark can homogenise prices if everyone follows it; it should inform, not replace, a host's judgement.")

# ============================ 8. CONCLUSION ============================
heading(doc, "8. Conclusion", 1)
para(doc, f"A machine-learning benchmark built in Python can price Hong Kong rentals {P(M['mae_reduction'])} more accurately than comparable-listings pricing on unseen hosts, and – more importantly for decisions – it explains the market: "
          f"stay length and the type and size of the listing matter most; transit proximity and location details add little; and the commercial risk lies in pricing *above* the benchmark, not below it. "
          f"The next steps are to add transaction or occupancy data to move from asking prices to revenue, and to refresh the model periodically.", align="justify")

# ============================ REFERENCES ============================
heading(doc, "References", 2)
for ref in ["Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.",
            "Hong Kong SAR Government. Hotel and Guesthouse Accommodation Ordinance (Cap. 349).",
            "Inside Airbnb (2026). Hong Kong listings and reviews data, scraped 25 September 2026. http://insideairbnb.com",
            "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.",
            "OpenStreetMap contributors. Railway station data retrieved via the Overpass API. https://www.openstreetmap.org",
            "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.",
            "Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure competition. *Journal of Political Economy, 82*(1), 34–55.",
            "Xu, J. J., Chen, D., Chau, M., Li, L., & Zheng, H. (2022). Peer-to-peer loan fraud detection: Constructing features from transaction data. *MIS Quarterly, 46*(3), 1777–1792."]:
    para(doc, ref, size=8.5, after=1)

# ============================ APPENDIX ============================
page_break(doc)
heading(doc, "Appendix A – Reproducibility and hyper-parameters", 1)
para(doc, f"**Files.** `Code GroupName IS3240.ipynb` runs end-to-end (data in `data/`; MTR stations cached in `data/mtr_stations.csv`; results in `outputs/`). Random seed {R['meta']['random_state']}; host-split seed {R['split']['seed']}. "
          f"Libraries: pandas, NumPy, scikit-learn, statsmodels, shap, matplotlib, requests. The plain-language explanations use the OpenAI API when the environment variable OPENAI_API_KEY (or a git-ignored .env file) is set; without a key the notebook uses a deterministic template and still runs end to end.", size=9, align="justify")
bp = M["best_params"]
rows = [["Model", "Tuned hyper-parameters (grid search, 5-fold host-grouped CV on training data)", "CV MAE (log)"]]
for nm in ["Linear regression", "Random forest", "Gradient boosting"]:
    rows.append([nm, ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in bp[nm].items()), f"{M['cv_mae_log'][nm]:.3f}"])
table(doc, rows, [3.0, 12.2, 1.8], size=8, align=["l", "l", "r"])
heading(doc, "Appendix B – Accuracy by group (test set, selected model)", 1)
rows = [["Group", "n", "Median price", "MAE (HKD)", "Median % error", "Within ±20%"]]
for k, v in M["by_regime_room"].items():
    if v["n"] >= 20: rows.append([k, N(v["n"]), N(v["median_price"]), N(v["MAE"]), P(v["MedAPE"]), P(v["Within20"])])
for k, v in M["by_segment"].items():
    rows.append([k, N(v["n"]), N(v["median_price"]), N(v["MAE"]), P(v["MedAPE"]), P(v["Within20"])])
table(doc, rows, [6.6, 1.3, 2.4, 2.2, 2.5, 2.0], size=8, align=["l", "r", "r", "r", "r", "r"])
heading(doc, "Appendix C – Price map for monthly stays", 1)
figure(doc, FIG / "fig2c_price_map_monthly.png", 14.6, "**Figure A1 – Where whole flats are expensive (red) or cheap (blue) for monthly stays (28+ nights).**")
out = ROOT / OUT_NAME
doc.save(out)
print("saved", out)
