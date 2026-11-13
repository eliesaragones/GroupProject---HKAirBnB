# %% [markdown]
# ## 13. "Why this price?" – explaining each prediction in plain language
# A host who is told "1,030 HKD" immediately asks **why**. We answer in two steps, separating *computing* from *writing*:
# 1. **We compute the reasons with SHAP**, a standard explainability method for machine-learning models. SHAP splits the prediction of the selected model into one **contribution per
#    factor**: starting from the typical price of Hong Kong listings, each factor pushes this listing's price up or down, and the contributions add up exactly to the prediction.
#    We add the contributions of the factors of each business group (size, location & transit, amenities...), so that the result is readable for a host.
# 2. **A language model (OpenAI API, called with `requests` as in Week 4) writes the explanation** from those numbers only. It never sees our data or our model.
#
# *Risks of AI-generated text (Week 3).* The factors are given to the model in two lists (raising / lowering the price) and a **check** compares every number of its answer with the numbers
# we gave it; if one is not supported, a **fixed template** is used instead. Without an API key the template is used too, so the notebook always runs. The key is read from the
# environment (`OPENAI_API_KEY`) or a git-ignored `.env` file and is never written to the notebook or the repository.

# %%
def read_api_key():
    """The OpenAI key comes from the environment or a local .env file; it is never printed or saved."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None

OPENAI_KEY, OPENAI_MODEL = read_api_key(), os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
print("LLM explanations:", f"ON (model {OPENAI_MODEL})" if OPENAI_KEY else "OFF - no OPENAI_API_KEY found, the template is used")

# Business labels: the factors that a host recognises, in 9 groups
FACTOR_GROUPS = {
    "Stay length (minimum nights)": ["long_stay", "log_min_nights", "q_log_nights"], "Weekend check-in": ["q_weekend"], "Room type": ["room_type"],
    "Property type": ["property_type_grp"], "Size & capacity": ["accommodates", "bedrooms", "studio", "beds", "baths"], "Shared bathroom": ["bath_shared"],
    "Location & transit": ["district_grp", "region", "lat", "lon"] + FAMILIES["Transit & centrality"], "Local competition": FAMILIES["Local competition"],
    "Amenities": FAMILIES["Amenities"]}

import shap
EXPLAINER = shap.TreeExplainer(FINAL)                                        # SHAP for the selected tree model (contributions in log price)
START_PRICE = float(np.exp(np.ravel(EXPLAINER.expected_value)[0]))           # starting point: typical price of a Hong Kong listing in the model
GROUP_COLS = {label: family_columns(cols) for label, cols in FACTOR_GROUPS.items()}      # model columns of every business group

def factor_effects(row_df):
    """SHAP contribution of every factor group, as the % by which it raises or lowers the price (contributions of log price multiply)."""
    stay = int(row_df.long_stay.iloc[0])
    contributions = pd.Series(EXPLAINER.shap_values(encode(row_df))[0], index=X.columns)       # one number per model column
    effects = {label: float(np.exp(contributions[cols].sum()) - 1) for label, cols in GROUP_COLS.items()
               if not (label == "Weekend check-in" and stay == 1)}           # the check-in day only matters for short stays
    return pd.Series(effects).sort_values(key=abs, ascending=False)

# %%
def make_facts(listing, stay, price, lo, hi, effects):
    """The facts for the language model. The factors go in TWO lists, so that the direction of an effect cannot be confused."""
    top = effects[effects.abs() >= 0.03].head(6)             # effects of at least 3%, at most the 6 largest
    return {"listing": f"{listing['room_type']} in {listing['district']}, {listing['accommodates']} guests, " + ("monthly stay" if stay["long_stay"] else "short stay"),
            "benchmark_price_hkd_per_night": int(round(price, -1)), "band_80pct_hkd": [int(round(lo, -1)), int(round(hi, -1))],
            "typical_hong_kong_listing_price_hkd": int(round(START_PRICE, -1)),
            "note_on_comparison": "the typical Hong Kong listing is the starting point; the factors below explain why this listing is above or below it",
            "factors_associated_with_a_HIGHER_price": [{"factor": k, "higher_by_pct": int(round(100 * v))} for k, v in top.items() if v > 0] or "none",
            "factors_associated_with_a_LOWER_price": [{"factor": k, "lower_by_pct": int(round(-100 * v))} for k, v in top.items() if v < 0] or "none"}

def template_text(f):
    """Fallback without AI: a fixed sentence pattern filled with the numbers."""
    up, down = f["factors_associated_with_a_HIGHER_price"], f["factors_associated_with_a_LOWER_price"]
    text = (f"The benchmark for this listing ({f['listing']}) is about {f['benchmark_price_hkd_per_night']:,} HKD per night (80% band {f['band_80pct_hkd'][0]:,}-{f['band_80pct_hkd'][1]:,}), "
            f"compared with about {f['typical_hong_kong_listing_price_hkd']:,} HKD for the typical Hong Kong listing. ")
    if up != "none":
        text += "Associated with a higher price: " + "; ".join(f"{x['factor'].lower()} (+{x['higher_by_pct']}%)" for x in up) + ". "
    if down != "none":
        text += "Associated with a lower price: " + "; ".join(f"{x['factor'].lower()} (-{x['lower_by_pct']}%)" for x in down) + ". "
    return text + "These are statistical associations, not guarantees."

SYSTEM_PROMPT = ("You are a pricing analyst explaining a rental price benchmark to a Hong Kong host. Write 3-5 plain-English sentences (at most 120 words), no bullet points, no advice. "
                 "Use ONLY the facts and numbers in the JSON; never invent anything. Say the benchmark price and the 80% band, compare it with the typical Hong Kong listing, "
                 "then name the factors: those in the HIGHER list raise the price, those in the LOWER list lower it, with their percentages. "
                 "Use the words 'associated with', not 'causes'.")

def call_openai(facts):
    """Send the facts to the OpenAI API and return the answer (temperature 0 = as repeatable as possible)."""
    response = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {OPENAI_KEY}"}, timeout=45,
                             json={"model": OPENAI_MODEL, "temperature": 0, "max_tokens": 260,
                                   "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(facts)}]})
    response.raise_for_status()                                              # stops with an error if, e.g., the key is wrong
    return response.json()["choices"][0]["message"]["content"].strip()

def numbers_ok(text, facts):
    """True if every number of the text matches (within 3%, or +-1 for small values) a number of the facts we sent."""
    allowed = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", json.dumps(facts).replace(",", ""))] + [80.0]    # 80 = the "80% band"
    written = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]
    return all(any(abs(x - a) <= max(1.0, 0.03 * a) for a in allowed) for x in written)

# %%
def explain_price(listing, stay):
    """Benchmark price, band, reasons and explanation text for one listing."""
    row = make_row(**listing, **stay)
    price, lo, hi = benchmark(row)
    effects = factor_effects(row)
    facts = make_facts(listing, stay, price, lo, hi, effects)
    text, source = template_text(facts), "template"                          # by default: the template
    if OPENAI_KEY:
        try:                                                                 # try the language model; any failure falls back to the template
            answer = call_openai(facts)
            if numbers_ok(answer, facts):
                text, source = answer, f"OpenAI {OPENAI_MODEL}"
        except Exception as e:
            print("LLM call failed, template used:", type(e).__name__)
    return dict(price=price, lo=lo, hi=hi, effects=effects, text=text, source=source)

# Explain the three example listings (weekday short stay) and the Wan Chai flat as a monthly stay
cases = [(name, listing, STAYS["Short stay, weekday"]) for name, listing in PROFILES.items()] + [("Wan Chai flat, monthly stay", list(PROFILES.values())[0], STAYS["Monthly stay"])]
EXPLANATIONS = {name: explain_price(listing, stay) for name, listing, stay in cases}
for name, e in EXPLANATIONS.items():
    print(f"\n=== {name.replace(chr(10), ' ')} - {e['price']:,.0f} HKD (band {e['lo']:,.0f}-{e['hi']:,.0f}) [{e['source']}]\n{textwrap.fill(e['text'], 120)}")
RESULTS["explanations"] = dict(
    llm_enabled=bool(OPENAI_KEY), model=OPENAI_MODEL if OPENAI_KEY else None, start_price=START_PRICE,
    items=[dict(profile=name.replace("\n", " "), price=float(e["price"]), source=e["source"], text=e["text"])
           for name, e in EXPLANATIONS.items()])

# Figure 9: blue = factor groups that raise the price, orange = factor groups that lower it (relative to comparable listings); the 6 largest effects
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
for ax, name in zip(axes, list(PROFILES)[:2]):
    e = EXPLANATIONS[name]
    eff = e["effects"].head(6).sort_values()
    ax.barh(range(len(eff)), eff.values * 100, color=[PAL["blue"] if v > 0 else PAL["orange"] for v in eff.values], height=0.62)
    ax.set_yticks(range(len(eff))); ax.set_yticklabels(eff.index, fontsize=6.6); ax.grid(axis="y", visible=False)
    ax.set_xlim(-1.3 * np.abs(eff.values).max() * 100, 1.3 * np.abs(eff.values).max() * 100); ax.axvline(0, color=PAL["ink2"], lw=0.9); ax.set_xlabel("SHAP effect on the price (%)", fontsize=7)
    ax.set_title(f"{name.splitlines()[0]}: {e['price']:,.0f} HKD\n(typical Hong Kong listing: {START_PRICE:,.0f} HKD)", fontsize=7.8)
fig.suptitle("Why this price? SHAP contribution of each factor group, short weekday stay", x=0.01, ha="left", fontsize=9, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
savefig("fig9_explanation", fig)
save_results()

# %% [markdown]
# **Finding.** For each listing we can say *why* the benchmark is what it is: which factors push it above or below the typical Hong Kong listing, and by how much.
# The wording comes from a language model, but every number in it comes from SHAP and is verified before it is shown.
