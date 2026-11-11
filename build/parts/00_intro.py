# %% [markdown]
# # What should a Hong Kong host charge? A data-driven pricing benchmark for short- and mid-term rentals
# **IS3240 Advanced Business Programming in Python – Group Project (code notebook)**
#
# ---
# ## 1. The business problem
#
# Hong Kong has one of the most expensive and most regulated accommodation markets in the world. Operators of Airbnb-type
# listings – from individual hosts renting a spare room to professional property managers running dozens of flats – face
# the same recurring decision: **"what nightly price should I put on this listing?"**
#
# * **Too high** and the listing sits empty; **too low** and the operator gives revenue away on every night booked.
# * Pricing is hard because the market is *two markets in one*: Hong Kong law restricts renting out residential premises for
#   fewer than 28 consecutive days without a licence, so a large share of listings are actually **monthly rentals** whose
#   nightly price is a fraction of a tourist stay.
# * Price depends on many things at once – location, transit access, size, amenities, who the host is – and hosts usually
#   fall back on eyeballing a few "comparable" listings.
#
# **Our objective:** use machine learning in Python to build a *pricing benchmark* that (i) predicts a fair asking price and a
# plausible price band for a listing from information available **before the first guest arrives**, (ii) explains what drives
# price and by how much, (iii) flags listings that are priced far from the benchmark, and (iv) tests whether being priced above
# the benchmark is associated with weaker guest demand.
#
# **Who uses it?** *Hosts and property managers* (set / review prices, decide which upgrades are worth it), *investors* (which
# segments and districts are attractive) and, indirectly, *platforms and policy makers* (how big the monthly-rental segment is).
#
# ## 2. Analytical roadmap (follows the "story-telling" framework from Week 5)
# | Step | Question | Notebook section |
# |---|---|---|
# | Problem & motivation | Why does price benchmarking matter here? | 1 |
# | Data | What do we have, how good is it, what did we remove and why? | 2 – 3 |
# | Market understanding | What does the market look like (stay regimes, geography)? | 4 – 5 |
# | Useful factors | What factors might explain price, why, and how are they computed? | 6 – 7 |
# | Market segments | Which natural segments exist? (unsupervised learning, KMeans + PCA) | 8 |
# | Suitable models | Which models, evaluated how, and why? (supervised learning) | 9 – 11 |
# | Excellent performance | How good is the benchmark for a real user? | 11 |
# | Insight | What drives price? Who is over/under-priced? Does it cost demand? | 12 – 14 |
# | Action | How would a host use it? What should they do? | 15 – 16 |
#
# **Data:** Inside Airbnb – Hong Kong (listings detailed file, scraped 25 Sep 2026; reviews file), plus **MTR station
# locations from OpenStreetMap** (via the Overpass API, a web-data-collection step as in Week 4).
#
# **How to run:** place the three files in `data/` and run all cells (`Kernel → Restart & Run All`). The MTR stations are
# cached in `data/mtr_stations.csv`, so the notebook also runs offline. Every figure and number quoted in the report is
# produced here and saved to `outputs/`.
