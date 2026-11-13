# %% [markdown]
# # What should a Hong Kong host charge? A data-driven pricing benchmark for short- and mid-term rentals
# **IS3240 Advanced Business Programming in Python – Group Project (code notebook)**
#
# ## 1. The business problem
#
# Hong Kong has one of the most expensive and most regulated accommodation markets in the world. Operators of Airbnb-type
# listings – from individual hosts renting a spare room to professional property managers running dozens of flats – face
# the same recurring decision: **"what nightly price should I put on this listing?"**
#
# * **Too high** and the listing sits empty; **too low** and the operator gives revenue away on every night booked.
# * The market is *two markets in one*: Hong Kong law restricts renting out residential premises for fewer than 28 consecutive days
#   without a licence, so a large share of listings are in practice **monthly rentals** whose nightly price is a fraction of a tourist stay.
# * Price depends on many things at once (location, transit, size, amenities, who the host is) and hosts usually fall back on
#   looking at a few "comparable" listings.
#
# **Our objective:** use machine learning in Python to build a *pricing benchmark* that (i) predicts a fair asking price and a plausible
# price band from information available **before the first guest arrives**, (ii) explains what drives price and by how much, (iii) flags
# listings priced far from the benchmark and tests whether this is associated with weaker demand, and (iv) explains each price in plain language.
# The users are *hosts and property managers*, *investors* (which segments and districts to choose) and, indirectly, *platforms and policy makers*.
#
# ## Roadmap (follows the "story-telling" framework of Week 5)
# | Step | Question | Section |
# |---|---|---|
# | Problem and data | Why does it matter, what data do we have, what did we remove and why? | 1 – 3 |
# | Market understanding | What does the market look like (stay types, geography, price map)? | 4 – 5 |
# | Useful factors | Which factors might explain the price, why, and how are they computed? | 6 |
# | Market segments | Which natural segments exist? (unsupervised learning) | 7 |
# | Suitable models and performance | Which models, evaluated how, and how good is the benchmark? | 8 |
# | Insight | Which factors matter? Who is over- or under-priced? | 9 – 11 |
# | Action | How would a host use it, and why is the price what it is? | 12 – 13 |
#
# **Data:** Inside Airbnb – Hong Kong (detailed listings and reviews, scraped 25 Sep 2026) plus the **MTR station locations from
# OpenStreetMap** (Overpass API, a web-data-collection step as in Week 4).
#
# **How to run:** place the data files in `data/` and run all cells (*Restart & Run All*). The MTR stations are saved in
# `data/mtr_stations.csv`, so the notebook also runs offline. Every figure and number quoted in the report is produced here and saved in `outputs/`.
