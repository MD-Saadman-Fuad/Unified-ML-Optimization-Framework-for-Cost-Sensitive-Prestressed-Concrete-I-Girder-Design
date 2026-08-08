# What Are We Actually Building?
### A Plain-English Guide to the Prestressed Concrete I-Girder ML Project

---

## The Engineering World We Work In

Imagine you are a civil engineer and your job is to design a concrete bridge. The bridge has beams running along its length — these are called **I-girders** (named after the letter "I" because of their shape: wide at the top and bottom, narrow in the middle).

Before you can build anything, you need to decide things like:

- How **deep** should the beam be?
- How many **beams** do you need across the width of the bridge?
- How many **steel wires** (called strands) do you need inside the beam to hold it together under load?
- Where should those wires be **positioned**?

These are not guesses. Engineers run **mathematical optimization programs** that test thousands of combinations and find the one design that is structurally safe AND costs the least money.

---

## The Problem

The cost of materials — concrete, steel rebar, and prestressing strands — changes constantly with the market.

Every time prices change, the engineer has to **re-run the entire optimization program from scratch**. This can take a long time and is expensive to do repeatedly.

> **Think of it like this:** Every time the price of flour changes, you have to completely re-bake and re-taste 1,000 versions of a cake just to figure out the cheapest good-tasting recipe. That is slow, tedious, and wasteful.

---

## Our Solution

We ran that slow optimization program across **670 different design scenarios** — covering different combinations of material prices and bridge lengths — and recorded all the results in a spreadsheet.

These 670 scenarios cover:
- **3 price levels for concrete** (low, medium, high — measured in dollars per cubic yard)
- **3 price levels for prestressing strands** (low, medium, high — measured in dollars per pound)
- **4 price levels for steel rebar** (low, medium, high, and an extra level — measured in dollars per pound)
- **5 different bridge span lengths** (100 ft, 120 ft, 140 ft, 160 ft, and 180 ft)
- **5 separate optimization runs** per combination (to account for natural variation in the solver results)

Then we feed all those results into a **machine learning model**.

The model studies the patterns and essentially "memorizes" the relationship between:

| What Goes In | What Comes Out |
|---|---|
| Price of concrete (per cubic yard) | Recommended beam depth (in inches) |
| Price of steel rebar (per pound) | Lateral spacing between beams (in feet) |
| Price of prestressing strands (per pound) | Number of beams needed |
| Length of the bridge span (in feet) | Depth and width of the bottom flange (in inches) |
| | Number of steel strands per beam |
| | Position of the harping point (where strands change angle) |

After training, when someone gives the model **new prices**, it can instantly predict the best design — in milliseconds — without running the slow optimization again.

> **Analogy:** Instead of baking 1,000 cakes every time flour prices change, you study all the past results, learn the pattern, and from then on you can just *look up* the best recipe almost instantly.

---

## What Does the Final Product Look Like?

A **web page** where a bridge engineer can:

1. Type in today's prices for concrete, steel rebar, and prestressing strands
2. Type in the length of the bridge span they need to design
3. Click a button
4. **Instantly see** the recommended beam dimensions, number of beams, and number of strands — all calculated by the trained model
5. See a **live drawing** of the I-beam cross-section that updates in real-time as the numbers change

No specialist software. No waiting. Just open a browser, enter the prices, get the answer.

---

## Why Does This Matter?

Right now, engineers either use expensive specialized software or spend significant time re-running calculations every time material market prices shift.

This tool makes that process **instant and accessible**, and could be used directly:

- On a **construction site** where a contractor needs a quick estimate
- In a **procurement meeting** where cost decisions are being made in real time
- In a **university or research setting** for studying how material costs affect structural design choices

It bridges the gap between **structural engineering knowledge** and **modern AI tools** to make design work faster, cheaper, and more accessible to everyone involved in a project — not just the specialists.

---

## A Simple Summary in One Sentence

> We trained a computer to instantly recommend the most cost-efficient bridge beam design for any given set of material prices, so engineers never have to run slow optimization software again.
