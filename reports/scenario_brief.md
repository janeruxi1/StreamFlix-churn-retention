# 📝 PM Brief — Subscriber Retention Program

**From:** Marcus Lee, Sr. PM, Subscriber Retention
**To:** Data Science Team
**Re:** Cost-aware churn-targeting model for the monthly paid tier

---

## Background

StreamFlix's paid subscription business has ~2.1M active monthly subscribers across three tiers (Basic, Standard, Premium). **Monthly churn averages ~5.5%** but spikes meaningfully at the 2-month and 12-month tenure points (free-trial-to-paid drop-off and annual-anniversary reassessment).

Up to now, the Retention team has run a single broad-brush campaign — a generic $5 monthly credit offered to *any* subscriber who hits month 11 — which costs ~$280k/month and has unclear ROI because we never measure who would have stayed anyway.

The team wants to **move from blanket campaigns to a cost-aware, model-driven targeting system.**

## Business problem

> **For each subscriber, predict the probability they will churn in the next 30 days and recommend the cheapest intervention (if any) expected to retain them — within a fixed monthly budget — to maximize net dollar value to the business.**

Three decisions need to be made for each subscriber every month:
1. Are they at meaningful risk of churning?
2. If yes, which intervention should we offer?
3. What's the expected ROI if we do?

## Intervention menu

The Retention team has negotiated three pre-built interventions with the relevant ops teams:

| Intervention | Cost per offer | Estimated retention uplift (vs no offer) | Notes |
|---|---|---|---|
| **Curated playlist email** | $1 | +6pp | Lightweight; safe to offer anyone |
| **$5 monthly credit** | $5 | +14pp | The current default; assume real cost = $5 of LTV |
| **1-month free Premium upgrade** | $12 | +22pp | Best uplift but tier-upgrade pressure on revenue |

(These uplift estimates come from a prior A/B test the Growth team ran; the model should treat them as inputs, not learn them. For this project we'll use these published values as ground truth.)

## Economic parameters

| Parameter | Value | Notes |
|---|---|---|
| Average customer LTV (Basic) | $9 / month × 14 expected months retained | $126 / churn averted |
| Average customer LTV (Standard) | $14 / month × 16 expected months | $224 / churn averted |
| Average customer LTV (Premium) | $19 / month × 18 expected months | $342 / churn averted |
| Monthly retention budget | **$200,000** | Hard cap |
| Decision horizon | 30 days | Predict churn in next 30 days |

## Decision sought

> **Recommend a targeting policy** that maximizes expected net retention value subject to the $200k monthly budget. The policy must specify:
>
> 1. A risk-probability threshold above which we engage a subscriber
> 2. A rule for assigning the right intervention to each engaged subscriber (cheapest one expected to retain them)
> 3. Expected number of churns averted and net dollar value at the chosen threshold

## Stakeholders

- **Primary decision-maker:** Marcus Lee (PM, Retention)
- **Finance:** must sign off on expected ROI
- **Ops:** capacity-constrained on the Premium upgrade — cap at 10k offers/month
- **CX team:** wants to ensure we're not over-targeting one-time grievance churners

## What we want from the data science team

1. A **calibrated churn-probability model** for the next 30 days
2. **Explainability** — for each at-risk subscriber, why the model flagged them, in PM language
3. A **cost-aware decision rule** that picks the optimal intervention per subscriber
4. A **decision memo** with the recommended policy, expected savings, and sensitivity to budget/cost assumptions
5. A **Streamlit decision tool** the Retention team can use to score new subscriber profiles ad hoc

## Risks / open questions

- **Calibration matters more than discrimination.** A model that ranks well but is miscalibrated will mis-size the intervention budget. Calibration is a first-class metric.
- **Segments may need different policies.** New users (tenure < 3 months) may respond very differently to a curated playlist than tenured users.
- **Anchoring.** The Curated-playlist arm of the prior A/B test had a small treatment group (~3k); the +6pp uplift estimate carries wider uncertainty than the $5-credit and Premium-upgrade estimates.
