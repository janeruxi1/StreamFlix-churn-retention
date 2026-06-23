"""
Phase 2 -- EDA + Survival Analysis
====================================

With the data validated in Phase 1, we now look for the *patterns* that
the model will need to capture and the *segments* that matter for the
cost-aware decision rule downstream.

Sections:
    A. Population overview          -- who is in the dataset
    B. Churn rates by categorical segment
    C. Kaplan-Meier survival curves -- overall + by plan + by billing
    D. Hazard by tenure month       -- m2 / m12 spike visualization
    E. Engagement-trend distribution -- churners vs. non-churners
    F. Tenure x cohort heatmap      -- 2D segment view
    G. Phase 2 verdict / handoff to Phase 3

All figures saved under reports/figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import load_subscribers

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = load_subscribers("data/subscribers.csv")


# =====================================================================
# A. Population overview
# =====================================================================
print("=" * 70)
print("A. POPULATION OVERVIEW")
print("=" * 70)
for col in ["plan_tier", "billing_cycle", "country", "payment_method",
            "engagement_cohort", "auto_renew", "multi_profile", "promo_active"]:
    if df[col].dtype == bool:
        share = df[col].mean()
        print(f"  {col:<22}  True share = {share:.1%}")
    else:
        top_shares = df[col].value_counts(normalize=True).round(3)
        print(f"  {col}:")
        for k, v in top_shares.items():
            print(f"      {k:<15} {v:.1%}")


# =====================================================================
# B. Churn rate by categorical segment
# =====================================================================
print("\n" + "=" * 70)
print("B. CHURN RATE BY CATEGORICAL SEGMENT")
print("=" * 70)

seg_cols = ["plan_tier", "billing_cycle", "country", "payment_method",
            "engagement_cohort", "auto_renew", "multi_profile"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
overall_rate = df["churned_next_30d"].mean()

for ax, col in zip(axes, seg_cols):
    seg = df.groupby(col)["churned_next_30d"].agg(["mean", "count"]).reset_index()
    seg = seg.sort_values("mean", ascending=False)
    bars = ax.bar(seg[col].astype(str), seg["mean"] * 100,
                  color="#5B8FF9", alpha=0.85, edgecolor="white")
    ax.axhline(overall_rate * 100, color="#F6AD55", linestyle="--",
               linewidth=1.2, label=f"overall {overall_rate:.1%}")
    for b, rate, n in zip(bars, seg["mean"], seg["count"]):
        ax.text(b.get_x() + b.get_width() / 2,
                rate * 100 + 0.1,
                f"{rate:.1%}\n(n={n:,})",
                ha="center", fontsize=8)
    ax.set_title(col, fontweight="bold", fontsize=10)
    ax.set_ylabel("churn rate (%)", fontsize=8)
    ax.set_ylim(0, max(seg["mean"]) * 100 * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_visible(False)
plt.suptitle("Churn rate by segment (vs overall baseline)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_churn_by_segment.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_churn_by_segment.png")

# Print the segments that are >= 1.5x the baseline (high-risk)
print(f"\nOverall churn rate: {overall_rate:.2%}")
print("Segments with churn rate >= 1.5x baseline:")
for col in seg_cols:
    seg = df.groupby(col)["churned_next_30d"].mean()
    for seg_val, rate in seg.items():
        if rate >= 1.5 * overall_rate:
            print(f"  {col:<20} = {seg_val:<15}  rate = {rate:.2%}")


# =====================================================================
# C. Kaplan-Meier survival curves
# =====================================================================
print("\n" + "=" * 70)
print("C. KAPLAN-MEIER SURVIVAL CURVES")
print("=" * 70)


def km(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Simple Kaplan-Meier estimator -- no lifelines dependency.

    durations : tenure_months at observation
    events    : 1 if churned in the next 30 days, 0 if censored (still active)
    """
    durations = np.asarray(durations)
    events = np.asarray(events)
    event_times = np.sort(np.unique(durations[events == 1]))
    rows = []
    s = 1.0
    for t in event_times:
        d = ((durations == t) & (events == 1)).sum()
        n = (durations >= t).sum()
        if n == 0:
            break
        s *= (1 - d / n)
        rows.append((t, s, n, d))
    return pd.DataFrame(rows, columns=["t", "S", "at_risk", "events"])


# Overall + by plan + by billing
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
overall = km(df["tenure_months"], df["churned_next_30d"])
ax.step(overall["t"], overall["S"], where="post", color="#5B8FF9", linewidth=2)
ax.set_title("Overall survival (S(t))", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylabel("P(still subscribed)")
ax.set_ylim(0.5, 1.01)
ax.grid(True, linestyle="--", alpha=0.4)

ax = axes[1]
plan_colors = {"Basic": "#F6735B", "Standard": "#F6BD16", "Premium": "#5AD8A6"}
for plan, color in plan_colors.items():
    sub = df[df["plan_tier"] == plan]
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{plan} (n={len(sub):,})")
ax.set_title("Survival by plan tier", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

ax = axes[2]
bill_colors = {"monthly": "#F6735B", "annual": "#5AD8A6"}
for billing, color in bill_colors.items():
    sub = df[df["billing_cycle"] == billing]
    curve = km(sub["tenure_months"], sub["churned_next_30d"])
    ax.step(curve["t"], curve["S"], where="post", color=color, linewidth=2,
            label=f"{billing} (n={len(sub):,})")
ax.set_title("Survival by billing cycle", fontweight="bold")
ax.set_xlabel("tenure (months)")
ax.set_ylim(0.5, 1.01)
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Kaplan-Meier Survival Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_kaplan_meier.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_kaplan_meier.png")


# =====================================================================
# D. Hazard rate by tenure month
# =====================================================================
print("\n" + "=" * 70)
print("D. HAZARD RATE BY TENURE MONTH (m2/m12 spike check)")
print("=" * 70)

tenure_hazard = (
    df.groupby("tenure_months")["churned_next_30d"]
      .agg(["mean", "count"])
      .reset_index()
      .rename(columns={"mean": "hazard"})
)

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(tenure_hazard["tenure_months"], tenure_hazard["hazard"] * 100,
       color="#5B8FF9", alpha=0.85, edgecolor="white", width=0.85)
ax.axhline(overall_rate * 100, color="gray", linestyle="--",
           linewidth=1, label=f"overall {overall_rate:.1%}")
ax.axvline(2, color="#F6AD55", linestyle="--", linewidth=1.5,
           label="m2 trial drop")
ax.axvline(12, color="#F6AD55", linestyle="--", linewidth=1.5,
           label="m12 anniversary")
ax.set_xlabel("tenure (months)")
ax.set_ylabel("hazard = 30-day churn rate (%)")
ax.set_title("Hazard rate by tenure month -- m2 and m12 spikes",
             fontweight="bold")
ax.set_xlim(-0.5, 36)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_tenure_hazard.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_tenure_hazard.png")


# =====================================================================
# E. Engagement-trend distribution: churners vs non-churners
# =====================================================================
print("\n" + "=" * 70)
print("E. ENGAGEMENT TREND -- churners vs non-churners")
print("=" * 70)

# Trend ratio: scale watch_hours_last_7d up to a 30-day equivalent (× 30/7)
# then divide by watch_hours_last_30d. Values < 1 = declining; >1 = growing.
df["watch_trend_7d_to_30d"] = (
    df["watch_hours_last_7d"] * (30 / 7) /
    df["watch_hours_last_30d"].clip(lower=0.1)
).clip(upper=3.0)

print("\nTrend ratio statistics (watch_hours_7d × 30/7 / watch_hours_30d):")
print(df.groupby("churned_next_30d")["watch_trend_7d_to_30d"]
        .describe(percentiles=[0.25, 0.5, 0.75]).round(3))

fig, ax = plt.subplots(figsize=(10, 5))
for outcome, color, label in [
    (0, "#5AD8A6", "did NOT churn"),
    (1, "#F6735B", "churned"),
]:
    vals = df.loc[df["churned_next_30d"] == outcome, "watch_trend_7d_to_30d"]
    ax.hist(vals, bins=50, alpha=0.55, label=f"{label} (n={len(vals):,})",
            color=color, edgecolor="white", density=True)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="flat (=1)")
ax.set_xlabel("watch_trend_7d_to_30d  (<1 = declining, >1 = growing)")
ax.set_ylabel("density")
ax.set_title("Engagement trend by churn outcome", fontweight="bold")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_engagement_trend.png", dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_engagement_trend.png")


# =====================================================================
# F. Tenure x engagement-cohort heatmap
# =====================================================================
print("\n" + "=" * 70)
print("F. CHURN HEATMAP: tenure bucket x engagement cohort")
print("=" * 70)

df["tenure_bucket"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 1, 2, 5, 11, 13, 24, 60],
    labels=["m0-1", "m2 (trial)", "m3-5", "m6-11", "m12-13 (anniv)",
            "m14-24", "m25+"],
)
heatmap = df.pivot_table(
    index="tenure_bucket", columns="engagement_cohort",
    values="churned_next_30d", aggfunc="mean", observed=True,
)
cohort_order = ["heavy", "regular", "casual"]
heatmap = heatmap[cohort_order]

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heatmap.values * 100, cmap="RdYlGn_r", aspect="auto",
               vmin=0, vmax=heatmap.values.max() * 100)
ax.set_xticks(range(len(heatmap.columns)))
ax.set_xticklabels(heatmap.columns)
ax.set_yticks(range(len(heatmap.index)))
ax.set_yticklabels(heatmap.index)
ax.set_xlabel("engagement cohort")
ax.set_ylabel("tenure bucket")
ax.set_title("Churn rate (%) by tenure × engagement cohort",
             fontweight="bold")
for i in range(heatmap.shape[0]):
    for j in range(heatmap.shape[1]):
        ax.text(j, i, f"{heatmap.values[i, j] * 100:.1f}%",
                ha="center", va="center", fontsize=10,
                color="white" if heatmap.values[i, j] * 100 > 6 else "black")
plt.colorbar(im, ax=ax, label="churn rate (%)")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_tenure_cohort_heatmap.png",
            dpi=140, bbox_inches="tight")
print(f"Saved -> {FIG_DIR}/02_tenure_cohort_heatmap.png")
print(f"\nHeatmap values:\n{(heatmap * 100).round(2)}")


# =====================================================================
# G. Verdict / Phase 3 setup
# =====================================================================
print("\n" + "=" * 70)
print("G. PHASE 2 VERDICT")
print("=" * 70)

# Key findings summary
trend_decline = (df.loc[df["churned_next_30d"] == 1, "watch_trend_7d_to_30d"]
                   .median())
trend_stable = (df.loc[df["churned_next_30d"] == 0, "watch_trend_7d_to_30d"]
                  .median())
spread_pp = (
    df.groupby("plan_tier")["churned_next_30d"].mean().max() -
    df.groupby("plan_tier")["churned_next_30d"].mean().min()
) * 100
mobile_signal = heatmap.loc["m2 (trial)", "casual"] - heatmap.loc["m25+", "heavy"]

print(f"\nKey patterns for Phase 3 feature engineering:")
print(f"  - Plan tier spread:        {spread_pp:.2f}pp churn-rate difference")
print(f"  - Engagement trend median: churners {trend_decline:.3f} vs "
      f"non-churners {trend_stable:.3f}  (<1 = declining)")
print(f"  - Tenure × cohort max gap: m2 casual ({heatmap.loc['m2 (trial)', 'casual']*100:.1f}%) "
      f"vs m25+ heavy ({heatmap.loc['m25+', 'heavy']*100:.1f}%) "
      f"= {mobile_signal*100:.1f}pp spread")
print(f"\nReady for Phase 3 (feature engineering).")
