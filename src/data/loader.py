"""Data loading utility for the StreamFlix subscriber dataset (v3 schema)."""
from pathlib import Path
import pandas as pd


EXPECTED_COLUMNS = {
    # Identity & demographics
    "subscriber_id", "tenure_months", "plan_tier", "billing_cycle",
    "country", "payment_method",
    # Account state
    "auto_renew", "multi_profile", "promo_active",
    # Engagement
    "engagement_cohort",
    "watch_hours_last_7d", "watch_hours_last_30d", "watch_hours_last_90d",
    "distinct_titles_7d", "distinct_titles_30d", "distinct_titles_90d",
    "days_since_last_login", "logins_last_30d",
    # Support
    "support_tickets_7d", "support_tickets_30d", "support_tickets_90d",
    # Billing health
    "payment_failures_30d", "payment_failures_90d", "payment_failures_180d",
    # Lifecycle
    "days_since_plan_change", "days_until_promo_expires",
    # Economics & target
    "monthly_revenue", "churned_next_30d",
}




def load_subscribers(path: str | Path = "data/subscribers.csv") -> pd.DataFrame:
    """Load the synthetic StreamFlix subscriber dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it with: python src/data/simulate.py"
        )
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df
