"""Data loading utility for the StreamFlix subscriber dataset."""
from pathlib import Path
import pandas as pd


EXPECTED_COLUMNS = {
    "subscriber_id", "tenure_months", "plan_tier", "billing_cycle",
    "country", "payment_method", "auto_renew", "multi_profile",
    "avg_watch_hours_3mo", "distinct_titles_30d", "days_since_last_login",
    "support_tickets_90d", "past_payment_failures", "monthly_revenue",
    "churned_next_30d",
}


def load_subscribers(path: str | Path = "data/subscribers.csv") -> pd.DataFrame:
    """Load the synthetic StreamFlix subscriber dataset.

    Generate it first with `python src/data/simulate.py` if the file is missing.
    """
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
