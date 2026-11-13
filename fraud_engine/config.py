"""Project-wide configuration: seeds, paths, column groups, cost scenarios."""
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 3

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transaction_fraud_risk_advanced_dataset.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SUPP = OUT / "supplementary"
for _p in (OUT, FIG, SUPP):
    _p.mkdir(parents=True, exist_ok=True)

TARGET = "fraud"
ID_COL = "transaction_id"

RAW_NUMERIC = [
    "hour", "day_of_week", "transaction_amount", "distance_from_usual_location_km",
    "transactions_last_1h", "transactions_last_24h", "account_age_days",
    "failed_logins_24h", "prior_chargebacks", "device_age_days", "merchant_risk_score",
    "ip_risk_score", "email_age_days", "login_to_transaction_minutes",
    "new_device", "billing_address_mismatch",
]
RAW_CATEGORICAL = [
    "card_presence", "channel", "country", "device_type", "merchant_category",
    "authentication_level", "customer_tier",
]
MISSING_NUMERIC = [
    "transaction_amount", "distance_from_usual_location_km", "merchant_risk_score",
    "ip_risk_score", "login_to_transaction_minutes",
]
# assumption: platform home market (largest share, 55%)
HOME_COUNTRY = "Sri Lanka"


@dataclass(frozen=True)
class CostScenario:
    """Hypothetical per-transaction cost assumptions (USD).

    missed fraud (approved)  : amount * loss_multiplier + chargeback_fee
    legit blocked            : block_fp_cost (customer friction / churn / support call)
    any block                : block_ops_cost (processing + notification)
    capacity limits          : max_review_rate / max_block_rate cap the share of volume that
                               may be reviewed / blocked (None = unconstrained)
    any manual review        : review_cost (analyst time) ; legit reviewed also
                               pays review_friction_cost (delay / abandonment)
    review catches fraud     : with probability review_catch_rate
    """
    name: str
    loss_multiplier: float = 1.0
    chargeback_fee: float = 15.0
    block_fp_cost: float = 20.0
    block_ops_cost: float = 1.0
    review_cost: float = 4.0
    review_friction_cost: float = 2.0
    review_catch_rate: float = 0.90
    # analyst capacity: at most 10% of volume can be reviewed
    max_review_rate: Optional[float] = 0.10
    # customer-experience guardrail: at most 5% of volume blocked
    max_block_rate: Optional[float] = 0.05

    def as_dict(self):
        return asdict(self)


SCENARIOS = [
    CostScenario("Base"),
    CostScenario("High fraud loss", loss_multiplier=1.5, chargeback_fee=40.0),
    CostScenario("Low fraud loss", loss_multiplier=0.6, chargeback_fee=5.0),
    CostScenario("Expensive false positives",
                 block_fp_cost=60.0, review_friction_cost=6.0),
    CostScenario("Cheap false positives", block_fp_cost=6.0,
                 review_friction_cost=1.0),
    CostScenario("Expensive reviews", review_cost=15.0),
    CostScenario("Cheap reviews", review_cost=1.5),
    CostScenario("Weak analysts", review_catch_rate=0.70),
    CostScenario("Review capacity 5%", max_review_rate=0.05),
    CostScenario("Review capacity 20%", max_review_rate=0.20),
    CostScenario("No capacity limits",
                 max_review_rate=None, max_block_rate=None),
]
BASE = SCENARIOS[0]
