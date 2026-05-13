"""
ModelServe — Feast Feature Definitions
"""

from datetime import timedelta
from pathlib import Path
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FEATURES_PATH = str(_PROJECT_ROOT / "training" / "features.parquet")

cc_num_entity = Entity(
    name="cc_num",
    join_keys=["cc_num"],
    description="Credit card number",
)

fraud_features_source = FileSource(
    name="fraud_features_source",
    path=_FEATURES_PATH,
    timestamp_field="event_timestamp",
)

fraud_features_view = FeatureView(
    name="fraud_features",
    entities=[cc_num_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="merchant", dtype=Int64),
        Field(name="category", dtype=Int64),
        Field(name="gender", dtype=Int64),
        Field(name="state", dtype=Int64),
        Field(name="amt", dtype=Float64),
        Field(name="hour", dtype=Float64),
        Field(name="day_of_week", dtype=Float64),
        Field(name="month", dtype=Float64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="time_of_day", dtype=Int64),
        Field(name="amt_log", dtype=Float64),
        Field(name="amt_squared", dtype=Float64),
        Field(name="amt_x_category", dtype=Float64),
        Field(name="amt_x_merchant", dtype=Float64),
        Field(name="merchant_avg_amt", dtype=Float64),
        Field(name="merchant_std_amt", dtype=Float64),
        Field(name="category_avg_amt", dtype=Float64),
        Field(name="category_std_amt", dtype=Float64),
    ],
    source=fraud_features_source,
    online=True,
)