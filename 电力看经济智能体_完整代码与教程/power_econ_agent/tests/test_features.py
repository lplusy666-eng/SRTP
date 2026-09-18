from __future__ import annotations

import numpy as np

from power_econ.config import load_settings
from power_econ.data.synthetic import SyntheticPowerEconomyGenerator
from power_econ.features.builder import FeatureBuilder


def test_feature_groups_and_publication_lag() -> None:
    settings = load_settings("configs/quick_test.yaml")
    raw = SyntheticPowerEconomyGenerator(settings).generate()
    features, spec = FeatureBuilder(settings).transform(raw)
    assert set(spec.groups) == {"load", "weather", "calendar", "macro", "event"}
    assert len(spec.feature_columns) == len(set(spec.feature_columns))
    assert np.isfinite(features[spec.feature_columns].to_numpy()).all()
    assert "industrial_yoy_published" in spec.groups["macro"]
    # With a one-month release lag, first-month forecast features are neutral and
    # do not read the same-month target value.
    first_month = features["timestamp"].dt.month.eq(features["timestamp"].dt.month.iloc[0])
    assert features.loc[first_month, "industrial_yoy_published"].nunique() == 1
    assert features.loc[first_month, "industrial_yoy_published"].iloc[0] == 0.0
