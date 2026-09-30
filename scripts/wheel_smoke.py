"""Functional smoke test run against each installed release wheel."""

import polars as pl

from composer_alloc_polars import ema_with_validity, rsi_with_validity


frame = pl.DataFrame({"price": [1.0, 2.0, 3.0, 4.0], "valid": [True] * 4})
for expression, expected in [
    (ema_with_validity(pl.col("price"), pl.col("valid"), 2), [None, 1.5, 2.5, 3.5]),
    (rsi_with_validity(pl.col("price"), pl.col("valid"), 2), [None, None, 100.0, 100.0]),
]:
    for result in [frame.select(expression.alias("indicator")), frame.lazy().select(expression.alias("indicator")).collect()]:
        assert result.schema["indicator"] == pl.Struct({"value": pl.Float64, "valid": pl.Boolean})
        assert result["indicator"].struct.field("value").to_list() == expected
        assert result["indicator"].struct.field("valid").to_list() == [x is not None for x in expected]

poisoned = pl.DataFrame({"price": [1.0, 2.0, 3.0, 4.0], "valid": [True, True, False, True]})
for function in [ema_with_validity, rsi_with_validity]:
    rows = poisoned.select(function(pl.col("price"), pl.col("valid"), 2).alias("indicator"))["indicator"].to_list()
    assert rows[2:] == [{"value": None, "valid": False}] * 2
