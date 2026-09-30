"""Small regression checks for the existing plugin expressions."""

import polars as pl
import pytest

from composer_alloc_polars import filter_select_weights, rolling_max_drawdown


def test_filter_select_weights_keeps_existing_order_and_tie_break():
    frame = pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 2.0], "c": [3.0, 1.0]})
    result = frame.select(
        filter_select_weights(pl.col("a"), pl.col("b"), pl.col("c"), n=1, reverse=False).alias("low"),
        filter_select_weights(pl.col("a"), pl.col("b"), pl.col("c"), n=1, reverse=True).alias("high"),
    )
    assert result["low"].to_list() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert result["high"].to_list() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]


def test_rolling_max_drawdown_keeps_window_behavior():
    frame = pl.DataFrame({"value": [100.0, 90.0, 95.0]})
    result = frame.select(rolling_max_drawdown(pl.col("value"), 2).alias("drawdown"))
    assert result["drawdown"][0] is None
    assert result["drawdown"][1:].to_list() == pytest.approx([10.0, 0.0], abs=1e-12)
