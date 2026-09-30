"""Contract tests for the compiled Polars indicator expressions."""

import math
import random

import polars as pl
import pytest

import composer_alloc_polars as cap


def reference(values, period, kind):
    """Independent contiguous-input definitions of the specified SMA seeds."""
    result = [None] * len(values)
    if kind == "ema":
        if len(values) < period:
            return result
        previous = sum(values[:period]) / period
        result[period - 1] = previous
        alpha = 2 / (period + 1)
        for index in range(period, len(values)):
            previous = alpha * values[index] + (1 - alpha) * previous
            result[index] = previous
    else:
        if len(values) <= period:
            return result
        deltas = [new - old for old, new in zip(values, values[1:])]
        gains = [max(delta, 0) for delta in deltas]
        losses = [max(-delta, 0) for delta in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        def score():
            total = avg_gain + avg_loss
            return 0.0 if total == 0 else 100 * avg_gain / total

        result[period] = score()
        for index in range(period + 1, len(values)):
            avg_gain = (avg_gain * (period - 1) + gains[index - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[index - 1]) / period
            result[index] = score()
    return result


def evaluate(kind, values, validity, period, *, lazy=False, chunks=False):
    frame = pl.DataFrame({"value": values, "source_valid": validity})
    if chunks:
        split = len(values) // 2
        frame = frame.slice(0, split).vstack(frame.slice(split, len(values) - split), in_place=False)
        frame = frame.rechunk(False)
    expr = getattr(cap, f"{kind}_with_validity")(
        pl.col("value"), pl.col("source_valid"), period
    ).alias("indicator")
    result = frame.lazy().select(expr).collect() if lazy else frame.select(expr)
    assert result.schema["indicator"] == pl.Struct(
        {"value": pl.Float64, "valid": pl.Boolean}
    )
    rows = result["indicator"].to_list()
    for row in rows:
        assert row["valid"] is (row["value"] is not None)
        if row["valid"]:
            assert math.isfinite(row["value"])
    return rows


def assert_scores(rows, expected):
    assert len(rows) == len(expected)
    for row, want in zip(rows, expected):
        if want is None:
            assert row == {"value": None, "valid": False}
        else:
            assert row["valid"] is True
            assert row["value"] == pytest.approx(want, abs=1e-12, rel=1e-13)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_frozen_expected_vectors(kind):
    values = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
    expected = {
        "ema": [None, 1.5, 2.5, 13 / 6, 25 / 18, 97 / 54],
        "rsi": [None, None, 100.0, 50.0, 25.0, 62.5],
    }
    assert_scores(evaluate(kind, values, [True] * len(values), 2), expected[kind])


@pytest.mark.parametrize("period", [3, 14, 200])
def test_frozen_longer_period_seed_boundaries(period):
    values = [float(index) for index in range(period + 2)]
    valid = [True] * len(values)
    ema_expected = [None] * (period - 1) + [
        (period - 1) / 2,
        (period + 1) / 2,
        (period + 3) / 2,
    ]
    rsi_expected = [None] * period + [100.0, 100.0]
    assert_scores(evaluate("ema", values, valid, period), ema_expected)
    assert_scores(evaluate("rsi", values, valid, period), rsi_expected)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("period", [1, 2, 3, 14, 200])
@pytest.mark.parametrize("pattern", ["rising", "falling", "flat", "mixed", "random"])
def test_contiguous_values_match_independent_formula(kind, period, pattern):
    rng = random.Random(1977)
    walks = [50.0]
    for _ in range(209):
        walks.append(walks[-1] + rng.uniform(-3.0, 3.0))
    patterns = {
        "rising": [float(index) for index in range(210)],
        "falling": [float(210 - index) for index in range(210)],
        "flat": [7.0] * 210,
        "mixed": [100.0 + ((index * 17) % 29) for index in range(210)],
        "random": walks,
    }
    values = patterns[pattern]
    assert_scores(evaluate(kind, values, [True] * len(values), period), reference(values, period, kind))


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_seed_boundary_and_period_longer_than_series(kind):
    period = 3
    seed_length = period if kind == "ema" else period + 1
    values = [float(index) for index in range(seed_length)]
    assert_scores(evaluate(kind, values, [True] * len(values), period), reference(values, period, kind))
    assert_scores(evaluate(kind, values, [True] * len(values), 200), [None] * len(values))


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_leading_numeric_invalidity_recovers(kind, bad):
    suffix = [1.0, 2.0, 3.0, 4.0]
    rows = evaluate(kind, [bad, *suffix], [True] * 5, 2)
    assert_scores(rows[:1], [None])
    assert rows[1:] == evaluate(kind, suffix, [True] * 4, 2)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_finite_synthetic_prefix_and_null_validity_recover(kind):
    suffix = [1.0, 2.0, 3.0, 4.0]
    rows = evaluate(kind, [999.0, 888.0, *suffix], [False, None, True, True, True, True], 2)
    assert_scores(rows[:2], [None, None])
    assert rows[2:] == evaluate(kind, suffix, [True] * 4, 2)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_interior_numeric_invalidity_is_absorbing(kind, bad):
    rows = evaluate(kind, [1.0, 2.0, 3.0, bad, 4.0, 5.0], [True] * 6, 2)
    assert_scores(rows[3:], [None, None, None])


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("invalid", [False, None])
@pytest.mark.parametrize("position", [1, 4])
def test_interior_provenance_failure_is_absorbing(kind, invalid, position):
    validity = [True] * 7
    validity[position] = invalid
    rows = evaluate(kind, [float(i) for i in range(7)], validity, 3)
    assert_scores(rows[position:], [None] * (7 - position))


def test_flat_rsi_zero_is_valid_and_invalid_zero_is_null():
    rows = evaluate("rsi", [3.0, 3.0, 3.0, 3.0], [True] * 4, 2)
    assert rows[2] == {"value": 0.0, "valid": True}
    rows = evaluate("rsi", [3.0, 3.0, 3.0, 3.0], [True, True, False, True], 2)
    assert_scores(rows, [None] * 4)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_wrapper_casts_numeric_and_validity_inputs(kind):
    fn = getattr(cap, f"{kind}_with_validity")
    frame = pl.DataFrame({"value": [1, 2, 3, 4], "source_valid": [1, 1, 1, 1]})
    result = frame.select(fn(pl.col("value"), pl.col("source_valid"), 2).alias("result"))
    assert_scores(result["result"].to_list(), reference([1.0, 2.0, 3.0, 4.0], 2, kind))


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_struct_can_be_cached_and_fields_extracted(kind):
    fn = getattr(cap, f"{kind}_with_validity")
    frame = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "source_valid": [True] * 4})
    result = (
        frame.lazy()
        .with_columns(fn(pl.col("value"), pl.col("source_valid"), 2).alias("indicator"))
        .select(
            pl.col("indicator").struct.field("value").alias("score"),
            pl.col("indicator").struct.field("valid").alias("ok"),
        )
        .collect()
    )
    assert result["score"].to_list() == reference([1.0, 2.0, 3.0, 4.0], 2, kind)
    assert result["ok"].to_list() == [score is not None for score in result["score"]]


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_empty_schema_and_field_extraction(kind):
    fn = getattr(cap, f"{kind}_with_validity")
    expr = fn(pl.col("value"), pl.col("source_valid"), 2)
    for size in [0, 5]:
        frame = pl.DataFrame({"value": [1.0] * size, "source_valid": [True] * size})
        result = frame.select(expr.alias("result"))
        assert result.schema["result"] == pl.Struct({"value": pl.Float64, "valid": pl.Boolean})
        extracted = frame.select(
            expr.struct.field("value").alias("score"),
            expr.struct.field("valid").alias("valid"),
        )
        assert extracted.schema == {"score": pl.Float64, "valid": pl.Boolean}


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("chunks", [False, True])
def test_multiple_aliases_and_indicators_in_one_frame(lazy, chunks):
    values = [float(i) for i in range(12)]
    frame = pl.DataFrame({"price": values, "ok": [True] * len(values)})
    if chunks:
        frame = frame.slice(0, 5).vstack(frame.slice(5), in_place=False)
        assert frame.get_column("price").n_chunks() > 1
        assert frame.get_column("ok").n_chunks() > 1
    expressions = [
        cap.rsi_with_validity(pl.col("price"), pl.col("ok"), 2).alias("rsi_a"),
        cap.rsi_with_validity(pl.col("price"), pl.col("ok"), 3).alias("rsi_b"),
        cap.ema_with_validity(pl.col("price"), pl.col("ok"), 2).alias("ema_a"),
        cap.ema_with_validity(pl.col("price"), pl.col("ok"), 3).alias("ema_b"),
    ]
    result = frame.lazy().select(expressions).collect() if lazy else frame.select(expressions)
    for name, kind, period in [("rsi_a", "rsi", 2), ("rsi_b", "rsi", 3), ("ema_a", "ema", 2), ("ema_b", "ema", 3)]:
        assert_scores(result[name].to_list(), reference(values, period, kind))


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("period", [True, False, 0, -1, 1.5, "2", None, 2**63])
def test_python_period_rejected(kind, period):
    fn = getattr(cap, f"{kind}_with_validity")
    with pytest.raises(ValueError):
        fn(pl.col("value"), pl.col("source_valid"), period)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("period", [None, 0, -1])
def test_direct_rust_period_rejected(kind, period):
    from polars.plugins import register_plugin_function

    literal = pl.lit(period, dtype=pl.Int64)
    expr = register_plugin_function(
        plugin_path=cap._LIB,
        function_name=f"{kind}_with_validity",
        args=[pl.col("value").cast(pl.Float64), pl.col("source_valid").cast(pl.Boolean), literal],
        is_elementwise=False,
    )
    frame = pl.DataFrame({"value": [1.0, 2.0], "source_valid": [True, True]})
    with pytest.raises(pl.exceptions.PolarsError):
        frame.select(expr)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
@pytest.mark.parametrize("case", ["missing_arg", "extra_arg", "value_type", "validity_type"])
def test_direct_plugin_rejects_bad_boundary_inputs(kind, case):
    from polars.plugins import register_plugin_function

    frame = pl.DataFrame({"value": [1.0, 2.0], "source_valid": [True, True]})
    args = [pl.col("value"), pl.col("source_valid"), pl.lit(2, dtype=pl.Int64)]
    if case == "missing_arg":
        args.pop()
    elif case == "extra_arg":
        args.append(pl.lit(2, dtype=pl.Int64))
    elif case == "value_type":
        args[0] = pl.lit("bad")
    else:
        args[1] = pl.lit(1.0)
    expr = register_plugin_function(
        plugin_path=cap._LIB,
        function_name=f"{kind}_with_validity",
        args=args,
        is_elementwise=False,
    )
    with pytest.raises(pl.exceptions.PolarsError):
        frame.select(expr)


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_capability_flags_and_exports(kind):
    name = f"{kind}_with_validity"
    flag = f"HAS_{kind.upper()}_WITH_VALIDITY"
    assert name in cap.__all__
    assert flag in cap.__all__
    assert getattr(cap, flag) is True


@pytest.mark.parametrize("kind", ["ema", "rsi"])
def test_missing_symbol_raises_before_registration(kind, monkeypatch):
    flag = f"HAS_{kind.upper()}_WITH_VALIDITY"
    monkeypatch.setattr(cap, flag, False)
    with pytest.raises(RuntimeError, match="rebuild"):
        getattr(cap, f"{kind}_with_validity")(pl.lit(1.0), pl.lit(True), 2)
