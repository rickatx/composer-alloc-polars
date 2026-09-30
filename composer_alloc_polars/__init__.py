import ctypes
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent


def _plugin_lib_path() -> Path | None:
    """Find the platform-specific compiled extension for capability checks."""
    for suffix in ("*.so", "*.pyd", "*.dll", "*.dylib"):
        for path in _LIB.glob(f"_lib{suffix}"):
            return path
    return None


def _has_symbol(symbol: str) -> bool:
    """Report whether the compiled extension exports a Polars plugin symbol."""
    path = _plugin_lib_path()
    if path is None:
        return False
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        return False
    return hasattr(lib, symbol)


HAS_ROLLING_MAX_DRAWDOWN = _has_symbol("_polars_plugin_field_rolling_max_drawdown")
HAS_RSI_WITH_VALIDITY = _has_symbol("_polars_plugin_field_rsi_with_validity")
HAS_EMA_WITH_VALIDITY = _has_symbol("_polars_plugin_field_ema_with_validity")


def filter_select_weights(*score_exprs: pl.Expr, n: int, reverse: bool) -> pl.Expr:
    """Select up to ``n`` scores per row and return their equal weights."""
    if not score_exprs:
        raise ValueError("filter_select_weights requires at least one score expression")
    return register_plugin_function(
        plugin_path=_LIB,
        function_name="filter_select_weights",
        args=[
            *score_exprs,
            pl.lit(int(n), dtype=pl.Int64),
            pl.lit(bool(reverse)),
        ],
        is_elementwise=True,
    )


def rolling_max_drawdown(expr: pl.Expr, window: int) -> pl.Expr:
    """Return the existing rolling maximum drawdown expression."""
    if not HAS_ROLLING_MAX_DRAWDOWN:
        raise RuntimeError(
            "rolling_max_drawdown plugin not available; rebuild composer_alloc_polars"
        )
    return register_plugin_function(
        plugin_path=_LIB,
        function_name="rolling_max_drawdown",
        args=[expr, pl.lit(int(window), dtype=pl.Int64)],
        is_elementwise=False,
    )


def _indicator_with_validity(
    name: str, available: bool, values: pl.Expr, validity: pl.Expr, period: int
) -> pl.Expr:
    """Validate and register a recursive indicator with typed plugin inputs."""
    if isinstance(period, bool) or not isinstance(period, int) or not 0 < period <= 2**63 - 1:
        raise ValueError(f"{name} requires a positive integer period")
    if not available:
        raise RuntimeError(f"{name} plugin not available; rebuild composer_alloc_polars")
    return register_plugin_function(
        plugin_path=_LIB,
        function_name=name,
        args=[
            values.cast(pl.Float64),
            validity.cast(pl.Boolean),
            pl.lit(period, dtype=pl.Int64),
        ],
        is_elementwise=False,
    )


def rsi_with_validity(values: pl.Expr, validity: pl.Expr, period: int) -> pl.Expr:
    """Return Wilder RSI as ``Struct(value: Float64, valid: Boolean)``.

    ``validity`` marks source provenance. A null or non-finite value, or
    validity other than true, is invalid. A leading invalid prefix is ignored;
    invalidity after the first accepted price permanently poisons this frame.
    The first score follows ``period + 1`` valid prices. Invalid output has
    ``value=None`` and ``valid=False``.
    """
    return _indicator_with_validity(
        "rsi_with_validity", HAS_RSI_WITH_VALIDITY, values, validity, period
    )


def ema_with_validity(values: pl.Expr, validity: pl.Expr, period: int) -> pl.Expr:
    """Return SMA-seeded EMA as ``Struct(value: Float64, valid: Boolean)``.

    ``validity`` marks source provenance. A null or non-finite value, or
    validity other than true, is invalid. A leading invalid prefix is ignored;
    invalidity after the first accepted value permanently poisons this frame.
    The first EMA is the mean of ``period`` valid values. Invalid output has
    ``value=None`` and ``valid=False``.
    """
    return _indicator_with_validity(
        "ema_with_validity", HAS_EMA_WITH_VALIDITY, values, validity, period
    )


__all__ = [
    "HAS_ROLLING_MAX_DRAWDOWN",
    "HAS_RSI_WITH_VALIDITY",
    "HAS_EMA_WITH_VALIDITY",
    "filter_select_weights",
    "rolling_max_drawdown",
    "rsi_with_validity",
    "ema_with_validity",
]
