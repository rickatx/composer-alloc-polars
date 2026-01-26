import ctypes
from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent


def _plugin_lib_path() -> Path | None:
    for suffix in ("*.so", "*.pyd", "*.dll", "*.dylib"):
        for path in _LIB.glob(f"_lib{suffix}"):
            return path
    return None


def _has_symbol(symbol: str) -> bool:
    path = _plugin_lib_path()
    if path is None:
        return False
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        return False
    return hasattr(lib, symbol)


HAS_ROLLING_MAX_DRAWDOWN = _has_symbol("_polars_plugin_field_rolling_max_drawdown")


def add_one(expr: pl.Expr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=_LIB,
        function_name="add_one",
        args=[expr],
        is_elementwise=True,
    )


def filter_select_weights(*score_exprs: pl.Expr, n: int, reverse: bool) -> pl.Expr:
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


__all__ = [
    "HAS_ROLLING_MAX_DRAWDOWN",
    "add_one",
    "filter_select_weights",
    "rolling_max_drawdown",
]
