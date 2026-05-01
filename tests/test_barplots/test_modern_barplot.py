import os

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from moderngraph.barplots.modern_barplot import (
    _compute_max_val,
    _contrast_color,
    _extract_data,
    _nice_step,
    plot_modern_barplot,
)


def test_nice_step():
    assert _nice_step(100) == 20
    assert _nice_step(10) == 2


def test_contrast_color():
    assert _contrast_color("#000000") == "#ffffff"
    assert _contrast_color("#ffffff") == "#000000"


def test_extract_data_pandas():
    df = pd.DataFrame({"y": ["A", "B"], "val1": [10, 20], "val2": [30, 40]})
    rows, categories = _extract_data(df, "y", ["val1", "val2"])
    assert rows == ["A", "B"]
    assert len(categories) == 2
    assert np.array_equal(categories[0], [10, 20])
    assert np.array_equal(categories[1], [30, 40])


def test_extract_data_polars():
    df = pl.DataFrame({"y": ["A", "B"], "val1": [10, 20], "val2": [30, 40]})
    rows, categories = _extract_data(df, "y", ["val1", "val2"])
    assert rows == ["A", "B"]
    assert len(categories) == 2
    assert np.array_equal(categories[0], [10, 20])
    assert np.array_equal(categories[1], [30, 40])


def test_extract_data_invalid_type():
    with pytest.raises(ValueError):
        _extract_data([1, 2, 3], "y", ["val"])  # type: ignore


def test_compute_max_val():
    categories = [np.array([10, 20]), np.array([30, 40])]
    assert _compute_max_val(categories, "percent") == 100.0


def test_plot_barplot_standard_pandas(tmp_path):
    df = pd.DataFrame({"y": ["A", "B"], "val": [10, 20]})
    out_file = tmp_path / "test_pandas_barplot.png"
    plot_modern_barplot(
        df,
        y_col="y",
        val_cols="val",
        barmode="standard",
        output_path=str(out_file),
        show=False,
    )
    assert os.path.exists(out_file)


def test_plot_barplot_stacked_polars(tmp_path):
    df = pl.DataFrame({"y": ["A", "B"], "val1": [10, 20], "val2": [30, 40]})
    out_file = tmp_path / "test_polars_stacked_barplot.png"
    plot_modern_barplot(
        df,
        y_col="y",
        val_cols=["val1", "val2"],
        barmode="stacked",
        output_path=str(out_file),
        show=False,
    )
    assert os.path.exists(out_file)


def test_plot_barplot_percent(tmp_path):
    df = pd.DataFrame({"y": ["A", "B"], "val1": [10, 20], "val2": [30, 40]})
    out_file = tmp_path / "test_percent_barplot.png"
    plot_modern_barplot(
        df,
        y_col="y",
        val_cols=["val1", "val2"],
        barmode="percent",
        output_path=str(out_file),
        show=False,
    )
    assert os.path.exists(out_file)


def test_plot_barplot_invalid_args():
    df = pd.DataFrame({"y": ["A", "B"], "val1": [10, 20], "val2": [30, 40]})

    with pytest.raises(ValueError, match="must refer to a single column"):
        plot_modern_barplot(
            df, y_col="y", val_cols=["val1", "val2"], barmode="standard"
        )

    with pytest.raises(ValueError, match="contain at least two category columns"):
        plot_modern_barplot(df, y_col="y", val_cols="val1", barmode="stacked")

    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        plot_modern_barplot(empty_df, y_col="y", val_cols="val1", barmode="standard")


def test_plot_barplot_optional_args(tmp_path):
    df = pd.DataFrame({"y": ["A", "B"], "val": [10, 20]})
    out_file = tmp_path / "test_optional_args_barplot.png"

    plot_modern_barplot(
        df,
        y_col="y",
        val_cols="val",
        figsize=(10, 6),
        title="Custom Title",
        subtitle="Custom Subtitle",
        output_path=str(out_file),
        show=False,
    )
    assert os.path.exists(out_file)

    # Test subtitle only
    plot_modern_barplot(
        df,
        y_col="y",
        val_cols="val",
        title="",
        subtitle="Only subtitle",
        show=False,
    )


def test_plot_barplot_short_bar(tmp_path):
    # Test short bar condition
    df = pl.DataFrame({"y": ["A", "B"], "val": [1, 100]})
    plot_modern_barplot(df, y_col="y", val_cols="val", barmode="standard", show=False)


def test_plot_barplot_show(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    df = pd.DataFrame({"y": ["A"], "val": [10]})
    plot_modern_barplot(df, "y", "val", barmode="standard", show=True, output_path=None)
