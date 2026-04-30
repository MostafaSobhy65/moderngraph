import os

import matplotlib
import pandas as pd
import polars as pl

matplotlib.use("Agg")  # Use non-interactive backend for testing

from moderngraph.heatmaps import (
    extract_pandas_data,
    extract_polars_data,
    plot_modern_heatmap,
)


def test_extract_pandas_data():
    df = pd.DataFrame({"y": ["A", "B", "A", "B"], "x": ["C", "C", "D", "D"], "val": [1, 2, 3, 4]})
    rows, cols, data = extract_pandas_data(df, "y", "x", "val")
    assert rows == ["A", "B"]
    assert cols == ["C", "D"]
    assert data.shape == (2, 2)
    assert data[0, 0] == 1
    assert data[1, 1] == 4


def test_extract_polars_data():
    df = pl.DataFrame({"y": ["A", "B", "A", "B"], "x": ["C", "C", "D", "D"], "val": [1, 2, 3, 4]})
    rows, cols, data = extract_polars_data(df, "y", "x", "val")
    assert rows == ["A", "B"]
    assert cols == ["C", "D"]
    assert data.shape == (2, 2)
    assert data[0, 0] == 1
    assert data[1, 1] == 4


def test_plot_heatmap_pandas(tmp_path):
    df = pd.DataFrame({"y": ["A", "A", "B", "B"], "x": ["C", "D", "C", "D"], "val": [10, 20, 30, 40]})
    out_file = tmp_path / "test_pandas_heatmap.png"

    plot_modern_heatmap(df, y_col="y", x_col="x", val_col="val", output_path=str(out_file), show=False)

    assert os.path.exists(out_file)


def test_plot_heatmap_polars(tmp_path):
    df = pl.DataFrame({"y": ["A", "A", "B", "B"], "x": ["C", "D", "C", "D"], "val": [10, 20, 30, 40]})
    out_file = tmp_path / "test_polars_heatmap.png"

    plot_modern_heatmap(df, y_col="y", x_col="x", val_col="val", output_path=str(out_file), show=False)

    assert os.path.exists(out_file)
