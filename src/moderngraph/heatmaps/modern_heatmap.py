from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from moderngraph.theme import Theme


def make_cmap(colors=None):
    if colors:
        return LinearSegmentedColormap.from_list("heatmap", colors)
    return Theme.get_cmap("corp_heatmap")


def value_to_color(v, vmin, vmax, cmap):
    # Avoid division by zero
    if vmax == vmin:
        return cmap(0.5)[:3]
    return cmap((v - vmin) / (vmax - vmin))[:3]


def text_color_for_bg(rgb, threshold=0.55):
    brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if brightness < threshold else "#333333"


def draw_cells(
    ax,
    data,
    rows,
    cols,
    cmap,
    vmin,
    vmax,
    cell_size,
    gap,
    radius,
    fontsize,
    show_annotations,
):
    for ri, _ in enumerate(rows):
        for ci, _ in enumerate(cols):
            x = ci * (cell_size + gap)
            y = (len(rows) - 1 - ri) * (cell_size + gap)
            val = data[ri, ci]
            color = value_to_color(val, vmin, vmax, cmap)
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    cell_size,
                    cell_size,
                    boxstyle=f"round,pad=0,rounding_size={radius}",
                    linewidth=0,
                    facecolor=color,
                )
            )
            if show_annotations:
                ax.text(
                    x + cell_size / 2,
                    y + cell_size / 2,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=text_color_for_bg(color),
                )


def draw_row_labels(ax, rows, cell_size, gap, label_x, fontsize, label_color):
    for ri, label in enumerate(rows):
        y = (len(rows) - 1 - ri) * (cell_size + gap) + cell_size / 2
        ax.text(
            label_x,
            y,
            label,
            ha="right",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=label_color,
        )


def draw_col_labels(ax, cols, cell_size, gap, label_y, fontsize, label_color):
    for ci, label in enumerate(cols):
        x = ci * (cell_size + gap) + cell_size / 2
        ax.text(
            x,
            label_y,
            label,
            ha="center",
            va="top",
            fontsize=fontsize,
            fontweight="bold",
            color=label_color,
        )


def draw_title(
    ax,
    title,
    subtitle,
    total_h,
    title_x,
    title_fontsize,
    subtitle_fontsize,
    title_color,
    subtitle_color,
):
    ax.text(
        title_x,
        total_h + 0.5,
        title,
        ha="left",
        va="bottom",
        fontsize=title_fontsize,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        title_x,
        total_h + 0.2,
        subtitle,
        ha="left",
        va="bottom",
        fontsize=subtitle_fontsize,
        color=subtitle_color,
    )


def draw_legend(
    ax, cmap, vmin, vmax, total_h, legend_x, marker_size, fontsize, legend_color
):
    for gy in np.linspace(0, total_h, 300):
        ax.plot(
            legend_x,
            gy,
            "s",
            color=cmap(gy / total_h)[:3],
            markersize=marker_size,
            markeredgewidth=0,
        )
    ax.text(
        legend_x,
        -0.15,
        f"Low\n({int(vmin)})",
        ha="center",
        va="top",
        fontsize=fontsize,
        color=legend_color,
    )
    ax.text(
        legend_x,
        total_h + 0.15,
        f"High\n({int(vmax)})",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=legend_color,
    )


def extract_pandas_data(
    df: pd.DataFrame, y_col: str, x_col: str, val_col: str
) -> Tuple[List[str], List[str], np.ndarray]:
    rows = list(df[y_col].unique())
    cols = list(df[x_col].unique())
    df_pivot = df.pivot(index=y_col, columns=x_col, values=val_col)
    return rows, cols, df_pivot.loc[rows, cols].to_numpy()


def extract_polars_data(
    df: pl.DataFrame, y_col: str, x_col: str, val_col: str
) -> Tuple[List[str], List[str], np.ndarray]:
    rows = df.get_column(y_col).unique(maintain_order=True).to_list()
    cols = df.get_column(x_col).unique(maintain_order=True).to_list()
    df_pivot = df.pivot(index=y_col, on=x_col, values=val_col)

    # Ensure correct row order
    order_df = pl.DataFrame({y_col: rows, "_order": range(len(rows))})
    df_pivot = df_pivot.join(order_df, on=y_col).sort("_order")

    return rows, cols, df_pivot.select(cols).to_numpy()


def plot_modern_heatmap(
    df: Union[pd.DataFrame, pl.DataFrame],
    y_col: str,
    x_col: str,
    val_col: str,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    figsize: Tuple[float, float] = (14, 8),
    cell_size: float = 0.9,
    gap: float = 0.12,
    radius: float = 0.13,
    cell_fontsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 15,
    subtitle_fontsize: int = 13,
    legend_fontsize: int = 10,
    label_color: Optional[str] = None,
    title_color: Optional[str] = None,
    subtitle_color: Optional[str] = None,
    legend_color: Optional[str] = None,
    bg_color: Optional[str] = None,
    title: str = "Heatmap",
    subtitle: str = "",
    label_x_offset: float = -0.2,
    label_y_offset: float = -0.2,
    legend_x_offset: float = 0.5,
    legend_marker_size: float = 7,
    output_path: Optional[str] = "heatmap.png",
    dpi: int = 150,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = True,
    show_annotations: bool = True,
) -> None:
    """
    Plots a modern, aesthetically pleasing heatmap.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): DataFrame containing the data (Pandas or Polars).
        y_col (str): Column name for the y-axis (rows).
        x_col (str): Column name for the x-axis (columns).
        val_col (str): Column name for the cell values.
        colors (Optional[List[Tuple[float, float, float]]]): Custom colors for the colormap.
        figsize (Tuple[float, float]): Figure size.
        cell_size (float): Size of each square cell.
        gap (float): Gap between cells.
        radius (float): Corner radius of the cells.
        cell_fontsize (int): Font size for cell annotations.
        label_fontsize (int): Font size for axis labels.
        title_fontsize (int): Font size for the title.
        subtitle_fontsize (int): Font size for the subtitle.
        legend_fontsize (int): Font size for the legend text.
        label_color (Optional[str]): Color for axis labels.
        title_color (Optional[str]): Color for title.
        subtitle_color (Optional[str]): Color for subtitle.
        legend_color (Optional[str]): Color for legend text.
        bg_color (Optional[str]): Background color of the plot.
        title (str): Title text for the plot.
        subtitle (str): Subtitle text for the plot.
        label_x_offset (float): Offset for x-axis labels.
        label_y_offset (float): Offset for y-axis labels.
        legend_x_offset (float): X-offset for the legend.
        legend_marker_size (float): Size of the legend markers.
        output_path (Optional[str]): File path to save the plot (if any).
        dpi (int): Resolution for the saved plot.
        vmin (Optional[float]): Minimum data value for the colormap.
        vmax (Optional[float]): Maximum data value for the colormap.
        show (bool): Whether to display the plot.
        show_annotations (bool): Whether to show text annotations inside cells.
    """
    # Use theme fallbacks
    label_color = label_color or Theme.get_color("label")
    title_color = title_color or Theme.get_color("title")
    subtitle_color = subtitle_color or Theme.get_color("subtitle")
    legend_color = legend_color or Theme.get_color("legend")
    bg_color = bg_color or Theme.get_color("background")

    if isinstance(df, pl.DataFrame):
        rows, cols, data = extract_polars_data(df, y_col, x_col, val_col)
    elif isinstance(df, pd.DataFrame):
        rows, cols, data = extract_pandas_data(df, y_col, x_col, val_col)
    else:
        raise ValueError("Unsupported DataFrame type. Use pandas or polars.")

    cmap = make_cmap(colors)
    vmin = vmin if vmin is not None else np.nanmin(data)
    vmax = vmax if vmax is not None else np.nanmax(data)

    total_w = len(cols) * (cell_size + gap) - gap
    total_h = len(rows) * (cell_size + gap) - gap

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    draw_cells(
        ax,
        data,
        rows,
        cols,
        cmap,
        vmin,
        vmax,
        cell_size,
        gap,
        radius,
        cell_fontsize,
        show_annotations=show_annotations,
    )
    draw_row_labels(
        ax, rows, cell_size, gap, label_x_offset, label_fontsize, label_color
    )
    draw_col_labels(
        ax, cols, cell_size, gap, label_y_offset, label_fontsize, label_color
    )
    draw_title(
        ax,
        title,
        subtitle,
        total_h,
        0,
        title_fontsize,
        subtitle_fontsize,
        title_color,
        subtitle_color,
    )
    draw_legend(
        ax,
        cmap,
        vmin,
        vmax,
        total_h,
        total_w + legend_x_offset,
        legend_marker_size,
        legend_fontsize,
        legend_color,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
    if show:
        plt.show()
    plt.close(fig)
