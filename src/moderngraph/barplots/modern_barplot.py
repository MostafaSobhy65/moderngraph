from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.patches import Patch

from moderngraph.theme import Theme


def _nice_step(max_val: float, target_ticks: int = 5) -> float:
    raw = max_val / target_ticks
    magnitude = 10 ** np.floor(np.log10(raw if raw > 0 else 1))
    normalized = raw / magnitude
    nice = (
        1 if normalized < 1.5 else 2 if normalized < 3 else 5 if normalized < 7 else 10
    )
    return nice * magnitude


def _contrast_color(hex_color: str) -> str:
    """Return black or white for maximum contrast against the given hex color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 0.45 else "#ffffff"


def _draw_legend(
    ax,
    labels: List[str],
    colors: List[str],
    fontsize: int,
    text_color: str,
    len_rows: int,
) -> None:
    handles = [
        Patch(facecolor=colors[i % len(colors)], label=label)
        for i, label in enumerate(labels)
    ]
    dynamic_va_padding = 1.0 / max(1, len_rows * 1.35)
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1, 1 + dynamic_va_padding),
        ncols=len(labels),
        frameon=False,
        fontsize=fontsize,
        labelcolor=text_color,
    )


def _extract_data(
    df: Union[pd.DataFrame, pl.DataFrame],
    y_col: str,
    val_cols: List[str],
) -> Tuple[List[str], List[np.ndarray]]:
    if isinstance(df, pl.DataFrame):
        return df.get_column(y_col).to_list(), [
            df.get_column(c).to_numpy() for c in val_cols
        ]
    if isinstance(df, pd.DataFrame):
        return df[y_col].tolist(), [df[c].to_numpy() for c in val_cols]
    raise ValueError("Unsupported DataFrame type. Use pandas or polars.")


def _compute_max_val(categories: List[np.ndarray], barmode: str) -> float:
    if barmode == "percent":
        return 100.0
    raw = (
        np.sum(categories, axis=0).max()
        if barmode == "stacked"
        else max(c.max() for c in categories)
    )
    magnitude = 10 ** np.floor(np.log10(raw if raw > 0 else 1))
    return float(np.ceil(raw / magnitude) * magnitude * 1.1)


def draw_axes_and_spines(
    ax,
    y_pos: np.ndarray,
    labels: List[str],
    text_color: str,
    label_fontsize: int,
    max_val: float,
    is_percent: bool = False,
) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        labels, color=text_color, fontweight="bold", fontsize=label_fontsize
    )
    ax.tick_params(axis="both", which="both", length=0)

    step = _nice_step(max_val)
    ticks = np.arange(0, max_val + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{int(t)}%" if is_percent else f"{int(t)}" for t in ticks],
        color=text_color,
        fontsize=label_fontsize,
    )
    ax.grid(axis="x", color="#eeeeee", linestyle="-", linewidth=1, zorder=0)
    ax.set_xlim(-(max_val * 0.05), max_val * 1.05)


def draw_title(
    ax,
    title: str,
    subtitle: str,
    title_fontsize: int,
    subtitle_fontsize: int,
    title_color: str,
    subtitle_color: str,
) -> None:
    if title and subtitle:
        # Title reserves space; subtitle slips underneath it seamlessly, avoiding huge manual pads
        ax.set_title(
            f"{title}\n",
            loc="left",
            color=title_color,
            fontsize=title_fontsize,
            fontweight="bold",
        )
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=subtitle_fontsize,
            color=subtitle_color,
        )
    elif title:
        ax.set_title(
            title,
            loc="left",
            color=title_color,
            fontsize=title_fontsize,
            fontweight="bold",
        )
    elif subtitle:
        ax.set_title(
            subtitle, loc="left", color=subtitle_color, fontsize=subtitle_fontsize
        )


def plot_modern_barplot(
    df: Union[pd.DataFrame, pl.DataFrame],
    y_col: str,
    val_cols: Union[str, List[str]],
    barmode: Literal["standard", "stacked", "percent"] = "standard",
    colors: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    figwidth: float = 8,
    line_thickness: int = 25,
    min_label_width: float = 0.01,
    bar_spacing: int = 6,
    label_fontsize: int = 10,
    title_fontsize: int = 18,
    subtitle_fontsize: int = 13,
    label_color: Optional[str] = None,
    title_color: Optional[str] = None,
    subtitle_color: Optional[str] = None,
    bg_color: Optional[str] = None,
    bar_bg_color: str = "#f2f2f2",
    title: str = "Bar Graph",
    subtitle: str = "",
    output_path: Optional[str] = None,
    dpi: int = 150,
    show_legend: bool = True,
    legend_labels: Optional[List[str]] = None,
    show: bool = True,
) -> None:
    label_color = label_color or Theme.get_color("label")
    title_color = title_color or Theme.get_color("title")
    subtitle_color = subtitle_color or Theme.get_color("subtitle")
    bg_color = bg_color or Theme.get_color("background")
    colors = colors or Theme.get_palette("corp_categorical")

    if isinstance(val_cols, str):
        val_cols = [val_cols]

    if barmode == "standard" and len(val_cols) > 1:
        raise ValueError(
            "For 'standard' barmode, 'val_cols' must refer to a single column."
        )
    if barmode in ["stacked", "percent"] and len(val_cols) < 2:
        raise ValueError(
            f"For '{barmode}' barmode, 'val_cols' must contain at least two category columns."
        )
    if df.shape[0] == 0:
        raise ValueError("DataFrame is empty.")

    rows, categories = _extract_data(df, y_col, val_cols)

    header_inches = 1.2
    inches_per_bar = (line_thickness + bar_spacing) / 72.0
    if figsize is not None:
        figwidth, figheight = figsize
    else:
        figwidth = figwidth
        figheight = max(1.0, len(rows) * inches_per_bar + header_inches)
    figsize = (figwidth, figheight)

    is_percent = barmode == "percent"
    data_to_plot = (
        [cat / np.sum(categories, axis=0) * 100 for cat in categories]
        if is_percent
        else categories
    )
    max_val = _compute_max_val(categories, barmode)
    label_padding = max_val * 0

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    y_pos = np.arange(len(rows))

    for i, y in enumerate(y_pos):
        # Background track
        ax.plot(
            [0, max_val],
            [y, y],
            color=bar_bg_color,
            linewidth=line_thickness,
            solid_capstyle="round",
            zorder=1,
        )

        if barmode in ["stacked", "percent"]:
            running_total = sum(cat[i] for cat in data_to_plot)
            # Draw segments backwards
            for j in range(len(data_to_plot) - 1, -1, -1):
                if running_total > 0:
                    ax.plot(
                        [0, running_total],
                        [y, y],
                        color=colors[j % len(colors)],
                        linewidth=line_thickness,
                        solid_capstyle="round",
                        zorder=2 + (len(data_to_plot) - j),
                    )
                running_total -= data_to_plot[j][i]

            current_start = 0.0
            for j, cat_vals in enumerate(data_to_plot):
                val = cat_vals[i]
                if val >= max_val * min_label_width:
                    text = f"{int(round(val))}%" if is_percent else str(int(val))

                    # Safely pad from the right edge. Fallback to center if the segment is extremely small.
                    segment_padding = min(label_padding, val / 2)

                    ax.text(
                        current_start
                        + val
                        - segment_padding,  # Placed on the right inside the segment
                        float(y),
                        text,
                        color=_contrast_color(colors[j % len(colors)]),
                        fontweight="bold",
                        fontsize=label_fontsize,
                        ha="right",
                        va="center_baseline",
                        zorder=10,
                    )
                current_start += val
        else:
            val = data_to_plot[0][i]
            if val > 0:
                ax.plot(
                    [0, val],
                    [y, y],
                    color=colors[0],
                    linewidth=line_thickness,
                    solid_capstyle="round",
                    zorder=2,
                )
                text = f"{int(round(val))}%" if is_percent else str(int(val))

                # Check if bar is wide enough to hold the text inside
                if val >= max_val * 0.08:
                    segment_padding = min(label_padding, val / 2)
                    ax.text(
                        val - segment_padding,
                        float(y),
                        text,
                        color=_contrast_color(colors[0]),
                        fontweight="bold",
                        fontsize=label_fontsize,
                        ha="right",  # Placed on the right inside the bar
                        va="center_baseline",
                        zorder=3,
                    )
                else:
                    # Text spills outward gracefully if the bar is too short
                    ax.text(
                        val + label_padding,
                        float(y),
                        text,
                        color=label_color,
                        fontweight="bold",
                        fontsize=label_fontsize,
                        ha="left",  # Anchored on the right side, outside the bar
                        va="center_baseline",
                        zorder=3,
                    )

    draw_axes_and_spines(
        ax, y_pos, rows, label_color, label_fontsize, max_val, is_percent
    )
    draw_title(
        ax,
        title,
        subtitle,
        title_fontsize,
        subtitle_fontsize,
        title_color,
        subtitle_color,
    )
    if show_legend and barmode in ["stacked", "percent"]:
        labels = legend_labels or val_cols
        # Scale the vertical padding inversely with the number of rows to keep the absolute physical padding constant
        _draw_legend(ax, labels, colors, label_fontsize, label_color, len(rows))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
    if show:
        plt.show()
    plt.close(fig)
