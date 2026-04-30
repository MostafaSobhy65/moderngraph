import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

_THEME_DIR = Path(__file__).parent
_COLORS_FILE = _THEME_DIR / "colors.json"


def load_theme_config() -> Dict[str, Any]:
    with open(_COLORS_FILE, "r") as f:
        return json.load(f)


THEME_CONFIG = load_theme_config()


class Theme:
    """Centralized theme configuration for all moderngraph plots."""

    @classmethod
    def get_palette(cls, name: str) -> List[Any]:
        return THEME_CONFIG["palettes"].get(name, [])

    @classmethod
    def get_color(cls, element: str) -> str:
        return THEME_CONFIG["ui"].get(element, "black")

    @classmethod
    def get_cmap(cls, palette_name: str) -> LinearSegmentedColormap:
        colors = cls.get_palette(palette_name)
        return LinearSegmentedColormap.from_list(f"{palette_name}_cmap", colors)

    @classmethod
    def set_matplotlib_defaults(cls):
        """Update global matplotlib rcParams with your standard theme."""
        categorical_colors = cls.get_palette("categorical")
        if categorical_colors:
            plt.rcParams["axes.prop_cycle"] = cycler(color=categorical_colors)

        plt.rcParams["text.color"] = cls.get_color("label")
        plt.rcParams["axes.labelcolor"] = cls.get_color("label")
        plt.rcParams["xtick.color"] = cls.get_color("label")
        plt.rcParams["ytick.color"] = cls.get_color("label")
        plt.rcParams["figure.facecolor"] = cls.get_color("background")
        plt.rcParams["axes.facecolor"] = cls.get_color("background")

    @classmethod
    def load_custom_theme(cls, path_to_json: str):
        """Allows users to inject their own theme config from their workspace"""
        with open(path_to_json, "r") as f:
            custom_config = json.load(f)
            THEME_CONFIG.update(custom_config)
            cls.set_matplotlib_defaults()


# Initialize global matplotlib defaults automatically (optional, or call manually)
Theme.set_matplotlib_defaults()
