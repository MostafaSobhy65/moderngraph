import json

from moderngraph.theme import Theme


def test_load_custom_theme(tmp_path):
    custom_theme = {
        "palettes": {"custom": ["#ff0000"]},
        "ui": {"background": "#ffffff"},
    }
    theme_file = tmp_path / "theme.json"
    theme_file.write_text(json.dumps(custom_theme))

    Theme.load_custom_theme(str(theme_file))
    assert Theme.get_color("background") == "#ffffff"
    assert Theme.get_palette("custom") == ["#ff0000"]
