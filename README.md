<div align="center">
  <img src="https://raw.githubusercontent.com/mostafasobhy65/moderngraph/main/static/banner.svg" alt="moderngraph banner" width="600" />


  [![Release](https://img.shields.io/pypi/v/moderngraph?label=Release)](https://github.com/mostafasobhy65/moderngraph/releases)
  [![License](https://img.shields.io/github/license/mostafasobhy65/moderngraph?label=License)](https://github.com/mostafasobhy65/moderngraph/blob/main/LICENSE)
  [![CI](https://img.shields.io/github/actions/workflow/status/mostafasobhy65/moderngraph/ci.yml?branch=main&label=CI)](https://github.com/mostafasobhy65/moderngraph/actions?query=workflow%3Aci)
  [![CD](https://img.shields.io/github/actions/workflow/status/mostafasobhy65/moderngraph/cd.yml?branch=main&label=CD)](https://github.com/mostafasobhy65/moderngraph/actions?query=workflow%3Acd)
  [![Coverage](https://codecov.io/gh/mostafasobhy65/moderngraph/graph/badge.svg)](https://codecov.io/gh/mostafasobhy65/moderngraph)


  **A modern, lightweight Python plotting library for modern, clean, highly customizable data visualizations.**
</div>

---

## Features

- **Minimalist API:** Create stunning visuals (heatmaps, bar plots, etc.) with just a few lines of code.
- **Highly Customizable:** Built on top of `seaborn` and `matplotlib`.
- **Data Ready:** Native support for `pandas` and `polars` DataFrames.
- **Smart Themes:** Clean, modern defaults to make your graphs presentation-ready out of the box.

## Installation

```bash
pip install moderngraph
```

## Quickstart

Render a beautiful heatmap effortlessly:

```python
import pandas as pd
from moderngraph.heatmaps.modern_heatmap import ModernHeatmap

# 1. Create some data
data = pd.DataFrame({
    "A": [1, 2],
    "B": [3, 4]
})

# 2. Plot a beautiful heatmap
heatmap = ModernHeatmap(data)
heatmap.plot()
```

> **Tip:** Check out the interactive [EDA Notebooks](notebooks/) for more in-depth examples!

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.
