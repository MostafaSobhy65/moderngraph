# moderngraph

**moderngraph** is a modern, lightweight Python plotting library aiming to simplify the creation of beautiful and highly customizable data visualizations (heatmaps, bar plots, etc.) using seaborn, matplotlib, and pandas.

## Installation

```bash
pip install moderngraph
```

## Usage

```python
import pandas as pd
from moderngraph.heatmaps.modern_heatmap import ModernHeatmap

# Create some data
data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Plot a beautiful heatmap
heatmap = ModernHeatmap(data)
heatmap.plot()
```