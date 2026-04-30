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

## Development

This project uses `uv` for dependency management and building.

1. Ensure you have `uv` installed (`pip install uv`).
2. Clone the repository and install dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```
3. Run tests before committing:
   ```bash
   pytest
   ```

## Releasing a new version

To bump the version and publish:

1. Update the `version` field in `pyproject.toml`.
2. Commit and push to the `main` branch.
3. The GitHub Actions release workflow will automatically create a tag and publish the package to PyPI.
