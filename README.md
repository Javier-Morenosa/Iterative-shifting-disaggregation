# ISD-Algorithm: Iterative Shifting Disaggregation
## Description
Python implementation of the Iterative Shifting Disaggregation (ISD) Algorithm, designed to disaggregate multiple low-frequency time series with irregular sampling and possibly overlapping intervals, into a single high-frequency (daily) time series.

## Features
* **Multi-source disaggregation**: Processes multiple low-frequency time series with irregular intervals.
* **Temporal consistency**: Maintains consistency between aggregation levels, ensuring that the sum of disaggregated values matches the original observations.
* **Flexible modeling**: Uses correlated exogenous variables to improve disaggregation accuracy.
* **Scientific API**: Programming interface compatible with the Python scientific ecosystem (NumPy, Pandas, SciPy).
* **Complete documentation**: Well-documented implementation based on the original academic paper.

## Installation
```
pip install isd-algorithm
```

## Quick Usage
```python
import pandas as pd
import numpy as np
from isd.core.models import LowFrequencySeries, ISDAlgorithm

# 1. Prepare low-frequency series
series_a = LowFrequencySeries(
    name="Series_A",
    observations=[
        (150, pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-31')),
        (180, pd.Timestamp('2023-02-01'), pd.Timestamp('2023-02-28')),
    ]
)

# 2. Prepare daily exogenous variables
dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='D')
temperature = pd.Series(
    np.sin(np.linspace(0, 2*np.pi, len(dates))) * 15 + 40,  # Synthetic temperature
    index=dates
)
exog_vars = pd.DataFrame({
    'temp': temperature,
}, index=dates)

# 3. Create and run the ISD algorithm
isd = ISDAlgorithm(
    lf_series=[series_a],
    exogenous_vars=exog_vars,
    n_lr_models=5,
    n_disagg_cycles=5,
    alpha=0.1
)

# 4. Disaggregate and get results
daily_series = isd.disaggregate(verbose=True)

# 5. Visualize results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(daily_series.index, daily_series.values, '-o', label='Disaggregated (ISD)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Disaggregated time series')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Advanced Example
For a more complete example showing how to disaggregate multiple time series with overlapping intervals, check the example notebook in the documentation.

## Recommended Parameters
The ISD algorithm has three main parameters that control its behavior:
- n_lr_models: Number of linear regression models trained
- n_disagg_cycles: Number of disaggregation cycles per model
- alpha: Weight for error redistribution

Recommendations based on the number of input series:
- 3 or more series: n_lr_models=10, n_disagg_cycles=10, alpha=0.05
- Fewer than 3 series: More iterations are required; try n_lr_models=15, n_disagg_cycles=15, alpha=0.1

## Requirements
- Python 3.7+
- NumPy
- Pandas
- SciPy
- Matplotlib (optional, for visualization)

## How to Cite
If you use this library in your research, please cite the original paper:
```
@article{quinn2024iterative,
    title={An Iterative Shifting Disaggregation Algorithm for Multi-Source, Irregularly Sampled, and Overlapped Time Series},
    author={Quinn, Colin O and Brown, Ronald H and Corliss, George F and Povinelli, Richard J},
    journal={Sensors},
    pages={895},
    year={2025},
    publisher={MDPI},
    doi={10.3390/s25030895}
}
```
