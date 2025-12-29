# Predictive Maintenance System with Survival Analysis

A system predicting equipment failure probability over time from sensor data using survival analysis techniques to enable optimized maintenance scheduling.

## Overview

Traditional predictive maintenance approaches use binary classification (fail/not fail) which loses critical timing information. This system instead models **time-to-failure distributions**, enabling maintenance teams to:

- Schedule interventions at optimal times based on failure probability curves
- Estimate remaining useful life (RUL) with confidence bounds
- Identify root causes through survival-based feature importance
- Transfer learned patterns across similar equipment types

## Core Components

### Signal Processing Pipeline
Extracts statistical features from raw time-series sensor data including:
- Time-domain statistics (mean, variance, skewness, kurtosis)
- Frequency-domain features (spectral entropy, peak frequencies)
- Rolling window aggregations with multiple time horizons

### Survival Analysis Models
- **Survival Forests**: Non-parametric ensemble method for time-to-event prediction with censored observations
- **Cox Proportional Hazards**: Semi-parametric model providing interpretable hazard ratios for root cause analysis

### Remaining Useful Life Estimation
Produces RUL predictions with confidence intervals, enabling risk-aware maintenance scheduling.

### Transfer Learning
Shares learned representations across similar equipment types, reducing data requirements for new machines.

## Project Structure

```
predictive-maintenance-survival/
├── src/
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature extraction pipeline
│   ├── models/                # Survival models implementation
│   ├── evaluation/            # Custom metrics and cross-validation
│   └── inference/             # RUL estimation and prediction
├── tests/                     # Unit and integration tests
├── notebooks/                 # Exploration and visualization
├── data/                      # Sample datasets
├── docs/                      # Documentation and implementation plan
├── configs/                   # Configuration files
└── scripts/                   # CLI utilities
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/predictive-maintenance-survival.git
cd predictive-maintenance-survival

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.features import SensorFeatureExtractor
from src.models import SurvivalForest
from src.inference import RULEstimator

# Load and preprocess sensor data
extractor = SensorFeatureExtractor()
features = extractor.fit_transform(sensor_data)

# Train survival model
model = SurvivalForest(n_estimators=100)
model.fit(features, durations, events)

# Estimate remaining useful life
estimator = RULEstimator(model)
rul, confidence_bounds = estimator.predict(new_sensor_data)
```

## Key Features

| Feature | Description |
|---------|-------------|
| Censoring-aware training | Properly handles equipment that hasn't failed yet |
| Confidence intervals | Provides uncertainty quantification for RUL estimates |
| Feature importance | Cox model coefficients identify degradation drivers |
| Transfer learning | Reduces data needs for new equipment types |
| Modular pipeline | Easy to swap components and extend functionality |

## Dataset

This project uses the NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS), which provides:
- Run-to-failure sensor recordings from turbofan engines
- Multiple operational settings and fault modes
- Ideal for survival analysis due to natural censoring patterns

## Requirements

- Python 3.9+
- NumPy, Pandas, Scikit-learn
- scikit-survival (survival analysis)
- lifelines (Cox models, Kaplan-Meier)
- SciPy (signal processing)

## Documentation

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the detailed implementation roadmap.

## License

MIT License

## Contributing

Contributions are welcome. Please open an issue first to discuss proposed changes.
