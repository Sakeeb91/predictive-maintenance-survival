# Implementation Plan: Predictive Maintenance System with Survival Analysis

## Expert Role

**ML Engineer / Data Scientist** specializing in survival analysis, time-series signal processing, and industrial IoT applications.

This role is optimal because the project requires:
- Signal processing expertise for sensor data feature extraction
- Deep understanding of survival analysis mathematics (hazard functions, censoring)
- Time-series modeling experience with irregular and incomplete data
- Statistical rigor for confidence interval estimation
- Transfer learning techniques for multi-equipment scenarios

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREDICTIVE MAINTENANCE SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  Raw Sensor  │───▶│  Signal Process  │───▶│   Feature Store          │  │
│  │    Data      │    │    Pipeline      │    │   (extracted features)   │  │
│  │  (CSV/HDF5)  │    │                  │    │                          │  │
│  └──────────────┘    └──────────────────┘    └────────────┬─────────────┘  │
│                                                            │                │
│                      ┌─────────────────────────────────────┘                │
│                      │                                                      │
│                      ▼                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    SURVIVAL ANALYSIS ENGINE                           │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │ │
│  │  │ Survival Forest │  │ Cox Proportional│  │  Custom CV with     │   │ │
│  │  │ (RSF/CSF)       │  │ Hazards Model   │  │  Censoring-Aware    │   │ │
│  │  │                 │  │                 │  │  Splitting          │   │ │
│  │  └────────┬────────┘  └────────┬────────┘  └─────────────────────┘   │ │
│  │           │                    │                                      │ │
│  │           ▼                    ▼                                      │ │
│  │  ┌─────────────────────────────────────────┐                         │ │
│  │  │         Survival Function S(t)          │                         │ │
│  │  │    (probability of survival past t)     │                         │ │
│  │  └────────────────────┬────────────────────┘                         │ │
│  └───────────────────────┼──────────────────────────────────────────────┘ │
│                          │                                                 │
│                          ▼                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    RUL ESTIMATION MODULE                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │ │
│  │  │ Median RUL from │  │ Confidence      │  │ Maintenance         │   │ │
│  │  │ Survival Curve  │  │ Intervals       │  │ Scheduling API      │   │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    TRANSFER LEARNING MODULE                           │ │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────┐  │ │
│  │  │ Equipment Embeddings    │───▶│ Fine-tuning for New Machines    │  │ │
│  │  │ (shared representations)│    │ (few-shot adaptation)           │  │ │
│  │  └─────────────────────────┘    └─────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw Sensor Readings (21 sensors, 3 operational settings)
         │
         ▼
┌─────────────────────────────────────────┐
│ 1. PREPROCESSING                        │
│    - Normalization per operating mode   │
│    - Missing value imputation           │
│    - Outlier detection                  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ 2. FEATURE EXTRACTION                   │
│    - Rolling statistics (5, 10, 20, 50) │
│    - Spectral features via FFT          │
│    - Degradation trend slopes           │
│    - Health indicators                  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ 3. SURVIVAL DATA FORMATTING             │
│    - Duration: cycles until failure     │
│    - Event: 1 if failed, 0 if censored  │
│    - Covariates: extracted features     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ 4. MODEL TRAINING                       │
│    - Split with censoring stratification│
│    - Fit survival forest + Cox model    │
│    - Evaluate with C-index, Brier score │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│ 5. RUL PREDICTION                       │
│    - Compute survival curve S(t)        │
│    - Extract median survival time       │
│    - Calculate confidence intervals     │
└─────────────────────────────────────────┘
```

---

## Technology Selection

| Component | Technology | Rationale | Tradeoffs | Fallback |
|-----------|------------|-----------|-----------|----------|
| **Core Language** | Python 3.9+ | Industry standard for ML, extensive libraries | Slower than compiled languages | None needed |
| **Data Handling** | Pandas + NumPy | Familiar, well-documented, sufficient for tabular data | Memory-bound for huge datasets | Polars if performance issues |
| **Survival Models** | scikit-survival | Most complete Python survival analysis library | Fewer models than R survival packages | lifelines (simpler API) |
| **Cox Models** | lifelines | Excellent documentation, visualization built-in | Less optimized than sksurv for large data | statsmodels CoxPHFitter |
| **Signal Processing** | SciPy | Standard library, FFT, filtering | No GPU acceleration | PyWavelets for wavelets |
| **ML Framework** | Scikit-learn | Consistent API, easy to extend | Not designed for survival analysis | None needed |
| **Visualization** | Matplotlib + Seaborn | Publication-quality, extensive control | Verbose syntax | Plotly for interactivity |
| **Testing** | pytest | Industry standard, fixtures, parametrize | None significant | unittest (stdlib) |
| **Config Management** | YAML + dataclasses | Simple, readable, type-safe | No schema validation | Pydantic for validation |

### Key Libraries Deep Dive

**scikit-survival** (sksurv):
- RandomSurvivalForest for non-parametric ensemble
- CoxPHSurvivalAnalysis for interpretable models
- Proper handling of censored observations
- Compatible with scikit-learn pipeline API

**lifelines**:
- KaplanMeierFitter for non-parametric survival curves
- CoxPHFitter with built-in diagnostics
- Excellent plotting utilities
- Confidence interval computation

---

## Phased Implementation Plan

### Phase 1: Data Pipeline and Feature Extraction

**Objective**: Load C-MAPSS dataset and extract meaningful features from sensor time series.

**Scope**:
- `src/data/loader.py` - Data loading and basic preprocessing
- `src/data/preprocessor.py` - Normalization, train/test split
- `src/features/time_domain.py` - Statistical feature extractors
- `src/features/extractor.py` - Main feature extraction pipeline

**Deliverables**:
1. Loaded dataset with train/test engines separated
2. Feature matrix with 50+ engineered features per sample
3. Survival-formatted data (duration, event, covariates)

**Verification**:
```bash
python -m pytest tests/test_data_loader.py -v
python scripts/verify_features.py  # Prints feature statistics
```

**Technical Challenges**:
- C-MAPSS has multiple subdatasets (FD001-FD004) with different complexities
- Operating condition normalization required before feature extraction
- Determining optimal rolling window sizes

**Definition of Done**:
- [ ] DataLoader returns pandas DataFrame with all 21 sensors
- [ ] Features extracted produce non-null values for all samples
- [ ] Train/test split preserves engine IDs (no data leakage)
- [ ] Feature extraction runs in < 30 seconds for full dataset

**Code Skeleton**:

```python
# src/data/loader.py
from pathlib import Path
import pandas as pd
import numpy as np

class CMAPSSDataLoader:
    """Loads NASA C-MAPSS turbofan engine degradation dataset."""

    SENSOR_COLUMNS = [f's_{i}' for i in range(1, 22)]
    SETTING_COLUMNS = ['setting_1', 'setting_2', 'setting_3']

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_dataset(self, subset: str = 'FD001') -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data for specified subset.

        Args:
            subset: One of 'FD001', 'FD002', 'FD003', 'FD004'

        Returns:
            Tuple of (train_df, test_df)
        """
        raise NotImplementedError

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RUL (remaining useful life) column to dataframe."""
        raise NotImplementedError
```

```python
# src/features/time_domain.py
import numpy as np
from scipy import stats

def compute_rolling_statistics(
    series: np.ndarray,
    window_sizes: list[int] = [5, 10, 20, 50]
) -> dict[str, float]:
    """Compute rolling statistics for a sensor time series.

    Args:
        series: 1D array of sensor readings
        window_sizes: List of window sizes for rolling calculations

    Returns:
        Dictionary mapping feature names to values
    """
    raise NotImplementedError

def compute_trend_features(series: np.ndarray) -> dict[str, float]:
    """Compute trend-based features indicating degradation.

    Includes linear regression slope, acceleration, and residuals.
    """
    raise NotImplementedError
```

---

### Phase 2: Survival Model Implementation

**Objective**: Implement and train survival forest and Cox proportional hazards models.

**Scope**:
- `src/models/survival_forest.py` - Random Survival Forest wrapper
- `src/models/cox_model.py` - Cox PH model with feature importance
- `src/evaluation/metrics.py` - C-index, integrated Brier score
- `src/evaluation/cross_validation.py` - Censoring-aware CV

**Deliverables**:
1. Trained survival forest model
2. Trained Cox model with hazard ratios
3. Cross-validation results with proper censoring handling
4. Model persistence (joblib/pickle)

**Verification**:
```bash
python -m pytest tests/test_survival_models.py -v
python scripts/train_and_evaluate.py --model rsf --cv 5
```

**Technical Challenges**:
- Right-censoring handling in cross-validation splits
- Hyperparameter tuning with survival-appropriate metrics
- Avoiding information leakage with time-series data

**Definition of Done**:
- [ ] Survival Forest achieves C-index > 0.70 on test set
- [ ] Cox model produces interpretable hazard ratios
- [ ] Cross-validation uses stratified censoring splits
- [ ] Models save and load correctly

**Code Skeleton**:

```python
# src/models/survival_forest.py
from sksurv.ensemble import RandomSurvivalForest
import numpy as np

class SurvivalForestModel:
    """Wrapper for Random Survival Forest with consistent API."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 10,
        random_state: int = 42
    ):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, duration: np.ndarray, event: np.ndarray):
        """Fit the survival forest.

        Args:
            X: Feature matrix (n_samples, n_features)
            duration: Time to event or censoring
            event: Boolean indicator (True if event occurred)
        """
        raise NotImplementedError

    def predict_survival_function(self, X: np.ndarray) -> np.ndarray:
        """Predict survival function S(t) for each sample."""
        raise NotImplementedError

    def predict_median_survival_time(self, X: np.ndarray) -> np.ndarray:
        """Predict median survival time (when S(t) = 0.5)."""
        raise NotImplementedError
```

```python
# src/evaluation/cross_validation.py
from sklearn.model_selection import StratifiedKFold
import numpy as np

class CensoringAwareCV:
    """Cross-validation that stratifies by censoring status and time quantiles."""

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        duration: np.ndarray,
        event: np.ndarray
    ):
        """Generate train/test indices with censoring-aware stratification.

        Yields:
            Tuple of (train_indices, test_indices)
        """
        raise NotImplementedError
```

---

### Phase 3: RUL Estimation with Confidence Intervals

**Objective**: Build inference module that produces RUL predictions with uncertainty quantification.

**Scope**:
- `src/inference/rul_estimator.py` - RUL prediction from survival curves
- `src/inference/confidence_intervals.py` - Bootstrap and analytical CIs
- `src/inference/scheduling.py` - Maintenance scheduling recommendations

**Deliverables**:
1. RUL predictions with point estimates and intervals
2. Scheduling recommendations based on risk thresholds
3. Visualization of survival curves with uncertainty bands

**Verification**:
```bash
python -m pytest tests/test_rul_estimation.py -v
python scripts/predict_rul.py --engine-id 1 --visualize
```

**Technical Challenges**:
- Survival curve may not cross 0.5 (median undefined)
- Bootstrap confidence intervals computationally expensive
- Calibration of prediction intervals

**Definition of Done**:
- [ ] RUL predictions within 15% of actual for 80% of engines
- [ ] Confidence intervals have correct coverage (95% CI covers 95% of true values)
- [ ] Scheduling API recommends intervention timing

**Code Skeleton**:

```python
# src/inference/rul_estimator.py
import numpy as np
from typing import Tuple

class RULEstimator:
    """Estimates Remaining Useful Life with confidence bounds."""

    def __init__(self, model, confidence_level: float = 0.95):
        self.model = model
        self.confidence_level = confidence_level

    def predict(
        self,
        X: np.ndarray,
        current_age: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict RUL with confidence interval.

        Args:
            X: Feature matrix for current state
            current_age: Current operational time of each equipment

        Returns:
            Tuple of (rul_point_estimate, lower_bound, upper_bound)
        """
        raise NotImplementedError

    def recommend_maintenance(
        self,
        X: np.ndarray,
        current_age: np.ndarray,
        risk_threshold: float = 0.10
    ) -> dict:
        """Recommend maintenance based on failure probability threshold.

        Returns:
            Dict with 'recommended_time', 'failure_probability', 'urgency'
        """
        raise NotImplementedError
```

---

### Phase 4: Transfer Learning for Multi-Equipment

**Objective**: Enable knowledge transfer across similar equipment types.

**Scope**:
- `src/models/transfer_learning.py` - Multi-equipment embedding model
- `src/models/domain_adaptation.py` - Fine-tuning for new equipment
- `configs/equipment_config.yaml` - Equipment type definitions

**Deliverables**:
1. Shared feature representation across FD001-FD004
2. Fine-tuning API for new equipment with limited data
3. Performance comparison: transfer vs from-scratch

**Verification**:
```bash
python -m pytest tests/test_transfer_learning.py -v
python scripts/transfer_experiment.py --source FD001 --target FD002
```

**Technical Challenges**:
- Distribution shift between operating conditions
- Determining optimal amount of target data for fine-tuning
- Negative transfer detection

**Definition of Done**:
- [ ] Transfer model outperforms from-scratch with < 50% target data
- [ ] Fine-tuning API works with as few as 10 target engines
- [ ] Negative transfer detection warns when source is unhelpful

**Code Skeleton**:

```python
# src/models/transfer_learning.py
import numpy as np
from typing import Optional

class TransferSurvivalModel:
    """Survival model with transfer learning capabilities."""

    def __init__(self, base_model, embedding_dim: int = 32):
        self.base_model = base_model
        self.embedding_dim = embedding_dim
        self.source_encoder = None
        self.is_pretrained = False

    def pretrain(
        self,
        X_source: np.ndarray,
        duration_source: np.ndarray,
        event_source: np.ndarray,
        equipment_ids: Optional[np.ndarray] = None
    ):
        """Pretrain on source domain data."""
        raise NotImplementedError

    def finetune(
        self,
        X_target: np.ndarray,
        duration_target: np.ndarray,
        event_target: np.ndarray,
        freeze_encoder: bool = True
    ):
        """Fine-tune on target domain with optional encoder freezing."""
        raise NotImplementedError

    def get_equipment_embedding(self, X: np.ndarray) -> np.ndarray:
        """Extract learned equipment representation."""
        raise NotImplementedError
```

---

### Phase 5: CLI, Visualization, and Documentation

**Objective**: Create user-friendly interfaces and comprehensive documentation.

**Scope**:
- `scripts/train.py` - Training CLI
- `scripts/predict.py` - Prediction CLI
- `notebooks/01_exploration.ipynb` - Data exploration
- `notebooks/02_model_comparison.ipynb` - Model benchmarking

**Deliverables**:
1. Complete CLI for training and prediction
2. Interactive notebooks for exploration
3. API documentation (docstrings + examples)

**Verification**:
```bash
python scripts/train.py --help
python scripts/predict.py --model-path models/rsf.pkl --data test.csv
```

**Definition of Done**:
- [ ] CLI handles all major workflows
- [ ] Notebooks run end-to-end without errors
- [ ] All public functions have docstrings with examples

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning | Mitigation |
|------|------------|--------|---------------|------------|
| Survival library version conflicts | Medium | 🟡 Medium | Import errors during setup | Pin exact versions in requirements.txt |
| Poor C-index on test data | Medium | 🔴 High | Validation C-index < 0.65 | Feature engineering iteration, try different models |
| Memory issues with large datasets | Low | 🟡 Medium | OOM errors on full data | Batch processing, downsample for dev |
| Confidence intervals too wide | Medium | 🟡 Medium | CI width > 50% of RUL | Increase data, reduce variance with ensembles |
| Transfer learning negative transfer | Medium | 🟡 Medium | Target performance drops | Domain similarity checks before transfer |
| Cox PH assumptions violated | Medium | 🟡 Medium | Schoenfeld residual tests fail | Switch to time-varying covariates or non-parametric |

---

## Testing Strategy

### Unit Tests
Test individual functions in isolation with known inputs/outputs.

```python
# tests/test_feature_extraction.py
import pytest
import numpy as np
from src.features.time_domain import compute_rolling_statistics

def test_rolling_mean_window_5():
    """Rolling mean with window 5 produces correct values."""
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = compute_rolling_statistics(series, window_sizes=[5])
    assert 'rolling_mean_5' in result
    assert result['rolling_mean_5'] == pytest.approx(8.0, rel=1e-5)  # Mean of last 5

def test_trend_slope_increasing():
    """Trend features detect increasing degradation pattern."""
    series = np.linspace(0, 100, 50)  # Linear increase
    result = compute_trend_features(series)
    assert result['trend_slope'] > 0

def test_trend_slope_stable():
    """Trend features detect stable (non-degrading) pattern."""
    series = np.ones(50) + np.random.randn(50) * 0.01
    result = compute_trend_features(series)
    assert abs(result['trend_slope']) < 0.1
```

### Integration Tests
Test component interactions.

```python
# tests/test_pipeline_integration.py
def test_full_pipeline_runs():
    """Feature extraction -> model training -> prediction completes."""
    loader = CMAPSSDataLoader(DATA_DIR)
    train_df, _ = loader.load_dataset('FD001')

    extractor = SensorFeatureExtractor()
    features = extractor.fit_transform(train_df)

    model = SurvivalForestModel(n_estimators=10)  # Small for speed
    model.fit(features.values, durations, events)

    survival_curves = model.predict_survival_function(features.values[:5])
    assert survival_curves.shape[0] == 5
```

### Validation Tests
Test model quality meets requirements.

```python
# tests/test_model_validation.py
def test_c_index_above_threshold():
    """Model achieves minimum required C-index."""
    model = load_trained_model('models/rsf.pkl')
    test_data = load_test_data()
    c_index = model.score(test_data.X, test_data.y)
    assert c_index > 0.70, f"C-index {c_index} below threshold 0.70"
```

---

## First Concrete Task

### File to Create: `src/data/loader.py`

### Function Signature:

```python
def load_dataset(data_dir: Path, subset: str = 'FD001') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NASA C-MAPSS dataset.

    Args:
        data_dir: Path to directory containing txt files
        subset: Dataset subset ('FD001', 'FD002', 'FD003', 'FD004')

    Returns:
        Tuple of (train_df, test_df) with columns:
        - engine_id: int
        - cycle: int
        - setting_1, setting_2, setting_3: float
        - s_1 through s_21: float (sensor readings)
    """
```

### Starter Code:

```python
"""NASA C-MAPSS dataset loader for turbofan engine degradation data."""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

# Column names for C-MAPSS dataset
COLUMN_NAMES = (
    ['engine_id', 'cycle'] +
    ['setting_1', 'setting_2', 'setting_3'] +
    [f's_{i}' for i in range(1, 22)]
)


def load_dataset(data_dir: Path, subset: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load NASA C-MAPSS turbofan engine degradation dataset.

    The C-MAPSS dataset contains run-to-failure data from simulated turbofan
    engines under different operating conditions and fault modes.

    Args:
        data_dir: Path to directory containing the txt data files
        subset: Which dataset subset to load. Options:
            - 'FD001': Single operating condition, single fault mode
            - 'FD002': Six operating conditions, single fault mode
            - 'FD003': Single operating condition, two fault modes
            - 'FD004': Six operating conditions, two fault modes

    Returns:
        Tuple of (train_df, test_df) DataFrames with columns:
            - engine_id: Unique identifier for each engine unit
            - cycle: Operational cycle number
            - setting_1, setting_2, setting_3: Operational settings
            - s_1 through s_21: Sensor measurements

    Example:
        >>> data_dir = Path('data/raw/CMAPSSData')
        >>> train_df, test_df = load_dataset(data_dir, 'FD001')
        >>> print(f"Train engines: {train_df['engine_id'].nunique()}")
        Train engines: 100
    """
    data_dir = Path(data_dir)

    # Validate subset
    valid_subsets = ['FD001', 'FD002', 'FD003', 'FD004']
    if subset not in valid_subsets:
        raise ValueError(f"subset must be one of {valid_subsets}, got '{subset}'")

    # Load training data
    train_path = data_dir / f'train_{subset}.txt'
    train_df = pd.read_csv(
        train_path,
        sep=r'\s+',  # Whitespace separated
        header=None,
        names=COLUMN_NAMES
    )

    # Load test data
    test_path = data_dir / f'test_{subset}.txt'
    test_df = pd.read_csv(
        test_path,
        sep=r'\s+',
        header=None,
        names=COLUMN_NAMES
    )

    return train_df, test_df


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life (RUL) column to training data.

    For training data, RUL = (max_cycle for engine) - current_cycle.
    This represents cycles until failure since training engines run to failure.

    Args:
        df: DataFrame with 'engine_id' and 'cycle' columns

    Returns:
        DataFrame with added 'rul' column

    Example:
        >>> train_df = compute_rul(train_df)
        >>> print(train_df[['engine_id', 'cycle', 'rul']].head())
           engine_id  cycle  rul
        0          1      1  191
        1          1      2  190
        2          1      3  189
    """
    df = df.copy()

    # Get max cycle per engine (this is when it failed)
    max_cycles = df.groupby('engine_id')['cycle'].max()

    # Map max cycles back and compute RUL
    df['max_cycle'] = df['engine_id'].map(max_cycles)
    df['rul'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])

    return df


def load_rul_targets(data_dir: Path, subset: str = 'FD001') -> pd.Series:
    """Load true RUL values for test set final cycles.

    The test data is cut off before failure. This function loads the
    true remaining cycles until failure for each engine's final observation.

    Args:
        data_dir: Path to directory containing RUL_*.txt files
        subset: Dataset subset to load

    Returns:
        Series indexed by engine_id with true RUL values
    """
    data_dir = Path(data_dir)
    rul_path = data_dir / f'RUL_{subset}.txt'

    rul_values = pd.read_csv(rul_path, header=None, names=['rul'])
    rul_values.index = rul_values.index + 1  # Engine IDs start at 1
    rul_values.index.name = 'engine_id'

    return rul_values['rul']


if __name__ == '__main__':
    # Quick test
    import sys

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/raw/CMAPSSData')

    print(f"Loading data from {data_dir}")
    train_df, test_df = load_dataset(data_dir, 'FD001')
    train_df = compute_rul(train_df)

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Number of training engines: {train_df['engine_id'].nunique()}")
    print(f"Number of test engines: {test_df['engine_id'].nunique()}")
    print(f"\nSample of training data with RUL:")
    print(train_df[['engine_id', 'cycle', 's_1', 's_2', 'rul']].head(10))
```

### Verification Method:

```bash
# Download C-MAPSS data first (free from NASA)
python src/data/loader.py data/raw/CMAPSSData
# Expected output: Train engines: 100, Test engines: 100
```

### First Commit Message:

```
Add C-MAPSS data loader with RUL computation

Implements data loading for NASA turbofan engine degradation dataset.
Handles all four subsets (FD001-FD004) with proper column naming
and RUL calculation for training data.
```

---

## Appendix: Concepts Requiring Deeper Understanding

Before implementing, ensure understanding of:

1. **Right Censoring**: When an engine hasn't failed yet, we know it survived *at least* this long, but not when it will fail. This is critical for survival analysis.

2. **Survival Function S(t)**: Probability of surviving past time t. S(0) = 1, S(∞) = 0.

3. **Hazard Function h(t)**: Instantaneous failure rate at time t given survival to t.

4. **C-index (Concordance)**: Measures ranking accuracy - probability that for two random samples, the one with higher predicted risk fails first.

5. **Cox Proportional Hazards Assumption**: Hazard ratios between samples remain constant over time. Violations require time-varying coefficients.

Recommended reading:
- [lifelines documentation - Introduction to Survival Analysis](https://lifelines.readthedocs.io/en/latest/Survival%20Analysis%20intro.html)
- [scikit-survival User Guide](https://scikit-survival.readthedocs.io/en/stable/user_guide/index.html)
