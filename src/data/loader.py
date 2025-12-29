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
