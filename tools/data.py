from typing import Optional, Union, NamedTuple
import shutil
import json
from copy import deepcopy
import datetime as dt
from pathlib import Path

import dill
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as sk_tt_split
import numpy as np
import pandas as pd

from .constants import SEED


def read_raw_data(filepath: Path):
    return pd.read_csv(filepath)


def train_test_split(dataset, target: str = 'Target', random_seed: int = SEED, test_pct: float = 0.2):
    return sk_tt_split(dataset, stratify=dataset[target], test_size=test_pct, random_state=random_seed)


def save_model(
        estimator: BaseEstimator,
        name: str,
        dirpath: Union[str, Path],
        metadata: Optional[dict] = None,
        overwrite: bool = False
):
    metadata = deepcopy(metadata or {})
    path = Path(dirpath) / name
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise ValueError('Cannot overwrite existing model')

    path.mkdir()

    metadata['timestamp'] = dt.datetime.utcnow().isoformat()
    with (path / 'metadata.json').open('w') as f:
        json.dump(metadata, f, indent=2)

    with (path / f'model_{name}.pkl').open('wb') as f:
        dill.dump(estimator, f)


def load_model(name: str, dirpath: Union[str, Path]):

    path = Path(dirpath) / name
    with (path / 'metadata.json').open('r') as f:
        metadata = json.load(f)

    with (path / f'model_{name}.pkl').open('rb') as f:
        estimator = dill.load(f)

    return ModelData(estimator=estimator, metadata=metadata)


class ModelData(NamedTuple):
    estimator: BaseEstimator
    metadata: dict
