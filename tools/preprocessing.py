from typing import Sequence
from pathlib import Path
from sklearn.model_selection import train_test_split as sk_tt_split
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from .constants import SEED


def find_column_group(dataset: pd.DataFrame, prefix: str):
    cols = dataset.columns
    return cols[cols.str.startswith(prefix)]


def collapse_cat(dataset: pd.DataFrame, prefix: str, stringify: bool = False):
    cols = find_column_group(dataset, prefix)
    values = (dataset[cols] * np.arange(len(cols))).sum(axis='columns')
    if stringify:
        values = values.astype(str)
    return dataset.assign(**{prefix: values})\
        .drop(columns=cols)


def as_2darray(s: pd.Series):
    return s.values.reshape(-1, 1)


def code_dependents(s: pd.Series):
    return s.apply(code_dependents_func).values.reshape(-1, 1)


def code_dependents_func(v):
    if v in {'yes', 'no'}:
        return v

    v = float(v)
    # 0.5 would indicate that two adults care for a single dependent.
    if v > 0.5:
        code = 'many'
    else:
        code = 'yes'
    return code


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Sequence[str], drop: bool = True):
        self.columns = columns
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        try:
            if self.drop:
                selection = X.loc[:, ~X.columns.isin(self.columns)]
            else:
                selection = X[self.columns]
            return selection
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class ClusterIndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cluster):
        self.cluster = cluster

    def fit(self, X, y=None):
        self.cluster.fit(X)
        return self

    def transform(self, X):
        cluster_indices = self.cluster.predict(X)
        return cluster_indices.reshape(-1, 1)


class VulnerabilityIndicator(BaseEstimator, TransformerMixin):
    '''
    Adds a metafeature indicating if the datapoint
    belongs to a vulnerable household. Used to guide the downstream classifier.
    '''

    def __init__(self, loss='hinge', random_state=None):
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y):
        self.clf_ = SGDClassifier(loss=self.loss, random_state=self.random_state)
        y_ = y != 4
        self.clf_.fit(X, y_)
        return self

    def transform(self, X):
        y_hat = self.clf_.predict(X)
        return y_hat.reshape(-1, 1)
