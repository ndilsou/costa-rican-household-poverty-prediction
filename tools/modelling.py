from functools import partial
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection
import seaborn as sns
from pathlib import Path
import numpy as np
from .constants import SEED

macro_f1_score = partial(metrics.f1_score, average='macro')


def cross_validate_estimator(
        estimator,
        X,
        y,
        scorer=macro_f1_score,
        sample_weight=None,
        n_splits=5,
        test_size=0.2,
        random_seed=SEED
):
    sample_weight = sample_weight if sample_weight is not None else np.ones(len(y))
    ss_split = model_selection.StratifiedShuffleSplit(n_splits=n_splits,
                                                      test_size=test_size,
                                                      random_state=random_seed)
    scores = np.empty(n_splits)
    for i, (train_index, test_index) in enumerate(ss_split.split(X, y)):
        estimator.fit(X[train_index], y[train_index])
        y_pred = estimator.predict(X[test_index])
        score = scorer(y[test_index], y_pred, sample_weight=sample_weight[test_index])
        scores[i] = score
    return scores.mean(), scores.std()


def show_confusion_matrix(estimator, X, y, sample_weight=None, test_size=0.2, binarizer=None, random_seed=SEED):
    sample_weight = sample_weight if sample_weight is not None else np.ones(len(y))
    x_train, x_test, y_train, y_test, _, sw_test = train_test_split(X, y, sample_weight, test_size=test_size, random_state=random_seed)
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    if binarizer:
        test_labels = binarizer.inverse_transform(y_test)
        pred_labels = binarizer.inverse_transform(y_pred)
    else:
        test_labels = y_test
        pred_labels = y_pred

    cm = metrics.confusion_matrix(test_labels, pred_labels, sample_weight=sw_test)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, vmin=0, vmax=1, annot=True)
    return cm


def tuning_report(results, n_top=3):
    '''
    credit: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    '''
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def test_report(estimator, X, y, sample_weight, scorer=macro_f1_score, binarizer=None):

    y_pred = estimator.predict(X)
    score = scorer(y, y_pred, sample_weight=sample_weight)

    print(f'score: {score}')
    if binarizer:
        true_labels = binarizer.inverse_transform(y)
        pred_labels = binarizer.inverse_transform(y_pred)
    else:
        true_labels = y
        pred_labels = y_pred
    cm = metrics.confusion_matrix(true_labels, pred_labels, sample_weight=sample_weight)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, vmin=0, vmax=1, annot=True)


