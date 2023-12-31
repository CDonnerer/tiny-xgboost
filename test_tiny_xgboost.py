import pytest

import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from tiny_xgboost import TinyXGBRegressor


@pytest.mark.parametrize("n_targets", [1, 2])
def test_tiny_xgboost_regression(n_targets):
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_targets=n_targets,
        random_state=123,
    )

    learner_params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "n_estimators": 100,
        "early_stopping_rounds": 10,
        "learning_rate": 0.3,
        "base_score": 0.5,
        "tree_method": "exact",
        "reg_lambda": 1,
        "min_child_weight": 1,
    }

    xgb = XGBRegressor(**learner_params)
    txgb = TinyXGBRegressor(**learner_params)

    xgb.fit(X, y, eval_set=[(X, y)], verbose=False)
    txgb.fit(X, y, eval_set=(X, y), verbose=False)

    for ii in range(1, learner_params["n_estimators"], 5):
        preds_tiny = txgb.predict(X, iteration_range=(0, ii))
        preds_og = xgb.predict(X, iteration_range=(0, ii))

        np.testing.assert_allclose(preds_tiny, preds_og, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("multi_strategy", ["multi_output_tree", "one_output_per_tree"])
def test_tiny_xgboost_multi_output_regression(multi_strategy):
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

    learner_params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "n_estimators": 400,
        "early_stopping_rounds": 20,
        "learning_rate": 0.3,
        "base_score": 0.5,
        "tree_method": "exact",
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "multi_strategy": multi_strategy,
    }

    txgb = TinyXGBRegressor(**learner_params)
    txgb.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=True)
