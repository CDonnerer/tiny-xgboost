import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import make_regression

from tiny_xgboost import TinyXGBRegressor


def test_tiny_xgboost_regression():
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=5, n_targets=1, random_state=12
    )

    learner_params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "n_estimators": 100,
        "early_stopping_rounds": 10,
        "learning_rate": 0.3,
        "base_score": 0.5,
        "tree_method": "exact",
        "reg_lambda": 1.0,
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