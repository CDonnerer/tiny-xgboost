"""Benchmark tiny-xgboost against xgboost.
"""
import time
from contextlib import contextmanager

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from tiny_xgboost import TinyXGBRegressor, SquaredError


@contextmanager
def timed(msg=None) -> float:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    print(f"Time {msg}: {time.perf_counter() - start:.3f} seconds")


def main():
    data = fetch_california_housing()
    limit = None
    random_state = 11
    print(f"Data size = {data.data.shape}")

    X, y = data.data[:limit], data.target[:limit]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, random_state=random_state
    )

    txgb = TinyXGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=100,
        early_stopping_rounds=10,
        learning_rate=0.3,
        base_score=0.5,
        reg_lambda=1.0,
    )
    with timed("tiny-XGB"):
        txgb.fit(X_train, y_train, eval_set=(X_eval, y_eval), verbose=False)
        y_pred_txgb = txgb.predict(X_test)

    xgb = XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=100,
        early_stopping_rounds=10,
        learning_rate=0.3,
        base_score=0.5,
        tree_method="exact",
        reg_lambda=1.0,
    )
    with timed("XGB"):
        xgb.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
        y_pred_xgb = xgb.predict(X_test)

    txgb_score = SquaredError().loss(y_test, y_pred_txgb)
    xgb_score = SquaredError().loss(y_test, y_pred_xgb)

    print(f"tiny-XGB test-rmse = {txgb_score:.5f}")
    print(f"XGB test-rmse = {xgb_score:.5f}")

    # Compare the grown trees
    for tree_num in [
        81,
    ]:
        print(f"tiny-XGB tree {tree_num}:")
        print(txgb.trees[tree_num].get_dump())

        bst = xgb.get_booster()
        trees = bst.get_dump(with_stats=True)
        print(f"XGB tree {tree_num}:")
        print(trees[tree_num])


if __name__ == "__main__":
    main()
