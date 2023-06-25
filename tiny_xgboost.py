"""Tiny xgboost implementation in Python & numpy.

Currently only supports reg:squarederror objective.
"""
from dataclasses import dataclass

import numpy as np


def find_best_split(X, grad, hess, lambd, gamma):
    grad_sum = np.sum(grad)
    hess_sum = np.sum(hess)

    best_gain = 0.0
    best_feature_id = None
    best_val = None
    best_left_instance_ids = None
    best_right_instance_ids = None

    for feature_id in range(X.shape[1]):
        f_unique_sorted, idx = np.unique(X[:, feature_id], return_inverse=True)

        # we first sum grads over identical feature vals
        grad_unique = np.bincount(idx, grad.ravel())
        hess_unique = np.bincount(idx, hess.ravel())
        # then do a cumsum to allow for fast finding of best split point
        grad_left_cumsum = np.cumsum(grad_unique, axis=0)
        hess_left_cumsum = np.cumsum(hess_unique, axis=0)
        grad_right_cumsum = grad_sum - grad_left_cumsum
        hess_right_cumsum = hess_sum - hess_left_cumsum

        all_split_gains = (
            np.square(grad_left_cumsum) / (hess_left_cumsum + lambd)
            + np.square(grad_right_cumsum) / (hess_right_cumsum + lambd)
            - np.square(grad_sum) / (hess_sum + lambd)
            - gamma
            - 1e-3  # TODO: seem to need a relatively large fudge factor?
        )

        split_id = all_split_gains.argmax()
        current_gain = all_split_gains[split_id]

        if current_gain > best_gain:
            best_gain = current_gain
            best_feature_id = feature_id
            # XGB seems to put the split midway between points?
            best_val = 0.5 * (f_unique_sorted[split_id] + f_unique_sorted[split_id + 1])
            # TODO: can probably do the below quicker based on ids we have?
            below_split = X[:, feature_id] < best_val
            best_left_instance_ids = np.argwhere(below_split).flatten()
            best_right_instance_ids = np.argwhere(~below_split).flatten()

    if best_gain <= 0.0:
        # stop if we can't find a good split
        return None, None, None, None, None
    else:
        return (
            best_gain,
            best_feature_id,
            best_val,
            best_left_instance_ids,
            best_right_instance_ids,
        )


def calc_leaf_weight(grad, hess, lambd):
    return -np.sum(grad) / (np.sum(hess) + lambd)


class Node:
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.feature_id = None
        self.split_val = None
        self.weight = None
        self.best_gain = None
        self.cover = None

    def split(self, *, X, grad, hess, depth, params):
        if depth == params.max_depth:
            self._set_leaf_node(grad=grad, hess=hess, params=params)
            return

        gain, feature_id, split_val, left_ids, right_ids = find_best_split(
            X=X, grad=grad, hess=hess, lambd=params.reg_lambda, gamma=params.gamma
        )
        if gain is None:
            # We get here when no split produced an increase in gain
            self._set_leaf_node(grad=grad, hess=hess, params=params)
            return

        else:
            self.feature_id = feature_id
            self.split_val = split_val
            self.cover = len(grad)
            self.gain = gain

            self.left_child = Node()
            self.left_child.split(
                X=X[left_ids],
                grad=grad[left_ids],
                hess=hess[left_ids],
                depth=depth + 1,
                params=params,
            )

            self.right_child = Node()
            self.right_child.split(
                X=X[right_ids],
                grad=grad[right_ids],
                hess=hess[right_ids],
                depth=depth + 1,
                params=params,
            )

    def _set_leaf_node(self, grad, hess, params):
        self.is_leaf = True
        self.cover = len(grad)
        leaf_weight = calc_leaf_weight(grad, hess, params.reg_lambda)
        self.weight = params.learning_rate * leaf_weight

    def predict(self, X):
        if self.is_leaf:
            return np.full(X.shape[0], self.weight, dtype="float32")
        else:
            below_split = X[:, self.feature_id] < self.split_val
            left_ids = np.argwhere(below_split).flatten()
            right_ids = np.argwhere(~below_split).flatten()

            left_preds = self.left_child.predict(X[left_ids, :])
            right_preds = self.right_child.predict(X[right_ids, :])

            preds = np.zeros(shape=X.shape[0], dtype="float32")
            preds[left_ids] = left_preds
            preds[right_ids] = right_preds
            return preds

    def get_dump(self, depth=0):
        indent = "\t" * depth

        if self.is_leaf:
            out_str = f"{indent}leaf={self.weight:.5f} cover={self.cover}\n"
            return out_str
        else:
            out_str = (
                f"{indent}[f{self.feature_id}<{self.split_val:.5f}] "
                f"gain={self.gain:.5f},cover={self.cover}\n"
            )
            out_str += self.left_child.get_dump(depth=depth + 1)
            out_str += self.right_child.get_dump(depth=depth + 1)
            return out_str


class Tree:
    def __init__(self):
        self._root = Node()

    def boost(self,*, X, grad, hess, params):
        self._root.split(X=X, grad=grad, hess=hess, depth=0, params=params)

    def predict(self, X):
        return self._root.predict(X)

    def get_dump(self):
        return self._root.get_dump(depth=0)


class SquaredError:
    def gradient_and_hessian(self, y, preds):
        grad = preds - y
        hess = np.full(len(y), 1.0, dtype="float32")
        return grad, hess

    def loss(self, y, preds):
        return np.sqrt(np.mean(np.square(y - preds)))  # RMSE for consistency with xgb
    
_objectives = {
    "reg:squarederror": SquaredError
}

@dataclass
class XGBParams:
    objective : str = "reg:squarederror"
    gamma: float = 0.0
    reg_lambda: float = 1.0
    max_depth: int = 3
    learning_rate: float = 0.3
    n_estimators: int = 100
    early_stopping_rounds: int = 10
    base_score: float = 0.5


class TinyXGBRegressor:
    def __init__(self, **params):
        self.params = XGBParams(**params)
        self.best_iteration = None
        self.objective = _objectives[self.params.objective]()

    def fit(self, X, y, *, eval_set=None, verbose=True):
        self.trees = []
        self.best_val_loss = np.finfo("float32").max

        # internally, we are in float32 world
        X = X.astype("float32", copy=False)
        y = y.astype("float32", copy=False)

        predictions =  np.full(len(y), self.params.base_score, dtype="float32")
        # TODO: this will break if no eval_set is provided
        eval_predictions =  np.full(len(eval_set[1]), self.params.base_score, dtype="float32")
  
        for ii in range(self.params.n_estimators):
            grad, hess = self.objective.gradient_and_hessian(y, predictions)

            tree = Tree()
            tree.boost(X, grad, hess, self.params)
            self.trees.append(tree)

            predictions += self.predict(
                X, iteration_range=(ii, ii + 1), include_base_score=False
            )

            train_loss = self.objective.loss(y, predictions)
            eval_predictions += self.predict(
                eval_set[0], iteration_range=(ii, ii + 1), include_base_score=False
            )
            val_loss = self.objective.loss(eval_set[1], eval_predictions)

            if verbose:
                print(f"[{ii}]\ttrain-loss={train_loss:.5f}, val-loss={val_loss:.5f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_iteration = ii

            if (ii - self.best_iteration) >= self.params.early_stopping_rounds:
                break

    def predict(self, X, iteration_range=None, include_base_score=True):
        X = X.astype("float32", copy=False)

        if iteration_range is None:
            if self.best_iteration:
                iteration_range = (0, self.best_iteration + 1)
            else:
                iteration_range = (0, len(self.trees))

        predictions = np.sum(
            [
                tree.predict(X)
                for tree in self.trees[iteration_range[0] : iteration_range[1]]
            ],
            axis=0,
        )
        if include_base_score:
            predictions += self.params.base_score
        return predictions