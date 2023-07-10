"""Tiny xgboost implementation in Python & numpy.

Currently only supports reg:squarederror objective.
"""
from dataclasses import dataclass

import numpy as np

# The 'official' xgboost fudge factor
# https://github.com/dmlc/xgboost/blob/master/include/xgboost/base.h#L315
KRT_EPS = np.float64(1e-6)


@dataclass
class SplitPoint:
    gain: float = None
    feature_id: int = None
    feature_value: float = None
    left_ids: np.ndarray = None
    right_ids: np.ndarray = None


def find_best_split(*, X, grad, hess, lambd, gamma, min_child_weight):
    grad_sum = np.sum(grad)
    hess_sum = np.sum(hess)

    best_gain = 0.0

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
        )

        # Null out any gains that would results in leaves below min_child_weight
        below_min_child_weight = (hess_left_cumsum < min_child_weight) | (
            hess_right_cumsum < min_child_weight
        )
        all_split_gains[below_min_child_weight] = np.float64(0.0)

        split_id = all_split_gains.argmax()
        current_gain = all_split_gains[split_id]

        if current_gain > best_gain:
            best_gain = current_gain
            best_feature_id = feature_id
            # XGB seems to put the split midway between points
            best_val = np.mean(f_unique_sorted[split_id : split_id + 2])
            below_split = X[:, feature_id] < best_val
            left_ids = np.flatnonzero(below_split)
            right_ids = np.flatnonzero(~below_split)

    if best_gain <= KRT_EPS:
        return None  # stop if we can't find a good split
    else:
        return SplitPoint(
            gain=best_gain,
            feature_id=best_feature_id,
            feature_value=best_val,
            left_ids=left_ids,
            right_ids=right_ids,
        )


def calc_leaf_weight(grad, hess, lambd):
    return -np.sum(grad) / (np.sum(hess) + lambd)


class BaseNode:
    def __init__(self):
        self.is_leaf = False
        self.weight = None
        self.cover = None

        self.split_point = None
        self.left_child = None
        self.right_child = None

    def split(self, *, X, grad, hess, depth, params):
        if depth == params.max_depth:
            self._set_leaf_node(grad=grad, hess=hess, params=params)
            return

        split_point = self._find_best_split(
            X=X,
            grad=grad,
            hess=hess,
            params=params,
        )

        if split_point is None:
            # We get here when no split produced an increase in gain
            self._set_leaf_node(grad=grad, hess=hess, params=params)
            return

        else:
            self.split_point = split_point
            self.cover = len(grad)

            self.left_child = self.__class__()
            self.left_child.split(
                X=X[self.split_point.left_ids],
                grad=grad[self.split_point.left_ids],
                hess=hess[self.split_point.left_ids],
                depth=depth + 1,
                params=params,
            )

            self.right_child = self.__class__()
            self.right_child.split(
                X=X[self.split_point.right_ids],
                grad=grad[self.split_point.right_ids],
                hess=hess[self.split_point.right_ids],
                depth=depth + 1,
                params=params,
            )

    def _find_best_split(self, X, grad, hess, params):
        pass

    def _set_leaf_node(self, grad, hess, params):
        pass

    def predict(self, X):
        if self.is_leaf:
            # below works for both scalar and array weight
            return np.ones(X.shape[0], dtype="float64") * self.weight
        else:
            below_split = (
                X[:, self.split_point.feature_id] < self.split_point.feature_value
            )
            left_ids = np.flatnonzero(below_split)
            right_ids = np.flatnonzero(~below_split)

            left_preds = self.left_child.predict(X[left_ids, :])
            right_preds = self.right_child.predict(X[right_ids, :])

            preds = np.zeros(shape=X.shape[0], dtype="float64")
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
                f"{indent}[f{self.split_point.feature_id}"
                f"<{self.split_point.feature_value:.5f}] "
                f"gain={self.split_point.gain:.5f},cover={self.cover}\n"
            )
            out_str += self.left_child.get_dump(depth=depth + 1)
            out_str += self.right_child.get_dump(depth=depth + 1)
            return out_str


class ScalarNode(BaseNode):
    def _find_best_split(self, X, grad, hess, params):
        return find_best_split(
            X=X,
            grad=grad,
            hess=hess,
            lambd=params.reg_lambda,
            gamma=params.gamma,
            min_child_weight=params.min_child_weight,
        )

    def _set_leaf_node(self, grad, hess, params):
        self.is_leaf = True
        self.cover = len(grad)
        leaf_weight = calc_leaf_weight(grad, hess, params.reg_lambda)
        self.weight = params.learning_rate * leaf_weight


class VectorNode(BaseNode):
    def _find_best_split(self, X, grad, hess, params):
        grad_output_summed = grad.sum(axis=1)
        hess_output_summed = hess.sum(axis=1)

        return find_best_split(
            X=X,
            grad=grad_output_summed,
            hess=hess_output_summed,
            lambd=params.reg_lambda,
            gamma=params.gamma,
            min_child_weight=params.min_child_weight,
        )

    def _set_leaf_node(self, grad, hess, params):
        self.is_leaf = True
        self.cover = len(grad)

        weights = []

        for grad_output, hess_output in zip(grad.T, hess.T):
            leaf_weight = calc_leaf_weight(grad_output, hess_output, params.reg_lambda)
            weights.append(leaf_weight)

        self.weight = params.learning_rate * np.array(weights).reshape(-1, 1)

    def predict(self, X):

        if self.is_leaf:
            # below works for both scalar and array weight
            return (np.ones(X.shape[0], dtype="float64") * self.weight).T
        else:
            below_split = (
                X[:, self.split_point.feature_id] < self.split_point.feature_value
            )
            left_ids = np.flatnonzero(below_split)
            right_ids = np.flatnonzero(~below_split)

            left_preds = self.left_child.predict(X[left_ids, :])
            right_preds = self.right_child.predict(X[right_ids, :])

            preds = np.zeros(shape=(X.shape[0], left_preds.shape[1]), dtype="float64")
            preds[left_ids] = left_preds
            preds[right_ids] = right_preds
            return preds


class Tree:
    def __init__(self):
        self._root = ScalarNode()

    def boost(self, *, X, grad, hess, params):
        self._root.split(X=X, grad=grad, hess=hess, depth=0, params=params)

    def predict(self, X):
        return self._root.predict(X)

    def get_dump(self):
        return self._root.get_dump(depth=0)


class MultiOutputTree:
    def __init__(self):
        self._root = VectorNode()

    def boost(self, *, X, grad, hess, params):
        self._root.split(X=X, grad=grad, hess=hess, depth=0, params=params)

    def predict(self, X):
        return self._root.predict(X)

    def get_dump(self):
        return self._root.get_dump(depth=0)


class SquaredError:
    def gradient_and_hessian(self, y, preds):
        grad = preds - y
        hess = np.full(y.shape, 1.0, dtype="float64")
        return grad, hess

    def loss(self, y, preds):
        return np.sqrt(np.mean(np.square(y - preds)))  # RMSE for consistency with xgb


_objectives = {"reg:squarederror": SquaredError}


@dataclass
class XGBParams:
    objective: str = "reg:squarederror"
    gamma: float = 0.0
    reg_lambda: float = 1.0
    max_depth: int = 3
    learning_rate: float = 0.3
    n_estimators: int = 100
    early_stopping_rounds: int = 10
    base_score: float = 0.5
    min_child_weight: float = 1.0
    tree_method: str = "exact"
    multi_strategy: str = "one_output_per_tree"
    num_outputs: int = 1

    def __post_init__(self):
        assert self.objective in _objectives.keys()
        assert self.tree_method == "exact"
        assert self.multi_strategy in ("one_output_per_tree", "multi_output_tree")


def _reshape_2d(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return x


def _as_float64(*args):
    if len(args) == 1:
        return args[0].astype("float64")
    else:
        return tuple(arg.astype("float64") for arg in args)


class Booster:
    def __init__(self, params):
        self._params = params
        self.objective = _objectives[self._params.objective]()

    def fit(self, X, y, *, eval_set=None, verbose=True):
        # TODO: only set those if eval loss
        self.best_val_loss = np.finfo("float64").max
        self.best_iteration = None

        X, y = _as_float64(X, y)
        X_val, y_val = _as_float64(eval_set[0], eval_set[1])

        y = _reshape_2d(y)
        y_val = _reshape_2d(y_val)

        self.num_outputs = y.shape[1]

        predictions = np.full(y.shape, self._params.base_score, dtype="float64")
        # TODO: this will break if no eval_set is provided
        eval_predictions = np.full(
            y_val.shape, self._params.base_score, dtype="float64"
        )

        if self._params.multi_strategy == "one_output_per_tree":
            self.trees = [[] for ii in range(self.num_outputs)]
        else:
            self.trees = [[]]

        for ii in range(self._params.n_estimators):
            grad, hess = self.objective.gradient_and_hessian(y, predictions)

            if self._params.multi_strategy == "multi_output_tree":
                tree = MultiOutputTree()
                tree.boost(X=X, grad=grad, hess=hess, params=self._params)
                self.trees[0].append(tree)
            else:
                for jj in range(self.num_outputs):
                    tree = Tree()
                    tree.boost(
                        X=X, grad=grad[:, jj], hess=hess[:, jj], params=self._params
                    )
                    self.trees[jj].append(tree)

            predictions += self.predict(
                X, iteration_range=(ii, ii + 1), include_base_score=False, training=True
            )

            train_loss = self.objective.loss(y, predictions)
            eval_predictions += self.predict(
                X_val,
                iteration_range=(ii, ii + 1),
                include_base_score=False,
                training=True,
            )
            val_loss = self.objective.loss(y_val, eval_predictions)

            if verbose:
                print(f"[{ii}]\ttrain-loss={train_loss:.5f}, val-loss={val_loss:.5f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_iteration = ii

            if (ii - self.best_iteration) >= self._params.early_stopping_rounds:
                break

        return self

    def predict(self, X, iteration_range=None, include_base_score=True, training=False):
        X = _as_float64(X)

        if iteration_range is None:
            if self.best_iteration:
                iteration_range = (0, self.best_iteration + 1)
            else:
                iteration_range = (0, len(self.trees))

        if self._params.multi_strategy == "multi_output_tree":
            predictions = np.sum(
                [
                    tree.predict(X)
                    for tree in self.trees[0][iteration_range[0] : iteration_range[1]]
                ],
                axis=0,
            )
        else:
            predictions = np.array(
                [
                    np.sum(
                        [
                            tree.predict(X)
                            for tree in self.trees[jj][
                                iteration_range[0] : iteration_range[1]
                            ]
                        ],
                        axis=0,
                    )
                    for jj in range(self.num_outputs)
                ]
            ).T

        if include_base_score:
            predictions += self._params.base_score

        if not training:
            predictions = predictions.squeeze()
        return predictions


class TinyXGBRegressor:
    def __init__(self, **params):
        self.params = XGBParams(**params)
        self.objective = _objectives[self.params.objective]()

    def fit(self, X, y, *, eval_set=None, verbose=True):
        self._Booster = Booster(self.params).fit(
            X, y, eval_set=eval_set, verbose=verbose
        )
        return self

    def predict(self, *args, **kwargs):
        return self._Booster.predict(*args, **kwargs)
