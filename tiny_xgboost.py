"""Tiny xgboost implementation in Python & numpy.
"""
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from scipy.stats import norm


# The 'official' xgboost fudge factor
# https://github.com/dmlc/xgboost/blob/master/include/xgboost/base.h#L315
KRT_EPS = np.float64(1e-6)

MIN_EXPONENT = np.log(np.float64(1e-32))
MAX_EXPONENT = np.log(np.finfo("float64").max) - 1
# Note: due to reparameterization, we need to ensure that the converted
# variance, exp(2 * std), is within bounds of np.float32 arrays
MIN_LOG_SCALE = MIN_EXPONENT / 2
MAX_LOG_SCALE = MAX_EXPONENT / 2


@dataclass
class SplitPoint:
    gain: float = None
    feature_id: int = None
    feature_value: float = None
    left_ids: np.ndarray = None
    right_ids: np.ndarray = None
    all_feature_values : np.ndarray = None
    all_gains : np.ndarray = None
    all_obj_left : np.ndarray = None
    all_obj_right : np.ndarray = None
    all_obj_left_num : np.ndarray = None
    all_obj_right_num : np.ndarray = None
    all_obj_left_denom : np.ndarray = None
    all_obj_right_denom : np.ndarray = None
    all_grads_left_cumsum : np.ndarray = None
    all_grads_right_cumsum : np.ndarray = None
    all_hess_left_cumsum : np.ndarray = None
    all_hess_right_cumsum : np.ndarray = None


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
        all_split_gains[below_min_child_weight] = 0.0

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


def find_best_split_two_params(*, X, grad, hess, lambd, gamma, min_child_weight):
    grad_sum = np.sum(grad, axis=0)
    hess_sum = np.sum(hess, axis=0)

    best_gain = 0.0

    for feature_id in range(X.shape[1]):
        f_unique_sorted, idx = np.unique(X[:, feature_id], return_inverse=True)

        # we first sum grads over identical feature vals
        # TODO: the below should be possible in a single pass
        grad_unique = np.zeros(shape=(f_unique_sorted.shape[0], 2))
        grad_unique[:, 0] = np.bincount(idx, grad[:, 0].ravel())
        grad_unique[:, 1] = np.bincount(idx, grad[:, 1].ravel())

        hess_unique = np.zeros(shape=(f_unique_sorted.shape[0], 2, 2))
        hess_unique[:, 0, 0] = np.bincount(idx, hess[:, 0, 0].ravel())
        hess_unique[:, 0, 1] = np.bincount(idx, hess[:, 0, 1].ravel())
        hess_unique[:, 1, 0] = np.bincount(idx, hess[:, 1, 0].ravel())
        hess_unique[:, 1, 1] = np.bincount(idx, hess[:, 1, 1].ravel())

        # then do a cumsum to allow for fast finding of best split point
        # the summed values here correspond to each 'potential' leaf
        grad_left_cumsum = np.cumsum(grad_unique, axis=0)
        hess_left_cumsum = np.cumsum(hess_unique, axis=0)
        grad_right_cumsum = grad_sum - grad_left_cumsum
        hess_right_cumsum = hess_sum - hess_left_cumsum

        def objective_term(grad, hess):
            """
            obj = (
                (lambda + hess_11) * grad_0 ^ 2
                + (lambda + hess_00) * grad_1 ^ 2
                - hess_01 * grad_0 ^ 2
                - hess_10 * grad_1 ^ 2
            ) / (
                (lambda + hess_00) * (lambda * hess_11) - hess_01 * hess_01
            )
            """
            numerator = (
                ((lambd + hess[:, 1, 1]) * np.square(grad[:, 0]))
                + ((lambd + hess[:, 0, 0]) * np.square(grad[:, 1]))
                - 2 * (hess[:, 0, 1] * grad[:, 0] * grad[:, 1])
            )
            denominator = (lambd + hess[:, 0, 0]) * (lambd + hess[:, 1, 1]) - (
                hess[:, 0, 1] * hess[:, 1, 0]
            )
            return numerator / denominator, numerator, denominator
        
        obj_left, obj_left_num, obj_left_denom = objective_term(
            grad=grad_left_cumsum, hess=hess_left_cumsum
        )
        obj_right, obj_right_num, obj_right_denom = objective_term(
            grad=grad_right_cumsum, hess=hess_right_cumsum
        )
        pre_split_obj, _ ,_  = objective_term(
            grad=grad_sum[np.newaxis, :], hess=hess_sum[np.newaxis, :]
        )

        all_split_gains = (
            obj_left
            + obj_right 
            - pre_split_obj
            - gamma
        )

        # Null out any gains that would results in leaves below min_child_weight
        # TODO: How shoud this be implemented here?
        # below_min_child_weight = (hess_left_cumsum < min_child_weight) | (
        #     hess_right_cumsum < min_child_weight
        # )
        # all_split_gains[below_min_child_weight] = 0.0

        split_id = all_split_gains.argmax()

        current_gain = all_split_gains[split_id]

        if current_gain > best_gain:
            best_gain = current_gain
            best_feature_id = feature_id
            best_feature_vals = f_unique_sorted
            best_feature_gains = all_split_gains
            best_obj_left = obj_left
            best_obj_right = obj_right

            best_obj_left_num = obj_left_num
            best_obj_left_denom = obj_left_denom

            best_obj_right_num = obj_right_num
            best_obj_right_denom = obj_right_denom

            best_grad_left_cumsum = grad_left_cumsum
            best_grad_right_cumsum = grad_right_cumsum

            best_hess_left_cumsum = hess_left_cumsum
            best_hess_right_cumsum = hess_right_cumsum

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
            all_feature_values=best_feature_vals,
            all_gains=best_feature_gains,
            all_obj_left=best_obj_left,
            all_obj_right=best_obj_right,
            all_obj_left_num=best_obj_left_num,
            all_obj_right_num=best_obj_right_num,
            all_obj_left_denom=best_obj_left_denom,
            all_obj_right_denom=best_obj_right_denom,
            all_grads_left_cumsum=best_grad_left_cumsum,
            all_grads_right_cumsum=best_grad_right_cumsum,
            all_hess_left_cumsum=best_hess_left_cumsum,
            all_hess_right_cumsum=best_hess_right_cumsum,
        )


def calc_leaf_weight(grad, hess, lambd):
    return -np.sum(grad) / (np.sum(hess) + lambd)


class BaseNode:
    def __init__(self, tree_method=None):
        self.tree_method = tree_method
        self.is_leaf = False
        self.weight = None
        self.cover = None

        self.split_point = None
        self.left_child = None
        self.right_child = None

    def split(self, *, X, grad, hess, depth, params, debug):
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
            debug[f"depth_{depth}"] = asdict(split_point)
            self.split_point = split_point
            self.cover = len(grad)

            self.left_child = self.__class__(self.tree_method)
            self.left_child.split(
                X=X[self.split_point.left_ids],
                grad=grad[self.split_point.left_ids],
                hess=hess[self.split_point.left_ids],
                depth=depth + 1,
                params=params,
                debug=debug
            )

            self.right_child = self.__class__(self.tree_method)
            self.right_child.split(
                X=X[self.split_point.right_ids],
                grad=grad[self.split_point.right_ids],
                hess=hess[self.split_point.right_ids],
                depth=depth + 1,
                params=params,
                debug=debug
            )

    def _find_best_split(self, X, grad, hess, params):
        pass

    def _set_leaf_node(self, grad, hess, params):
        pass

    def predict(self, X):
        pass

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

    def predict(self, X):
        if self.is_leaf:
            return np.full(X.shape[0], self.weight, dtype="float64")
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


class VectorNode(BaseNode):
    def __init__(self, tree_method):
        super().__init__(tree_method=tree_method)
        if self.tree_method == TreeMethod.exact_2d:
            self._find_best_split = self._find_best_split_two_params
            self._set_leaf_node = self._set_leaf_node_two_params
        else:
            self._find_best_split = self._find_best_split_summed
            self._set_leaf_node = self._set_leaf_node_summed

    def _find_best_split_summed(self, X, grad, hess, params):
        grad_output_mean = grad.mean(axis=1)
        hess_output_mean = hess.mean(axis=1)

        return find_best_split(
            X=X,
            grad=grad_output_mean,
            hess=hess_output_mean,
            lambd=params.reg_lambda,
            gamma=params.gamma,
            min_child_weight=params.min_child_weight,
        )

    def _find_best_split_two_params(self, X, grad, hess, params):
        return find_best_split_two_params(
            X=X,
            grad=grad,
            hess=hess,
            lambd=params.reg_lambda,
            gamma=params.gamma,
            min_child_weight=params.min_child_weight,
        )

    def _set_leaf_node_summed(self, grad, hess, params):
        self.is_leaf = True
        self.cover = len(grad)

        weights = []

        for grad_output, hess_output in zip(grad.T, hess.T):
            leaf_weight = calc_leaf_weight(grad_output, hess_output, params.reg_lambda)
            weights.append(leaf_weight)

        self.weight = params.learning_rate * np.array(weights).reshape(-1, 1)

    def _set_leaf_node_two_params(self, grad, hess, params):
        """
        1 / (
            (lambda + hess_00) * (lambda + hess_11) - hess_01 ^ 2
        ) * [
            (hess_01 * grad_1) - (lambda + hess_11) * grad_0,
            (hess_01 * grad_0) - (lambda + hess_00) * grad_1,
        ]

        """
        self.is_leaf = True
        self.cover = len(grad)

        grad_0 = np.sum(grad[:, 0])
        grad_1 = np.sum(grad[:, 1])

        hess_00 = np.sum(hess[:, 0, 0])
        hess_01 = np.sum(hess[:, 0, 1])
        hess_11 = np.sum(hess[:, 1, 1])

        scale = 1 / (
            (params.reg_lambda + hess_00) * (params.reg_lambda + hess_11)
            - np.square(hess_01)
        )
        weights = scale * np.array(
            [
                ((hess_01 * grad_1) - (params.reg_lambda + hess_11) * grad_0),
                ((hess_01 * grad_0) - (params.reg_lambda + hess_00) * grad_1),
            ]
        )
        self.weight = params.learning_rate * weights.reshape(-1, 1)

    def predict(self, X):
        if self.is_leaf:
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


class BaseTree:
    def __init__(self):
        self._root = None

    def boost(self, *, X, grad, hess, params):
        debug = {}
        self._root.split(X=X, grad=grad, hess=hess, depth=0, params=params, debug=debug)
        return debug

    def predict(self, X):
        return self._root.predict(X)

    def get_dump(self):
        return self._root.get_dump(depth=0)


class Tree(BaseTree):
    def __init__(self):
        self._root = ScalarNode()


class MultiOutputTree(BaseTree):
    def __init__(self, tree_method):
        self._root = VectorNode(tree_method=tree_method)


class SquaredError:
    def gradient_and_hessian(self, y, preds):
        grad = preds - y
        hess = np.full(y.shape, 1.0, dtype="float64")
        return grad, hess

    def loss(self, y, preds):
        return np.sqrt(np.mean(np.square(y - preds)))  # RMSE for consistency with xgb


class NormalDistribution:
    """Normal distribution with log scoring

    f(x) = exp( -[ (y - mean) / std ]^2 / 2 ) / std

    We reparameterize:

        a = mean         |  a = mean
        b = log ( std )  |  e^b = std   |  e^(2b) = var  | e^(-2b) = 1 / var

    The gradients are:

    d/da -log[f(y)] = - e^(-2b) * (y-a)      = (a-y) / var
    d/db -log[f(y)] = 1 - e^(-2b) * (y-a)^2  = 1 - (y-a)^2 / var

    to second order:

    d2/da2     -log[f(y)] = e^(-2b)                = 1 / var
    d/da d/db  -log[f(y)] =  2 * e^(-2b) * (y-a)   = 2 * (y-a)   / var
    d2/db2     -log[f(y)] = -2 * e^(-2b) * (x-a)^2 = 2 * (y-a)^2 / var
    d/db d/da  -log[f(y)] = -2 * e^(-2b) * (x-a)   = 2 * (y-a)   / var
    """

    def gradient_and_hessian(self, y, preds):
        y_flat = y.squeeze()

        loc = preds[:, 0]
        log_scale = np.clip(preds[:, 1], a_min=MIN_LOG_SCALE, a_max=MAX_LOG_SCALE)

        var = np.exp(2 * log_scale)

        grad = np.zeros(shape=(len(y), 2), dtype="float64")
        grad[:, 0] = (loc - y_flat) / var
        grad[:, 1] = 1 - ((y_flat - loc) ** 2) / var

        hess = np.zeros(shape=(len(y), 2, 2), dtype="float64")
        hess[:, 0, 0] = 1 / var
        hess[:, 0, 1] = 2 * (y_flat - loc) / var
        hess[:, 1, 0] = hess[:, 0, 1]  # because symmetry
        hess[:, 1, 1] = 2 * ((y_flat - loc) ** 2) / var
        return grad, hess

    def loss(self, y, preds):
        loc = preds[:, 0]
        log_scale = np.clip(preds[:, 1], a_min=MIN_LOG_SCALE, a_max=MAX_LOG_SCALE)
        scale = np.exp(log_scale)
        return -norm.logpdf(y, loc=loc, scale=scale).mean()


_objectives = {
    "reg:squarederror": SquaredError,
    "distribution:normal": NormalDistribution,
}


class MultiStrategy(str, Enum):
    one_output_per_tree = "one_output_per_tree"
    multi_output_tree = "multi_output_tree"


class TreeMethod(str, Enum):
    exact = "exact"
    exact_2d = "exact_2d"


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
    tree_method: str = TreeMethod.exact
    multi_strategy: str = MultiStrategy.one_output_per_tree
    num_outputs: int = None

    def __post_init__(self):
        assert self.objective in _objectives.keys()
        assert self.tree_method in TreeMethod._member_names_
        assert self.multi_strategy in MultiStrategy._member_names_

        if self.objective == "distribution:normal":
            self.num_outputs = 2
            self.multi_strategy = MultiStrategy.multi_output_tree
            self.tree_method = TreeMethod.exact_2d


def reshape_2d(x):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return x


def as_float64(*args):
    if len(args) == 1:
        return args[0].astype("float64")
    else:
        return tuple(arg.astype("float64") for arg in args)


class Booster:
    def __init__(self, params):
        self._params = params
        self.objective = _objectives[self._params.objective]()

    def fit(self, X, y, *, eval_set=None, verbose=True):

        fit_tracker = {}

        # TODO: only set those if eval loss
        self.best_val_loss = np.finfo("float64").max
        self.best_iteration = None

        X, y = as_float64(X, y)
        # TODO: this will break if no eval_set is provided
        X_val, y_val = as_float64(eval_set[0], eval_set[1])

        y = reshape_2d(y)
        y_val = reshape_2d(y_val)

        if self._params.num_outputs:
            self.num_outputs = self._params.num_outputs
        else:
            self.num_outputs = y.shape[1]

        predictions = np.full((y.shape[0], self.num_outputs), self._params.base_score, dtype="float64")
        eval_predictions = np.full(
            (y_val.shape[0], self.num_outputs), self._params.base_score, dtype="float64"
        )

        if self._params.multi_strategy == MultiStrategy.multi_output_tree:
            self.trees = [[]]
        else:
            self.trees = [[] for _ in range(self.num_outputs)]

        for ii in range(self._params.n_estimators):
            fit_tracker[ii] = {}

            grad, hess = self.objective.gradient_and_hessian(y, predictions)
            fit_tracker[ii].update({"grad": grad, "hess": hess})

            if self._params.multi_strategy == MultiStrategy.multi_output_tree:
                tree = MultiOutputTree(tree_method=self._params.tree_method)
                split_log = tree.boost(X=X, grad=grad, hess=hess, params=self._params)
                self.trees[0].append(tree)
                fit_tracker[ii].update({"splits": split_log})

            else:
                for jj in range(self.num_outputs):
                    tree = Tree()
                    tree.boost(
                        X=X, grad=grad[:, jj], hess=hess[:, jj], params=self._params
                    )
                    self.trees[jj].append(tree)

            predictions += self.predict(
                X,
                iteration_range=(ii, ii + 1),
                output_margin=True,
                strict_shape=True,
            )

            train_loss = self.objective.loss(y, predictions)
            eval_predictions += self.predict(
                X_val,
                iteration_range=(ii, ii + 1),
                output_margin=True,
                strict_shape=True,
            )
            val_loss = self.objective.loss(y_val, eval_predictions)

            if verbose:
                print(f"[{ii}]\ttrain-loss={train_loss:.5f}, val-loss={val_loss:.5f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_iteration = ii

            if (ii - self.best_iteration) >= self._params.early_stopping_rounds:
                break

        return self, fit_tracker
        
    def predict(
        self, X, iteration_range=None, output_margin=False, strict_shape=False
    ):
        X = as_float64(X)

        if iteration_range is None:
            if self.best_iteration:
                iteration_range = (0, self.best_iteration + 1)
            else:
                iteration_range = (0, len(self.trees))

        if self._params.multi_strategy == MultiStrategy.multi_output_tree:
            predictions = self._predict_tree(
                X=X, iteration_range=iteration_range, tree_id=0
            )
        else:
            predictions = np.array(
                [
                    self._predict_tree(X=X, iteration_range=iteration_range, tree_id=jj)
                    for jj in range(self.num_outputs)
                ]
            ).T

        if not output_margin:
            predictions += self._params.base_score

            # depending on the objective, we do some post processing of outputs
            if self._params.objective == "distribution:normal":
                log_scale = np.clip(predictions[:, 1], a_min=MIN_LOG_SCALE, a_max=MAX_LOG_SCALE)
                scale = np.exp(log_scale)
                predictions[:, 1] = scale

        if not strict_shape:
            predictions = predictions.squeeze()
        return predictions

    def _predict_tree(self, X, iteration_range, tree_id=0):
        return np.sum(
            [
                tree.predict(X)
                for tree in self.trees[tree_id][iteration_range[0] : iteration_range[1]]
            ],
            axis=0,
        )


class TinyXGBRegressor:
    def __init__(self, **params):
        self.params = XGBParams(**params)
        self.objective = _objectives[self.params.objective]()

    def fit(self, X, y, *, eval_set=None, verbose=True, debug=False):
        self._Booster, fit_tracker = Booster(self.params).fit(
            X, y, eval_set=eval_set, verbose=verbose
        )
        if debug:
            return self, fit_tracker
        else:
            return self

    def predict(self, *args, **kwargs):
        return self._Booster.predict(*args, **kwargs)
