import datetime
from typing import Sequence, Optional, Type, Collection, Callable, Tuple
from warnings import warn

import torch
from scipy.special import expit
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

from torch_hlm.mixed_effects_module import MixedEffectsModule, LogisticMixedEffectsModule


class MixedEffectsModel(BaseEstimator):
    def __init__(self,
                 fixeff_cols: Sequence[str],
                 raneff_cols: Sequence[str],
                 group_colname: str,
                 response_colname: str,
                 response_type: str = 'gaussian',
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.LBFGS,
                 optimizer_kwargs: Optional[dict] = None):
        """
        :param fixeff_cols: Sequence of column-names in the DataFrame that will be used as fixed-effects predictors.
        :param raneff_cols: Sequence of column-names in the DataFrame that will be used as random-effects predictors.
        :param group_colname: The column-name in the DataFrame with the group indicator.
        :param response_colname: The column-name in the DataFrame with the response/target.
        :param response_type: Either 'binary' or 'gaussian'
        :param fixed_effects_nn: A torch.nn.Module that takes in the fixed-effects predictor-matrix and outputs
        predictions for the fixed-effects. Defaults to a single-layer Linear layer with bias.
        :param optimizer_cls: A torch.optim.Optimizer, defaults to LBFGS.
        :param optimizer_kwargs: Keyword-arguments for the optimizer.
        """
        self.response_colname = response_colname
        self.group_colname = group_colname
        self.raneff_cols = raneff_cols
        self.fixeff_cols = fixeff_cols
        self.response_type = response_type
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.fixed_effects_nn = fixed_effects_nn

        self.module_ = None
        self.fe_names_ = None
        self.re_names_ = None
        self.loss_history_ = {}

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> 'MixedEffectsModel':
        """
        Initialize and fit the underlying `MixedEffectsModule` until loss converges.

        :param X: A dataframe with the predictors for fixed and random effects, the group-id column, and the response
        column.
        :param y: Should always be left `None`
        :param kwargs: Further keyword-args to pass to partial_fit
        :return: This instance, now fitted.
        """
        if y is not None:
            warn(f"Ignoring `y` passed to {type(self).__name__}.fit().")

        # initialize module:
        self.module_ = self._initialize_module()

        # build-model-mat:
        X, y, group_ids = self.build_model_mats(X)
        if y is None:
            raise ValueError(f"`X` is missing column '{self.response_colname}'")

        # initialize covariance to a sensible default:
        std_devs = torch.ones(len(self.re_names_))
        std_devs[1:] *= .2 / np.sqrt(len(self.re_names_))
        self.module_.set_re_cov(torch.diag_embed(std_devs ** 2))

        # TODO: implement two-step approach
        self.partial_fit(X=X, y=y, group_ids=group_ids, max_num_epochs=None, **kwargs)
        return self

    def partial_fit(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    group_ids: Sequence,
                    optimize_re_cov: bool = True,
                    callbacks: Collection[Callable] = (),
                    max_num_epochs: Optional[int] = 1,
                    early_stopping: Tuple[float, int] = (.001, 2),
                    **kwargs) -> 'MixedEffectsModel':
        """
        (Partially) fit the underlying `MixedEffectsModule`.

        :param X: From `build_model_mats()`
        :param y: From `build_model_mats()`
        :param group_ids: From `build_model_mats()`
        :param optimize_re_cov: Should the random-effects covariance matrix be optimized?
        :param callbacks: Functions to call on each epoch. Takes a single argument, the loss-history for this call to
        partial_fit.
        :param max_num_epochs: The maximum number of epochs to fit. For similarity to other sklearn estimator's
        `partial_fit()` behavior, this default to 1, so that a single call to `partial_fit()` performs a single epoch.
        :param early_stopping: If `num_epochs > 1`, then early-stopping critierion can be supplied: a tuple with
        `(tolerance, patience)` (defaults to loss-diff < .001 for 2 iters).
        :param kwargs: Further keyword-arguments passed to `MixedEffectsModule.fit_loop()`
        :return: This instance
        """
        if self.module_ is None:
            self.module_ = self._initialize_module()

        if max_num_epochs is None:
            max_num_epochs = float('inf')
        assert max_num_epochs > 0
        tol, patience = early_stopping

        callbacks = list(callbacks)
        fit_loop = self.module_.fit_loop(
            X=X,
            y=y,
            group_ids=group_ids,
            optimizer=self._initialize_optimizer(optimize_re_cov),
            **kwargs
        )
        lh = self.loss_history_[datetime.datetime.utcnow()] = []
        epoch = 0
        while True:
            loss = next(fit_loop)
            lh.append(loss)
            for callback in callbacks:
                callback(lh)
            if len(lh) > patience:
                if pd.Series(lh).diff()[-patience:].abs().max() < tol:
                    print(f"Loss-diff < {tol} for {patience} iters in a row, stopping.")
                    break
            epoch += 1
            if epoch >= max_num_epochs:
                if max_num_epochs > 1:
                    print("Reached `max_num_epochs`, stopping.")
                break
        return self

    def build_model_mats(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_cols = sorted(list(self.raneff_cols) + list(self.fixeff_cols))
        X = np.empty((len(df.index), len(all_cols)))
        for i, col in enumerate(all_cols):
            X[:, i] = df[col].values
        group_ids = df[self.group_colname].values
        y = None  # ok to omit in `predict`
        if self.response_colname in df.columns:
            y = df[self.response_colname].values
        return X, y, group_ids

    def _initialize_module(self) -> MixedEffectsModule:
        self.fe_names_ = ['Intercept']
        self.re_names_ = ['Intercept']
        module_kwargs = {'fixeff_features': [], 'raneff_features': [], 'fixed_effects_nn': self.fixed_effects_nn}
        all_cols = sorted(list(self.raneff_cols) + list(self.fixeff_cols))
        for i, col in enumerate(all_cols):
            if col in self.raneff_cols:
                module_kwargs['raneff_features'].append(i)
                self.re_names_.append(col)
            if col in self.fixeff_cols:
                module_kwargs['fixeff_features'].append(i)
                self.fe_names_.append(col)
        return MixedEffectsModule.from_name(nm=self.response_type, **module_kwargs)

    def _initialize_optimizer(self, optimize_re_cov: bool) -> torch.optim.Optimizer:
        optimizer_kwargs = {'lr': .001}
        if issubclass(self.optimizer_cls, torch.optim.LBFGS):
            optimizer_kwargs.update(lr=.05 if optimize_re_cov else .25, max_eval=12)
        optimizer_kwargs.update(self.optimizer_kwargs or {})
        optimizer_kwargs['params'] = []
        if not optimize_re_cov:
            for p in self.module_.parameters():
                if not self.module_._is_cov_param(p):
                    optimizer_kwargs['params'].append(p)
        return self.optimizer_cls(**optimizer_kwargs)

    def predict(self, X: pd.DataFrame, group_data: pd.DataFrame) -> np.ndarray:
        """
        :param X: A dataframe with the predictors for fixed and random effects, as well as the group-id column.
        :param group_data: Historical data that will be used to obtain RE-estimates for each group, which will the be
        used to generate predictions for X. Same format as X, except the response-column must also be included.
        :return: An ndarray of predictions
        """
        yhat = self._predict(X=X, group_data=group_data)
        if isinstance(self.module_, LogisticMixedEffectsModule):
            return (yhat > 0).astype('float')
        else:
            return yhat

    def predict_proba(self, X: pd.DataFrame, group_data: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of positive class if response_type is 'binary'.
        :param X: A dataframe with the predictors for fixed and random effects, as well as the group-id column.
        :param group_data: Historical data that will be used to obtain RE-estimates for each group, which will the be
        used to generate predictions for X. Same format as X, except the response-column must also be included.
        :return: An ndarray of probabilities
        """
        yhat = self._predict(X=X, group_data=group_data)
        if isinstance(self.module_, LogisticMixedEffectsModule):
            return expit(yhat)
        else:
            raise TypeError("`predict_proba` unexpected unless `response_type` is binary")

    def _predict(self, X: pd.DataFrame, group_data: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            X, _, group_ids = self.build_model_mats(X)
            X_t, y_t, group_ids_t = self.build_model_mats(group_data)
            pred = self.module_(X, group_ids=group_ids, group_data=self.module_.make_group_data(X_t, y_t, group_ids_t))
            return pred.numpy()
