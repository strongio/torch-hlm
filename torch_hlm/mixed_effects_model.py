from typing import Sequence, Optional, Type, Collection, Callable, Tuple, Dict, Union
from warnings import warn

import torch

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

from torch_hlm.mixed_effects_module import MixedEffectsModule
from torch_hlm.mixed_effects_module.utils import get_to_kwargs


class MixedEffectsModel(BaseEstimator):
    def __init__(self,
                 fixeff_cols: Sequence[str],
                 raneff_design: Dict[str, Sequence[str]],
                 response_colname: str,
                 response_type: str = 'gaussian',
                 module_kwargs: Optional[dict] = None,
                 optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.LBFGS,
                 optimizer_kwargs: Optional[dict] = None):
        """
        :param fixeff_cols: Sequence of column-names that will be used as fixed-effects predictors.
        :param raneff_design: A dictionary. If there is a single grouping-factor, the key in this dictionary indicates
        the column-name of the group-indicator; the value of this dictionary is the sequence of random-effects
        predictors. If there are multiple grouping-factors, then this dictionary will have multiple entries.
        :param response_colname: The column-name in the DataFrame with the response/target.
        :param response_type: Either 'binary' or 'gaussian'
        :param module_kwargs: Keyword-arguments for the `MixedEffectsModule`. One worth highlighting is
        `fixed_effects_nn` -- an optional torch.nn.Module that takes in the fixed-effects predictor-matrix and outputs
        predictions for the fixed-effects (defaults to a single-layer Linear with bias).
        :param optimizer_cls: A `torch.optim.Optimizer`, defaults to LBFGS.
        :param optimizer_kwargs: Keyword-arguments for the optimizer.
        """
        self.response_colname = response_colname
        self.fixeff_cols = fixeff_cols
        self.raneff_design = raneff_design
        self.response_type = response_type
        self.module_kwargs = module_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        self.module_ = None
        self.optimizer_ = None
        self.loss_history_ = []

        # preserve the ordering where possible
        assert len(self.fixeff_cols) == len(set(self.fixeff_cols)), "Duplicate `fixeff_cols`"
        self.all_model_mat_cols = list(self.fixeff_cols)
        for gf, gf_cols in self.raneff_design.items():
            assert len(gf_cols) == len(set(gf_cols)), f"Duplicate cols for '{gf}'"
            for col in gf_cols:
                if col in self.all_model_mat_cols:
                    continue
                self.all_model_mat_cols.append(col)

    def fix_re_cov(self, cov: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Fix the random-effects covariance based on known or plausible values.

        :param cov:
        :return:
        """
        self.module_ = self._initialize_module()
        if len(self.grouping_factors) == 1 and not isinstance(cov, dict):
            cov = {self.grouping_factors[0]: cov}
        for gf, gf_cov in cov.items():
            self.module_.set_re_cov(gf, gf_cov)
            for p in self.module_.covariance[gf].parameters():
                p.requires_grad_(False)

    @property
    def grouping_factors(self) -> Sequence[str]:
        return list(self.raneff_design.keys())

    def fit(self,
            X: pd.DataFrame,
            y=None,
            reset: str = 'warn',  # TODO: none?
            re_cov: Union[bool, torch.Tensor] = True,  # TODO: none?
            **kwargs) -> 'MixedEffectsModel':
        """
        Initialize and fit the underlying `MixedEffectsModule` until loss converges.

        :param X: A dataframe with the predictors for fixed and random effects, the group-id column, and the response
        column.
        :param y: Should always be left `None`
        :param reset: For multiple calls to fit, should we reset? Defaults to yes with a warning.
        :param re_cov: XXX
        :param kwargs: Further keyword-args to pass to partial_fit
        :return: This instance, now fitted.
        """
        if y is not None:
            warn(f"Ignoring `y` passed to {type(self).__name__}.fit().")

        # initialize module:
        if reset or not self.module_:
            if reset == 'warn' and self.module_:
                warn("Resetting module.")
            self.module_ = self._initialize_module()

            for gf, idx in self.module_.rf_idx.items():
                std_devs = torch.ones(len(idx) + 1)
                if len(idx):
                    std_devs[1:] *= (.5 / np.sqrt(len(idx)))
                self.module_.set_re_cov(gf, cov=torch.diag_embed(std_devs ** 2))

        # build-model-mat:
        X, y, group_ids = self.build_model_mats(X)
        if y is None:
            raise ValueError(f"`X` is missing column '{self.response_colname}'")

        self.partial_fit(X=X, y=y, group_ids=group_ids, max_num_epochs=None, re_cov=re_cov, **kwargs)
        return self

    def partial_fit(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    group_ids: Sequence,
                    re_cov: bool = True,
                    callbacks: Collection[Callable] = (),
                    max_num_epochs: Optional[int] = 1,
                    early_stopping: Tuple[float, int] = (.001, 2),
                    **kwargs) -> 'MixedEffectsModel':
        """
        (Partially) fit the underlying `MixedEffectsModule`.

        :param X: From `build_model_mats()`
        :param y: From `build_model_mats()`
        :param group_ids: From `build_model_mats()`
        :param re_cov: Should the random-effects covariance matrix be optimized?
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

        if isinstance(re_cov, (dict, torch.Tensor)):
            self.fix_re_cov(re_cov)
        elif isinstance(re_cov, bool):
            for p in self.module_.parameters():
                if self.module_._is_cov_param(p):
                    p.requires_grad_(re_cov)

        if self.optimizer_ is None:
            self.optimizer_ = self._initialize_optimizer()

        if max_num_epochs is None:
            max_num_epochs = float('inf')
        assert max_num_epochs > 0
        tol, patience = early_stopping

        X = torch.as_tensor(X, **get_to_kwargs(self.module_))
        y = torch.as_tensor(y, **get_to_kwargs(self.module_))

        callbacks = list(callbacks)
        fit_loop = self.module_.fit_loop(
            X=X,
            y=y,
            group_ids=group_ids,
            optimizer=self.optimizer_,
            **kwargs
        )
        self.loss_history_.append([])  # a separate loss-history for each call to `fit()`
        lh = self.loss_history_[-1]
        epoch = 0
        while True:
            try:
                loss = next(fit_loop)
                lh.append(loss)
                for callback in callbacks:
                    callback(lh)
            except KeyboardInterrupt:
                break

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

    def build_model_mats(self, df: pd.DataFrame) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray]:
        X = np.empty((len(df.index), len(self.all_model_mat_cols)), dtype='float32')
        for i, col in enumerate(self.all_model_mat_cols):
            if df[col].isnull().any():
                raise ValueError(f"Nans in `{col}`")
            X[:, i] = df[col].values
        X = torch.as_tensor(X, **get_to_kwargs(self.module_))
        group_ids = df[self.grouping_factors].values
        y = None  # ok to omit in `predict`
        if self.response_colname in df.columns:
            y = df[self.response_colname].values
            y = torch.as_tensor(y, **get_to_kwargs(self.module_))
        return X, y, group_ids

    def _initialize_module(self) -> MixedEffectsModule:
        module_kwargs = {
            'fixeff_features': [],
            'raneff_features': {gf: [] for gf in self.grouping_factors},
        }
        if self.module_kwargs:
            module_kwargs.update(self.module_kwargs)

        # organize model-mat indices:
        for i, col in enumerate(self.all_model_mat_cols):
            for gf, gf_cols in self.raneff_design.items():
                if col in gf_cols:
                    module_kwargs['raneff_features'][gf].append(i)
            if col in self.fixeff_cols:
                module_kwargs['fixeff_features'].append(i)

        # initialize module:
        return MixedEffectsModule.from_name(self.response_type, **module_kwargs)

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        optimizer_kwargs = {'lr': .001}
        if issubclass(self.optimizer_cls, torch.optim.LBFGS):
            optimizer_kwargs.update(lr=.10, max_eval=12)
        optimizer_kwargs.update(self.optimizer_kwargs or {})
        optimizer_kwargs['params'] = [p for p in self.module_.parameters() if p.requires_grad]
        return self.optimizer_cls(**optimizer_kwargs)

    def predict(self, X: pd.DataFrame, group_data: pd.DataFrame) -> np.ndarray:
        """
        :param X: A dataframe with the predictors for fixed and random effects, as well as the group-id column.
        :param group_data: Historical data that will be used to obtain RE-estimates for each group, which will the be
        used to generate predictions for X. Same format as X, except the response-column must also be included.
        :return: An ndarray of predictions
        """
        return self._predict(X=X, group_data=group_data)

    @torch.no_grad()
    def _predict(self, X: pd.DataFrame, group_data: pd.DataFrame) -> np.ndarray:
        X, _, group_ids = self.build_model_mats(X)
        X_t, y_t, group_ids_t = self.build_model_mats(group_data)
        if y_t is None:
            raise RuntimeError(
                f"`group_data` must include the response ('{self.response_colname}'), so that random-effects can "
                f"be computed."
            )
        pred = self.module_(X, group_ids=group_ids, re_solve_data=(X_t, y_t, group_ids_t))
        return pred.numpy()
