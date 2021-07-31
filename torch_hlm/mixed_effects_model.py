from functools import cached_property
from time import sleep
from typing import Sequence, Optional, Type, Collection, Callable, Tuple, Dict, Union
from warnings import warn

import torch

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torch_hlm.mixed_effects_module import MixedEffectsModule
from torch_hlm.mixed_effects_module.utils import get_to_kwargs


class MixedEffectsModel(BaseEstimator):
    """
    :param fixeff_cols: Sequence of column-names of the fixed-effects in the model-matrix ``X``.
    :param raneff_design: A dictionary, whose key(s) are the names of grouping factors and whose values are
     column-names for random-effects of that grouping factor.
    :param response: Column-name of response variable.
    :param response_type: Either 'binary' or 'gaussian'
    :param fixed_effects_nn: A `torch.nn.Module`` that takes in the fixed-effects model-matrix and outputs predictions.
     Default is to use a single-layer Linear module.
    :param covariance: Can pass string with parameterization (see ``torch_hlm.covariance``, default is 'log_cholesky').
     Alternatively, can pass a dictionary with keys as grouping factors and values as tensors. These will be used as
     the covariances for those grouping factors' random-effects, which will *not* be optimized (i.e. their
     ``requires_grad`` flag will be set to False and they will not be passed to the optimizer). This can be used
     in setting where optimizing the random-effects covariance is not supported.
    :param module_kwargs: Other keyword-arguments for the ``MixedEffectsModule``.
    :param optimizer_cls: A ``torch.optim.Optimizer``, defaults to LBFGS.
    :param optimizer_kwargs: Keyword-arguments for the optimizer.
    """

    def __init__(self,
                 fixeff_cols: Sequence[str],
                 raneff_design: Dict[str, Sequence[str]],
                 response: str,
                 response_type: str = 'gaussian',
                 loss_type: Optional[str] = None,
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 covariance: Union[str, Dict[str, torch.Tensor]] = 'log_cholesky',
                 module_kwargs: Optional[dict] = None,
                 optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.LBFGS,
                 optimizer_kwargs: Optional[dict] = None):
        self.fixeff_cols = fixeff_cols
        self.raneff_design = raneff_design
        self.response = response
        self.response_type = response_type
        self.loss_type = loss_type

        self.fixed_effects_nn = fixed_effects_nn
        self.covariance = covariance
        self.module_kwargs = module_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        self.module_ = None
        self.optimizer_ = None

    @cached_property
    def all_model_mat_cols(self) -> Sequence:
        # preserve the ordering where possible
        assert len(self.fixeff_cols) == len(set(self.fixeff_cols)), "Duplicate `fixeff_cols`"
        all_model_mat_cols = list(self.fixeff_cols)
        for gf, gf_cols in self.raneff_design.items():
            assert len(gf_cols) == len(set(gf_cols)), f"Duplicate cols for '{gf}'"
            for col in gf_cols:
                if col in all_model_mat_cols:
                    continue
                all_model_mat_cols.append(col)
        return all_model_mat_cols

    @property
    def grouping_factors(self) -> Sequence[str]:
        return list(self.raneff_design.keys())

    def fit(self, X: pd.DataFrame, y=None, reset: str = 'warn', **kwargs) -> 'MixedEffectsModel':
        """
        Initialize and fit the underlying `MixedEffectsModule` until loss converges.

        :param X: A dataframe with the predictors for fixed and random effects, the group-id column, and the response
        column.
        :param y: Should always be left `None`, as this is pulled from the dataframe.
        :param reset: For multiple calls to fit, should we reset? Defaults to yes with a warning.
        :param kwargs: Further keyword-args to pass to ``partial_fit()``
        :return: This instance, now fitted.
        """
        if y is not None:
            warn(f"Ignoring `y` passed to {type(self).__name__}.fit().")

        # initialize module:
        if reset or not self.module_:
            if reset == 'warn' and self.module_:
                warn("Resetting module.")
            self._initialize_module()

        # build-model-mat:
        X, y, group_ids = self.build_model_mats(X, expect_y=True)

        self.partial_fit(X=X, y=y, group_ids=group_ids, max_num_epochs=None, **kwargs)
        return self

    def partial_fit(self,
                    X: Union[np.ndarray, torch.Tensor],
                    y: Union[np.ndarray, torch.Tensor],
                    group_ids: Sequence,
                    stopping: Union['Stopping', tuple, dict] = ('params', .001),
                    callbacks: Collection[Callable] = (),
                    prog: bool = True,
                    max_num_epochs: Optional[int] = 1) -> 'MixedEffectsModel':
        """
        (Partially) fit the underlying ``MixedEffectsModule``.

        :param X: From ``build_model_mats()``
        :param y: From ``build_model_mats()``
        :param group_ids: From ``build_model_mats()``
        :param callbacks: Functions to call on each epoch. Takes a single argument, the loss-history for this call to
         partial_fit.
        :param max_num_epochs: The maximum number of epochs to fit. For similarity to other sklearn estimator's
         ``partial_fit()`` behavior, this default to 1, so that a single call to ``partial_fit()`` performs a single
         epoch.
        :return: This instance
        """
        if self.module_ is None:
            self.module_ = self._initialize_module()

        if self.optimizer_ is None:
            self.optimizer_ = self._initialize_optimizer()

        X = torch.as_tensor(X, **get_to_kwargs(self.module_))
        y = torch.as_tensor(y, **get_to_kwargs(self.module_))

        if max_num_epochs is None:
            max_num_epochs = float('inf')
        assert max_num_epochs > 0
        epoch = 0

        if isinstance(stopping, dict):
            stopping = Stopping(**stopping, optimizer=self.optimizer_)
        elif isinstance(stopping, (list, tuple)):
            stopping = Stopping(*stopping, optimizer=self.optimizer_)
        else:
            assert isinstance(stopping, Stopping)
            stopping.optimizer = self.optimizer_

        if prog:
            if isinstance(self.optimizer_, torch.optim.LBFGS):
                prog = tqdm(total=self.optimizer_.param_groups[0]['max_eval'])
            else:
                prog = tqdm(total=1)

        def _prog_update(loss):
            if prog:
                prog.update()
                prog.set_description(
                    f"Epoch {epoch:,}; Loss {loss.item():.4}; Convergence {stopping.get_info()}"
                )

        callbacks = list(callbacks)
        fit_loop = self.module_.fit_loop(
            X=X,
            y=y,
            group_ids=group_ids,
            optimizer=self.optimizer_,
            loss_type=self.loss_type,
            callbacks=[_prog_update]
        )

        while True:
            try:
                if prog:
                    prog.reset()
                    prog.set_description(f"Epoch {epoch:,}; Loss -; Convergence {stopping.get_info()}")
                epoch_loss = next(fit_loop)
                for callback in callbacks:
                    callback(epoch_loss)
            except KeyboardInterrupt:
                sleep(.5)
                break
            except RuntimeError as e:
                if 'set_re_cov' in str(e) and 'known' in str(e):
                    raise RuntimeError("Pass known covariance(s) at init: ``MixedEffectsModel(covariance=t)``") from e
                raise e

            if stopping(epoch_loss):
                break

            epoch += 1
            if epoch >= max_num_epochs:
                if max_num_epochs > 1:
                    print("Reached `max_num_epochs`, stopping.")
                break

        return self

    @torch.no_grad()
    def _check_convergence(self, old: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        new = torch.cat([param.view(-1) for param in self.module_.parameters()]).clone()
        if old is None:
            changes = None
        else:
            changes = (new - old).abs() / old.abs()
        return changes, new

    def build_model_mats(self,
                         df: pd.DataFrame,
                         expect_y: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray]:
        X = torch.as_tensor(df[self.all_model_mat_cols].values, **get_to_kwargs(self.module_))

        group_ids = df[self.grouping_factors].values

        y = None  # ok to omit in `predict`
        if self.response in df.columns:
            y = torch.as_tensor(df[self.response].values, **get_to_kwargs(self.module_))
        elif expect_y:
            raise ValueError(f"missing column '{self.response}'")
        return X, y, group_ids

    def _initialize_module(self) -> MixedEffectsModule:
        if self.fixed_effects_nn is not None:
            pass  # TODO: do we need to deep-copy?
        if isinstance(self.covariance, str):
            covariance_arg = self.covariance
        else:
            covariance_arg = 'log_cholesky'
            if len(self.grouping_factors) == 1 and not isinstance(self.covariance, dict):
                self.covariance = {self.grouping_factors[0]: self.covariance}
            from torch_hlm.covariance import Covariance
            covariance_arg = {
                k: Covariance.from_name('log_cholesky', rank=len(v)).set(torch.as_tensor(v))
                for k, v in self.covariance.items()
            }
            # self.covariance = {k: torch.as_tensor(v) for k, v in self.covariance.items()}

        module_kwargs = {
            'fixeff_features': [],
            'raneff_features': {gf: [] for gf in self.grouping_factors},
            'fixed_effects_nn': self.fixed_effects_nn,
            'covariance': covariance_arg
        }
        if self.module_kwargs:
            module_kwargs.update(self.module_kwargs)

        # organize model-mat indices. this works b/c it is sync'd with ``build_model_mats``. might be a better way to
        # make this dependency clearer...
        for i, col in enumerate(self.all_model_mat_cols):
            for gf, gf_cols in self.raneff_design.items():
                if col in gf_cols:
                    module_kwargs['raneff_features'][gf].append(i)
            if col in self.fixeff_cols:
                module_kwargs['fixeff_features'].append(i)

        # initialize module:
        self.module_ = MixedEffectsModule.from_name(self.response_type, **module_kwargs)

        if isinstance(self.covariance, dict):
            # freeze cov:
            for gf, gf_cov in self.covariance.items():
                for p in self.module_.covariance[gf].parameters():
                    p.requires_grad_(False)
        else:
            # sensible defaults, don't freeze
            for gf, idx in self.module_.rf_idx.items():
                std_devs = torch.ones(len(idx) + 1)
                if len(idx):
                    std_devs[1:] *= (.5 / np.sqrt(len(idx)))
                self.module_.covariance[gf].set(torch.diag_embed(std_devs ** 2))

        return self.module_

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        optimizer_kwargs = {'lr': .001}
        if issubclass(self.optimizer_cls, torch.optim.LBFGS):
            optimizer_kwargs.update(lr=.10, max_eval=12)
        optimizer_kwargs.update(self.optimizer_kwargs or {})
        optimizer_kwargs['params'] = [p for p in self.module_.parameters() if p.requires_grad]
        self.optimizer_ = self.optimizer_cls(**optimizer_kwargs)
        return self.optimizer_

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
                f"`group_data` must include the response ('{self.response}'), so that random-effects can "
                f"be computed."
            )
        pred = self.module_(X, group_ids=group_ids, re_solve_data=(X_t, y_t, group_ids_t))
        return pred.numpy()


class Stopping:
    def __init__(self,
                 type: str = 'params',
                 abstol: Optional[float] = .001,
                 reltol: Optional[float] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None):

        self.type = 'params' if type.startswith('param') else 'loss'
        self.optimizer = optimizer
        self.abstol = abstol
        self.reltol = reltol
        self.old_value = None
        self._info = (float('nan'), min(self.abstol or float('inf'), self.reltol or float('inf')))

    @torch.no_grad()
    def get_info(self, fmt: str = "{:.4}/{:}") -> str:
        return fmt.format(*self._info)

    @torch.no_grad()
    def get_new_value(self, loss: Optional[float]):
        if self.type == 'params':
            assert self.optimizer is not None
            flat_params = []
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    flat_params.append(p.view(-1))
            return torch.cat(flat_params)
        else:
            return torch.as_tensor([loss])

    @torch.no_grad()
    def __call__(self, loss: Optional[float]) -> bool:
        new_value = self.get_new_value(loss)
        if self.old_value is None:
            self.old_value = new_value
            return False
        old_value = self.old_value
        self.old_value = new_value
        abs_change = (new_value - old_value).abs()
        assert not (self.abstol is None and self.reltol is None)
        if self.abstol is not None and (abs_change > self.abstol).any():
            self._info = (abs_change.max(), self.abstol)
            return False
        if self.reltol is None:
            self._info = (abs_change.max(), self.abstol)  # even if we've converged, print up to date info
        else:
            rel_change = abs_change / old_value.abs()
            self._info = (rel_change.max(), self.reltol)
            if (rel_change > self.reltol).any():
                return False
        return True
