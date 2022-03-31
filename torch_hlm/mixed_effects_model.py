import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from torch_hlm.utils import FitFailedException

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property
from time import sleep
from typing import Sequence, Optional, Type, Collection, Callable, Tuple, Dict, Union
from warnings import warn

import torch

from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torch_hlm.mixed_effects_module import MixedEffectsModule
from torch_hlm.mixed_effects_module.utils import get_to_kwargs, pad_res_per_gf

N_FIT_RETRIES = int(os.getenv('HLM_N_FIT_RETRIES', 10))


class MixedEffectsModel(BaseEstimator):
    """
    :param fixeffs: Sequence of column-names of the fixed-effects in the model-matrix ``X``. Instead of string-
     columns-names, can also supply callables which take the input pandas ``DataFrame`` and return a list of
     column-names.
    :param raneff_design: A dictionary, whose key(s) are the names of grouping factors and whose values are
     column-names for random-effects of that grouping factor. As with ``fixeffs``, either string-column-names or
     callables can be used.
    :param response_type: Currently supports 'binomial' or 'gaussian'.
    :param loss_type: How loss is calculated. The default behavior and available options vary depending on
     ``response_type``. To see the default, call ``model.module_._get_default_loss_type()`` after fit.
    :param fixed_effects_nn: A ``torch.nn.Module`` that takes in the fixed-effects model-matrix and outputs predictions.
     Default is to use a single-layer Linear module.
    :param covariance: Can pass string with parameterization (see ``torch_hlm.covariance``, default is 'log_cholesky').
     Alternatively, can pass a dictionary with keys as grouping factors and values as tensors. These will be used as
     the covariances for those grouping factors' random-effects, which will *not* be optimized (i.e. their
     ``requires_grad`` flag will be set to False and they will not be passed to the optimizer). This can be used
     in settings where optimizing the random-effects covariance is not supported.
    :param module_kwargs: Other keyword-arguments for the ``MixedEffectsModule``.
    :param optimizer_cls: A ``torch.optim.Optimizer``, defaults to LBFGS.
    :param optimizer_kwargs: Keyword-arguments for the optimizer.
    """

    def __init__(self,
                 fixeffs: Sequence[Union[str, Callable]],
                 raneff_design: Dict[str, Sequence[Union[str, Callable]]],
                 response_type: str = 'gaussian',
                 loss_type: Optional[str] = None,
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 covariance: Union[str, Dict[str, torch.Tensor]] = 'log_cholesky',
                 module_kwargs: Optional[dict] = None,
                 optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.LBFGS,
                 optimizer_kwargs: Optional[dict] = None):

        self.fixeffs = fixeffs
        self.fixeffs_ = None
        self.raneff_design = raneff_design
        self.raneff_design_ = None
        self.response_type = response_type
        self.loss_type = loss_type

        self.fixed_effects_nn = fixed_effects_nn
        self.covariance = covariance
        self.module_kwargs = module_kwargs
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        self.module_ = None
        self.optimizer_ = None
        self._fit_failed = 0

    @cached_property
    def all_model_mat_cols(self) -> Sequence:
        # preserve the ordering where possible
        assert len(self.fixeffs_) == len(set(self.fixeffs_)), "Duplicate `fixeffs`"
        all_model_mat_cols = list(self.fixeffs_)
        for gf, gf_cols in self.raneff_design_.items():
            assert len(gf_cols) == len(set(gf_cols)), f"Duplicate cols for '{gf}'"
            for col in gf_cols:
                if col in all_model_mat_cols:
                    continue
                all_model_mat_cols.append(col)
        return all_model_mat_cols

    @property
    def grouping_factors(self) -> Sequence[str]:
        return list(self.raneff_design_.keys())

    @retry(retry=retry_if_exception_type(FitFailedException), reraise=True, stop=stop_after_attempt(N_FIT_RETRIES + 1))
    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: np.ndarray,
            reset: str = 'warn',
            verbose: bool = True,
            **kwargs) -> 'MixedEffectsModel':
        """
        Initialize and fit the underlying `MixedEffectsModule` until loss converges.

        :param X: A array/dataframe with the predictors for fixed and random effects and the group-id column(s).
        :param y: An array/dataframe with the response.
        :param reset: For multiple calls to fit, should we reset? Defaults to yes with a warning.
        :param verbose:
        :param kwargs: Further keyword-args to pass to ``partial_fit()``
        :return: This instance, now fitted.
        """
        # tallying fit-failurs:
        if self._fit_failed:
            if verbose:
                print(f"Retry attempt {self._fit_failed}/{N_FIT_RETRIES}")
            reset = True

        if kwargs.get('sample_weight') is not None:
            raise ValueError(
                f"Do not pass `sample_weight` to {type(self).__name__}.fit(); instead, add to last dim of ``y``."
            )

        # fixeffs and raneff_design can be callables:
        self.fixeffs_, self.raneff_design_ = self._standardize_cols(X, self.fixeffs, self.raneff_design, verbose)

        # initialize module:
        if reset or not self.module_:
            if reset == 'warn' and self.module_:
                warn("Resetting module.")
            self.module_ = self._initialize_module()
            self.optimizer_ = None

        # build-model-mat:
        X, group_ids, y, sample_weight = self.build_model_mats(X, y)

        kwargs['prog'] = kwargs.get('prog', verbose)
        kwargs['max_num_epochs'] = kwargs.get('max_num_epochs',None)

        try:
            self.partial_fit(
                X=X,
                y=y,
                group_ids=group_ids,
                sample_weight=sample_weight,
                **kwargs
            )
        except FitFailedException as e:
            self._fit_failed += 1
            raise e

        self._fit_failed = 0
        return self

    @classmethod
    def _standardize_cols(cls,
                          X: pd.DataFrame,
                          fixeffs: Sequence[Union[str, Callable]],
                          raneff_design: Dict[str, Sequence[Union[str, Callable]]],
                          verbose: bool = True
                          ) -> Tuple[Sequence[str], Dict[str, Sequence[str]]]:
        any_callables = False
        standarized_cols = {}
        assert 'fixeffs' not in raneff_design
        for cat, to_standardize in {**raneff_design, **{'fixeffs': fixeffs}}.items():
            if cat not in standarized_cols:
                standarized_cols[cat] = []
            for i,cols in enumerate(to_standardize):
                if callable(cols):
                    any_callables = True
                    cols = cols(X)
                    if not cols:
                        warn(f"Element {i} of {cat} was a callable that, when given X, returned no rows.")
                if isinstance(cols, str):
                    if cols not in X.columns:
                        raise RuntimeError(f"No column named {cols}")
                    cols = [cols]
                standarized_cols[cat].extend(cols)
        if any_callables and verbose:
            print(f"Model-features: {standarized_cols}")
        fixeffs = standarized_cols.pop('fixeffs')
        return fixeffs, standarized_cols

    def partial_fit(self,
                    X: Union[np.ndarray, torch.Tensor],
                    y: Union[np.ndarray, torch.Tensor],
                    group_ids: Sequence,
                    sample_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    stopping: Union['Stopping', tuple, dict] = (.001,),
                    callbacks: Collection[Callable] = (),
                    prog: bool = True,
                    max_num_epochs: Optional[int] = 1,
                    min_num_epochs: int = 0,
                    clip_grad: Optional[float] = 2.0,
                    **kwargs) -> 'MixedEffectsModel':
        """
        (Partially) fit the underlying ``MixedEffectsModule``.

        :param X: From ``build_model_mats()``
        :param y: From ``build_model_mats()``
        :param group_ids: From ``build_model_mats()``
        :param sample_weight: TODO
        :param stopping: args/kwargs to pass to :class:`torch_hlm.mixed_effects_model.Stopping` (e.g. ``(.01,)`` would
         use abstol of .01).
        :param callbacks: Functions to call on each epoch. Takes a single argument, the loss-history for this call to
         partial_fit.
        :param prog: TODO
        :param max_num_epochs: The maximum number of epochs to fit. For similarity to other sklearn estimator's
         ``partial_fit()`` behavior, this default to 1, so that a single call to ``partial_fit()`` performs a single
         epoch.
        :param min_num_epochs: TODO
        :param clip_grad: TODO
        :param kwargs: Further keyword arguments passed to ``MixedEffectsModule.fit_loop()``
        :return: This instance
        """
        if self.module_ is None:
            self.module_ = self._initialize_module()

        if self.optimizer_ is None:
            self.optimizer_ = self._initialize_optimizer()

        X = torch.as_tensor(X, **get_to_kwargs(self.module_))
        y = torch.as_tensor(y, **get_to_kwargs(self.module_))
        if sample_weight is None:
            sample_weight = torch.ones_like(y)
        sample_weight = torch.as_tensor(sample_weight)
        assert y.shape == sample_weight.shape

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

        _inner_callbacks = []
        if clip_grad:
            _inner_callbacks.append(lambda x: torch.nn.utils.clip_grad_value_(self.module_.parameters(), clip_grad))

        if prog:
            if isinstance(self.optimizer_, torch.optim.LBFGS):
                prog = tqdm(total=self.optimizer_.param_groups[0]['max_eval'])
            else:
                prog = tqdm(total=1)

            def _prog_update(loss):
                prog.update()
                prog.set_description(
                    f"Epoch {epoch:,}; Loss {loss.item():.4}; Convergence {stopping.get_info()}"
                )

            _inner_callbacks.append(_prog_update)

        fit_loop = self.module_.fit_loop(
            X=X,
            y=y,
            group_ids=group_ids,
            weights=sample_weight,
            optimizer=self.optimizer_,
            loss_type=self.loss_type,
            callbacks=_inner_callbacks,
            **kwargs
        )

        callbacks = list(callbacks)
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
            except ValueError as e:
                msg = str(e)
                if ('parameter' in msg or 'covariance' in msg) and ('nan' in msg or 'inf' in msg) or ('posdef' in msg):
                    raise FitFailedException(str(e))
                else:
                    raise e

            if stopping(epoch_loss):
                break

            epoch += 1
            if epoch >= max_num_epochs:
                if max_num_epochs > 1:
                    warn(f"Reached `max_num_epochs={max_num_epochs}`, stopping.")
                break

        if epoch < min_num_epochs:
            raise FitFailedException(f"Terminated on epoch={epoch}, but min_num_epochs={min_num_epochs}")

        return self

    def build_model_mats(self,
                         X: Union[pd.DataFrame, np.ndarray],
                         y: Optional[np.ndarray] = None
                         ) -> Tuple[torch.Tensor, np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not isinstance(X, pd.DataFrame):
            raise NotImplementedError("Non-df ``X`` not yet implemented")
        Xmm = torch.as_tensor(X[self.all_model_mat_cols].values, **get_to_kwargs(self.module_))

        group_ids = X[self.grouping_factors].values

        sample_weight = None
        if y is not None:
            if not isinstance(y, torch.Tensor):
                y = np.asanyarray(y)
            y = torch.as_tensor(y, **get_to_kwargs(self.module_))
            if len(y.shape) == self.module_.resp_ndim + 1:
                if y.shape[-1] == 2:
                    sample_weight = y[..., 1]
                    y = y[..., 0]
                elif y.shape[-1] == 1:
                    y = y.squeeze(-1)
                else:
                    raise ValueError(
                        f"If ``y`` has ``y.ndim == {self.module_.resp_ndim + 1}``, then expect last "
                        f"dim to be of extend 2 (1st for response, 2nd for sample-weight). Instead got {y.shape[-1]}."
                    )
            elif len(y.shape) != self.module_.resp_ndim:
                raise ValueError(
                    f"Expected y.ndim to be {self.module_.resp_ndim} (or +1 for weight). Got {len(y.shape)}."
                )

        return Xmm, group_ids, y, sample_weight

    def _initialize_module(self) -> MixedEffectsModule:
        if self.fixed_effects_nn is not None:
            pass  # TODO: do we need to deep-copy?
        if isinstance(self.covariance, str):
            covariance_arg = self.covariance
        else:
            if len(self.grouping_factors) == 1 and not isinstance(self.covariance, dict):
                self.covariance = {self.grouping_factors[0]: self.covariance}
            from torch_hlm.covariance import Covariance
            covariance_arg = {
                k: Covariance.from_name('log_cholesky', rank=len(v)).set(torch.as_tensor(v))
                for k, v in self.covariance.items()
            }

        module_kwargs = {
            'fixeff_features': [],
            'raneff_features': {gf: [] for gf in self.grouping_factors},
            'fixed_effects_nn': self.fixed_effects_nn,
            'covariance': covariance_arg
        }
        if self.module_kwargs:
            module_kwargs.update(self.module_kwargs)

        # organize model-mat indices. this works b/c it is sync'd with ``build_model_mats``, via ``all_model_mat_cols``.
        # might be a better way to make this dependency clearer...
        for i, col in enumerate(self.all_model_mat_cols):
            for gf, gf_cols in self.raneff_design_.items():
                if col in gf_cols:
                    module_kwargs['raneff_features'][gf].append(i)
            if col in self.fixeffs_:
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
            optimizer_kwargs.update(lr=.25, max_eval=12)
            loss_type = self.loss_type or self.module_._get_default_loss_type()
            if loss_type != 'iid':
                optimizer_kwargs['line_search_fn'] = 'strong_wolfe'
        optimizer_kwargs.update(self.optimizer_kwargs or {})
        optimizer_kwargs['params'] = [p for p in self.module_.parameters() if p.requires_grad]
        self.optimizer_ = self.optimizer_cls(**optimizer_kwargs)
        return self.optimizer_

    @torch.no_grad()
    def predict(self,
                X: pd.DataFrame,
                group_data: Optional[Tuple[pd.DataFrame, np.ndarray]],
                type: str = 'mean',
                unknown_groups: str = 'zero',
                **kwargs) -> np.ndarray:
        """
        :param X: A dataframe with the predictors for fixed and random effects, as well as the group-id column.
        :param group_data: Historical data -- an (X,y) tuple -- that will be used to obtain RE-estimates for each
         group, which will then be used to generate predictions for X. Can pass ``None`` if there's no historical data.
        :param type: What type of prediction? This can be any attribute of the distribution being predicted. For
         example, for a binomial model, we will generate a ``torch.distributions.Binomial`` object, and the
         prediction output from this method will be an attribute of that (e.g. type='logits' would get the predicted
         logits). Default 'mean'.
        :param unknown_groups: How should groups that are in ``group_ids`` but not ``group_data`` be handled?
         Default is to "zero"-out the random-effects, which means only fixed-effects will be used. Other options are
         "nan" (return `nan`s for these) or "raise" (raise an exception). Can prefix the first two options with "quiet"
         to suppress printing the number of unknown groups.
        :return: An ndarray of predictions
        """
        if 'verbose' not in kwargs:
            kwargs['verbose'] = 'prog'
        X, group_ids, *_ = self.build_model_mats(X)

        # solve random-effects:
        if group_data is not None:
            if not isinstance(group_data, (tuple, list)):
                raise TypeError("Expected ``group_data`` to be a X,y tuple.")
            re_solve_data = self.build_model_mats(*group_data)
            _, rs_group_ids, *_ = re_solve_data
        else:
            rs_group_ids = np.empty((0, len(self.grouping_factors)), dtype='int')

        if len(rs_group_ids):
            res_per_gf = self.module_.get_res(*re_solve_data, **kwargs)
        else:
            # i.e. re_solve_data is empty
            res_per_gf = {gf: torch.zeros((0, len(idx) + 1)) for gf, idx in self.module_.rf_idx.items()}

        res_per_gf_padded = pad_res_per_gf(
            res_per_gf, group_ids, rs_group_ids, fill_value=float('nan'), verbose=not unknown_groups.startswith('quiet')
        )
        unknown_groups = unknown_groups.replace('quiet', '').lstrip('_')
        assert unknown_groups in {'zero', 'nan', 'raise'}
        if unknown_groups == 'raise':
            for res in res_per_gf_padded.values():
                if torch.isnan(res).any():
                    raise RuntimeError("There are groups in ``group_ids`` that are not in ``group_data``")
        elif unknown_groups == 'zero':
            for res in res_per_gf_padded.values():
                res[torch.isnan(res)] = 0.

        # predict:
        dist = self.module_.predict_distribution_mode(X, group_ids=group_ids, res_per_gf=res_per_gf_padded)
        pred = getattr(dist, type)
        return pred.numpy()


class Stopping:
    def __init__(self,
                 abstol: Optional[float] = .001,
                 reltol: Optional[float] = None,
                 patience: int = 2,
                 type: str = 'params_and_loss',
                 optimizer: Optional[torch.optim.Optimizer] = None):

        self.type = type
        self.optimizer = optimizer
        self.abstol = abstol
        self.reltol = reltol
        self.old_value = None
        self.patience = patience
        self._patience_counter = 0
        self._info = (float('nan'), min(self.abstol or float('inf'), self.reltol or float('inf')))

    @torch.no_grad()
    def get_info(self, fmt: str = "{:.4}/{:}") -> str:
        return fmt.format(*self._info)

    @torch.no_grad()
    def get_new_value(self, loss: Optional[float]):
        flat_params = []
        if 'params' in self.type:
            assert self.optimizer is not None
            for g in self.optimizer.param_groups:
                for p in g['params']:
                    flat_params.append(p.view(-1))
        if 'loss' in self.type:
            flat_params.append(torch.as_tensor([loss]))
        return torch.cat(flat_params)

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
            self._patience_counter = 0
            return False
        if self.reltol is None:
            self._info = (abs_change.max(), self.abstol)  # even if we've converged, print up to date info
        else:
            rel_change = abs_change / old_value.abs()
            self._info = (rel_change.max(), self.reltol)
            if (rel_change > self.reltol).any():
                self._patience_counter = 0
                return False
        self._patience_counter += 1
        return self._patience_counter >= self.patience
