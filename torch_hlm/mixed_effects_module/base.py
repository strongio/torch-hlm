from typing import Union, Optional, Sequence, Iterator

import torch
import numpy as np
from tqdm.auto import tqdm

from .utils import log_chol_to_chol, ndarray_to_tensor, validate_1d_like


class _Evaluator:
    """
    Creates a `closure` for `torch.nn.Optimizer.step()` which clears the gradient, re-evaluates the model, and returns
    the loss
    """

    def __init__(self,
                 module: 'MixedEffectsModule',
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: Sequence,
                 optimizer: torch.optim.Optimizer,
                 **kwargs):
        self.module = module
        self.X = X
        self.y = y
        self.group_ids = group_ids
        self.optimizer = optimizer
        self.group_data = module.make_group_data(X, y, group_ids)
        self.static_re_solve_kwargs = module._static_re_solve_kwargs(self.group_data, **kwargs)
        self.progress = None
        if isinstance(optimizer, torch.optim.LBFGS):
            self.progress = tqdm(total=optimizer.param_groups[0]['max_eval'] * 2)
        else:
            self.progress = tqdm()
        self.epoch = None

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.progress:
            self.progress.reset()
            self.progress.set_description(f"Epoch {epoch:,}")

    def _get_res_per_group(self, **kwargs) -> torch.Tensor:
        re_solve_kwargs = self.module._re_solve_kwargs(
            self.group_data, static_kwargs=self.static_re_solve_kwargs
        )
        re_solve_kwargs.update(kwargs)
        return self.module._re_solve(**re_solve_kwargs)

    def __call__(self) -> torch.Tensor:
        self.optimizer.zero_grad()
        res_per_group_mat = self._get_res_per_group()
        pred = self.module(X=self.X, group_ids=self.group_ids, res_per_group=res_per_group_mat)
        loss = self.module.get_loss(pred, self.y, res_per_group=res_per_group_mat)
        loss.backward()
        if self.progress:
            self.progress.update()
            self.progress.set_description(f"Epoch {self.epoch:,}; Loss {loss.item():.4}")
        return loss


class MixedEffectsModule(torch.nn.Module):
    """
    Base-class for a torch.nn.Module that can be trained with mixed-effects
    """
    evaluator_cls = _Evaluator

    @classmethod
    def from_name(cls, nm: str, *args, **kwargs):
        if nm.lower() == 'gaussian':
            from torch_hlm.mixed_effects_module import GaussianMixedEffectsModule
            return GaussianMixedEffectsModule(*args, **kwargs)
        elif nm.lower() in {'binary', 'logistic'}:
            from torch_hlm.mixed_effects_module import LogisticMixedEffectsModule
            return LogisticMixedEffectsModule(*args, **kwargs)
        else:
            raise ValueError(f"Unrecognized '{nm}'; currently supported are logistic/binary and gaussian")

    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Optional[Union[int, Sequence]],
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 verbose: int = 1):
        """
        :param fixeff_features: Count, or column-indices, of features for fixed-effects model matrix (not including
        bias/intercept)
        :param raneff_features: Count, or column-indices, of features for random-effects model matrix (not including
        bias/intercept)
        :param fixed_effects_nn: An optional torch.nn.Module for the fixed-effects. The default is to create a
        `torch.nn.Linear(num_fixed_features, 1)`.
        :param verbose: Verbosity level; 1 (default) allows certain messages, 0 disables all messages, 2 (TODO)
        """
        super().__init__()
        self.verbose = verbose

        # fixed effects:
        if isinstance(fixeff_features, int):
            num_fixed_features = fixeff_features
            self.ff_idx = list(range(fixeff_features))
        else:
            num_fixed_features = len(fixeff_features)
            self.ff_idx = list(fixeff_features)
        if fixed_effects_nn is None:
            fixed_effects_nn = torch.nn.Linear(num_fixed_features, 1)
            fixed_effects_nn.weight.data[:] = .01 * torch.randn(num_fixed_features)
        self.fixed_effects_nn = fixed_effects_nn

        # random effects:
        if isinstance(raneff_features, int):
            self.rf_idx = list(range(raneff_features))
        else:
            self.rf_idx = list(raneff_features)

        # rfx covariance:
        self._re_cov_params = self._init_re_cov_params()

    # covariance -------
    # child-classes could override these 3 methods
    def re_distribution(self) -> torch.distributions.MultivariateNormal:
        L = log_chol_to_chol(
            log_diag=self._re_cov_params['cholesky_log_diag'], off_diag=self._re_cov_params['cholesky_off_diag']
        )
        return torch.distributions.MultivariateNormal(loc=torch.zeros(len(L)), scale_tril=L)

    def _init_re_cov_params(self) -> torch.nn.ParameterDict:
        _rank = len(self.rf_idx) + 1
        params = torch.nn.ParameterDict()
        _num_upper_tri = int(_rank * (_rank - 1) / 2)
        params['cholesky_log_diag'] = torch.nn.Parameter(data=.01 * torch.randn(_rank))
        params['cholesky_off_diag'] = torch.nn.Parameter(data=.01 * torch.randn(_num_upper_tri))
        return params

    def set_re_cov(self, cov: torch.Tensor):
        """
        :param cov: The covariance matrix of the random-effects.
        """
        with torch.no_grad():
            L = torch.cholesky(cov)
            self._re_cov_params['cholesky_log_diag'].data[:] = L.diag().log()
            self._re_cov_params['cholesky_off_diag'].data[:] = torch.tril(L).view(-1)
            assert torch.isclose(self.re_distribution().covariance_matrix, cov).all()

    def _is_cov_param(self, param: torch.Tensor) -> bool:
        return any(param is cov_param for cov_param in self._re_cov_params.values())

    # forward / re-solve -------
    def forward(self,
                X: Union[torch.Tensor, np.ndarray],
                group_ids: Sequence,
                group_data: Optional[dict] = None,
                res_per_group: Optional[Union[dict, torch.Tensor]] = None) -> torch.Tensor:
        """

        :param X: A 2D tensor or ndarray of predictors
        :param group_ids: A sequence which assigns each row in X to a group
        :param group_data: A dictionary whose keys are group-ids (corresponding to `group_ids`) and whose values are
        historical Xs and ys that will be used to solve the random effects for each group.
        :param res_per_group: Instead of passing `group_data` to and having the module solver the random-effect per
        group, can alternatively pass a dictionary whose keys are group-ids and whose values are 1d tensors with the
        random-effects per group.
        :return:
        """
        X = ndarray_to_tensor(X)
        if torch.isnan(X).any():
            raise ValueError("`nans` in `X`")

        if res_per_group is None:
            if group_data is None:
                raise TypeError("Must pass one of `group_data`, `betas_per_group`; got neither.")
            # _re_solve returns in sorted order;
            # so need to be sure unique ids in `group_data` are same as those in `group_ids`
            group_data = {g: group_data[g] for g in np.unique(group_ids)}
            res_per_group_mat = self._re_solve(**self._re_solve_kwargs(group_data))
            return self.forward(X=X, group_ids=group_ids, group_data=None, res_per_group=res_per_group_mat)

        if group_data is not None:
            raise TypeError("Must pass one of `group_data`, `betas_per_group`; got both.")

        # organize groups, getting an indexer that maps group[i] to rows in the original data
        ugroups, group_idx = np.unique(group_ids, return_inverse=True)

        # get yhat for raneffects w/broadcasting
        if isinstance(res_per_group, dict):
            res_per_group_mat = torch.zeros(len(ugroups), len(self.rf_idx) + 1)
            for i, group_id in enumerate(ugroups):
                res_per_group_mat[i] = ndarray_to_tensor(res_per_group[group_id])
        else:
            res_per_group_mat = ndarray_to_tensor(res_per_group)
        betas_broad = res_per_group_mat[group_idx]
        yhat_r = (self._get_Xr(X) * betas_broad).sum(1)

        # yhat for fixef:
        yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))

        # combine:
        return yhat_f + yhat_r

    def _get_Xr(self, X: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.ones((len(X), 1)), X[:, self.rf_idx]], 1)

    def get_res_per_group(self, group_data: dict) -> dict:
        with torch.no_grad():
            re_mat = self._re_solve(**self._re_solve_kwargs(group_data))
            out = {}
            for i, group_id in enumerate(sorted(group_data)):
                out[group_id] = re_mat[i]
        return out

    def _re_solve_kwargs(self, group_data: dict, static_kwargs: Optional[dict] = None) -> dict:
        """
        kwargs to re_solve that must be computed each time `forward()` is called
        """
        if static_kwargs is None:
            out = self._static_re_solve_kwargs(group_data)
        else:
            out = static_kwargs.copy()
        Xall = out.pop('Xall')
        out['offset'] = validate_1d_like(self.fixed_effects_nn(Xall[:, self.ff_idx]))
        out['offset'] = out['offset'].detach()  # TODO: make optional?
        out['prior_precision'] = self.re_distribution().precision_matrix
        return out

    def _static_re_solve_kwargs(self, group_data: dict, **kwargs) -> dict:
        """
        kwargs to re_solve that can be precomputed once, then re-used for multiple `forward()` calls
        """
        assert not len(kwargs), (
            f"`{type(self).__name__}` doesn't take additional kwargs; subclass should have removed: `{set(kwargs)}`"
        )
        out = {
            'Xall': [],
            'X': [],
            'XtX': [],
            'y': [],
            'group_ids': []
        }
        for group_id in sorted(group_data.keys()):
            Xg, yg = group_data[group_id]
            if isinstance(Xg, np.ndarray):
                Xg = torch.from_numpy(Xg.astype('float32'))
            if isinstance(yg, np.ndarray):
                yg = torch.from_numpy(yg.astype('float32'))
            Xrg = self._get_Xr(Xg)
            out['X'].append(Xrg)
            out['XtX'].append(Xrg.t() @ Xrg)
            out['Xall'].append(Xg)
            out['y'].append(yg)
            out['group_ids'].extend([group_id] * len(Xg))
        out['Xall'] = torch.cat(out['Xall'], 0)
        out['XtX'] = torch.stack(out['XtX'])
        out['X'] = torch.cat(out['X'], 0)
        out['y'] = torch.cat(out['y'])
        return out

    @staticmethod
    def _re_solve(X: torch.Tensor,
                  y: torch.Tensor,
                  group_ids: Sequence,
                  offset: torch.Tensor,
                  prior_precision: torch.Tensor,
                  **kwargs):
        """
        Given a batch of X,ys for groups, solve the random-effects.
        :param X: The N*K random-effects model matrix, including a dummy column of 1s for the intercept.
        :param y: The length-N target vector.
        :param group_ids: A length-N sequence that assigns each row of X to a group.
        :param offset: A length-N offset to apply to the predictions before solving.
        :param prior_precision: A K*K precision ("penalty") matrix
        :param kwargs: Keyword-arguments used by subclasses
        :return: A G*K matrix of solved random-effects, with each row corresponding to the (sorted) `group_ids`
        """
        raise NotImplementedError

    # fitting --------
    @staticmethod
    def make_group_data(X: Union[torch.Tensor, np.ndarray],
                        y: Union[torch.Tensor, np.ndarray],
                        group_ids: Sequence) -> dict:
        X = ndarray_to_tensor(X)
        y = ndarray_to_tensor(y)
        group_ids = np.asanyarray(group_ids)

        # torch.split requires we put groups into contiguous chunks:
        sort_idx = np.argsort(group_ids)
        group_ids = group_ids[sort_idx]
        X = X[sort_idx]
        y = y[sort_idx]
        # much faster approach to chunking than something like `[X[gid==group_ids] for gid in np.unique(group_ids)]`:
        unique_groups, counts_per_group = np.unique(group_ids, return_counts=True)
        counts_per_group = counts_per_group.tolist()
        group_data = {}
        for group_id, Xg, yg in zip(unique_groups, torch.split(X, counts_per_group), torch.split(y, counts_per_group)):
            group_data[group_id] = Xg, yg
        return group_data

    def fit_loop(self,
                 X: Union[torch.Tensor, np.ndarray],
                 y: Union[torch.Tensor, np.ndarray],
                 group_ids: Sequence,
                 optimizer: torch.optim.Optimizer,
                 re_solve_kwargs: Optional[dict] = None,
                 **kwargs) -> Iterator[float]:
        X = ndarray_to_tensor(X)
        y = ndarray_to_tensor(y)

        closure = self.evaluator_cls(module=self, X=X, y=y, group_ids=group_ids, optimizer=optimizer, **kwargs)
        closure.static_re_solve_kwargs.update(re_solve_kwargs or {})
        if not self.verbose:
            closure.progress = None

        epoch = 0
        while True:
            try:
                closure.set_epoch(epoch)
                loss = optimizer.step(closure).item()
                yield loss
                epoch += 1
            except KeyboardInterrupt:
                break

    def get_loss(self, pred: torch.Tensor, actual: torch.Tensor, res_per_group: torch.Tensor):
        raise NotImplementedError
