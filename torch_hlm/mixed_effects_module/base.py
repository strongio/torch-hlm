from collections import OrderedDict
from typing import Union, Optional, Sequence, Iterator, Dict, Tuple, Type
from warnings import warn

import torch
import numpy as np

from tqdm.auto import tqdm

from .utils import log_chol_to_chol, ndarray_to_tensor, validate_1d_like, chunk_grouped_data


class MixedEffectsModule(torch.nn.Module):
    """
    Base-class for a torch.nn.Module that can be trained with mixed-effects
    """
    solver_cls: Type['ReSolver'] = None
    likelihood_requires_raneffs = False

    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Dict[str, Union[int, Sequence]],
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 covariance_parameterization: str = 'log_cholesky',
                 verbose: int = 1):
        """
        :param fixeff_features: Thee column-indices of features for fixed-effects model matrix (not including
        bias/intercept); if a single integer is passed then these indices are `range(fixeff_features)`
        :param raneff_features: If there is a single grouping factor, then this argument is like `fixeff_features`: the
        indices in the mode-matrix corresponding to random-effects. If there are multiple grouping factors, then the
        keys are grouping factor labels and the values are these indices.
        :param fixed_effects_nn: An optional torch.nn.Module for the fixed-effects. The default is to create a
        `torch.nn.Linear(num_fixed_features, 1)`.
        :param verbose: Verbosity level; 1 (default) prints the loss on each optimization epoch; 0 disables everything,
        and 2 allows messages from the inner random-effects solver.
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
        else:
            for p in fixed_effects_nn.parameters():
                if p.dtype != torch.float32:
                    warn("`fixed_effects_nn` doesn't use torch.float32")
        self.fixed_effects_nn = fixed_effects_nn

        # random effects:
        if not isinstance(raneff_features, dict):
            raneff_features = {'group': raneff_features}
        self.rf_idx = OrderedDict()
        for grouping_factor, rf in raneff_features.items():
            if isinstance(rf, int):
                self.rf_idx[grouping_factor] = list(range(rf))
            else:
                self.rf_idx[grouping_factor] = list(rf)

        # rfx covariance:
        self.covariance_parameterization = covariance_parameterization.lower()
        self._re_cov_params = self._init_re_cov_params()

    @property
    def residual_var(self) -> Union[torch.Tensor, float]:
        return 1.0

    @property
    def grouping_factors(self) -> Sequence:
        return list(self.rf_idx.keys())

    # covariance -------
    # child-classes could override the first 3 methods for different parameterizations
    def _init_re_cov_params(self) -> torch.nn.ParameterDict:
        assert self.covariance_parameterization in {'low_rank', 'log_cholesky'}
        params = torch.nn.ParameterDict()
        for gf, idx in self.rf_idx.items():
            _rank = len(idx) + 1
            if self.covariance_parameterization == 'low_rank':
                _low_rank = int(np.sqrt(_rank))  # TODO
                params[f'{gf}_lr'] = torch.nn.Parameter(data=.1 * torch.randn(_rank, _low_rank))
                params[f'{gf}_log_std_devs'] = torch.nn.Parameter(data=.1 * torch.randn(_rank))
            else:
                _num_upper_tri = int(_rank * (_rank - 1) / 2)
                params[f'{gf}_cholesky_log_diag'] = torch.nn.Parameter(data=.1 * torch.randn(_rank))
                params[f'{gf}_cholesky_off_diag'] = torch.nn.Parameter(data=.1 * torch.randn(_num_upper_tri))
        return params

    def re_distribution(self, grouping_factor: str = None, eps: float = 1e-6) -> torch.distributions.MultivariateNormal:
        if grouping_factor is None:
            if len(self.grouping_factors) > 1:
                raise ValueError("Must specify `grouping_factor`")
            grouping_factor = self.grouping_factors[0]

        assert self.covariance_parameterization in {'low_rank', 'log_cholesky'}
        if self.covariance_parameterization == 'low_rank':
            covariance_matrix = (
                    self._re_cov_params[f'{grouping_factor}_lr'] @ self._re_cov_params[f'{grouping_factor}_lr'].t() +
                    torch.diag_embed(self._re_cov_params[f'{grouping_factor}_log_std_devs'].exp() ** 2)
            )
        else:
            L = log_chol_to_chol(
                log_diag=self._re_cov_params[f'{grouping_factor}_cholesky_log_diag'],
                off_diag=self._re_cov_params[f'{grouping_factor}_cholesky_off_diag']
            )
            covariance_matrix = L @ L.t()

        covariance_matrix = covariance_matrix * self.residual_var + eps * torch.eye(len(covariance_matrix))

        try:
            dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(len(covariance_matrix)), covariance_matrix=covariance_matrix
            )
        except RuntimeError as e:
            if 'cholesky' not in str(e):
                raise e
            dist = None
        if dist is None or not torch.isclose(dist.covariance_matrix, dist.precision_matrix.inverse(), atol=1e-04).all():
            if eps < .01:
                if eps > 1e-04:
                    warn(f"eps of {eps} insufficient to ensure posdef, increasing 10x")
                return self.re_distribution(grouping_factor, eps=eps * 10)
        return dist

    def set_re_cov(self, grouping_factor: str, cov: torch.Tensor):
        rel_cov = cov / self.residual_var
        if self.covariance_parameterization == 'low_rank':
            raise NotImplementedError("Cannot set cov when `covariance_parameterization == 'low_rank'`")
        with torch.no_grad():
            L = torch.cholesky(rel_cov)
            self._re_cov_params[f'{grouping_factor}_cholesky_log_diag'].data[:] = L.diag().log()
            self._re_cov_params[f'{grouping_factor}_cholesky_off_diag'].data[:] = \
                L[tuple(torch.tril_indices(len(L), len(L), offset=-1))]
            assert torch.isclose(self.re_distribution(grouping_factor).covariance_matrix, cov, atol=1e-04).all()

    def _is_cov_param(self, param: torch.Tensor) -> bool:
        return any(param is cov_param for cov_param in self._re_cov_params.values())

    def _get_prior_precisions(self, detach: bool) -> Dict[str, torch.Tensor]:
        out = {}
        for gf in self.grouping_factors:
            out[gf] = self.re_distribution(gf).precision_matrix
            if detach:
                out[gf] = out[gf].detach()
        return out

    # forward / re-solve -------
    def forward(self,
                X: torch.Tensor,
                group_ids: Sequence,
                re_solve_data: Optional[Tuple[torch.Tensor, torch.Tensor, Sequence]] = None,
                res_per_gf: Optional[Union[dict, torch.Tensor]] = None) -> torch.Tensor:
        """

        :param X: A 2D model-matrix Tensor
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param re_solve_data: A tuple of (X,y,group_ids) that will be used for establishing the random-effects.
        :param res_per_gf: Instead of passing `re_solve_data` and having the module solve the random-effects per
        group, can alternatively pass these random-effects. This should be a dictionary whose keys are grouping-factors
        and whose values are tensors. Each tensor's row corresponds to the groups for that grouping factor, in sorted
        order. If there is only one grouping factor, can pass a tensor instead of a dict.
        :return: Tensor of predictions
        """
        X, = self._validate_tensors(X)

        if res_per_gf is None:
            if re_solve_data is None:
                raise TypeError("Must pass one of `re_solve_data`, `res_per_gf`; got neither.")
            # `res_per_gf` are matrices with rows corresponding to groups; so it needs to assume a fixed set of groups
            *_, rs_group_ids = re_solve_data
            group_ids = self._validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))
            rs_group_ids = self._validate_group_ids(rs_group_ids, num_grouping_factors=len(self.grouping_factors))
            for i, gf in enumerate(self.grouping_factors):
                if set(group_ids[:, i]) != set(rs_group_ids[:, i]):
                    raise RuntimeError(
                        f"Please remove levels of '{gf}' from `re_solve_data` that are not present in `X`/`group_ids`."
                    )

            with torch.no_grad():
                res_per_gf = self.get_res(*re_solve_data)
                return self.forward(X=X, group_ids=group_ids, re_solve_data=None, res_per_gf=res_per_gf)

        if re_solve_data is not None:
            raise TypeError("Must pass one of `re_solve_data`, `betas_per_group`; got both.")

        # validate group_ids
        group_ids = self._validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))

        # get yhat for raneffects:
        if not isinstance(res_per_gf, dict):
            if len(self.grouping_factors) > 1:
                raise ValueError("`res_per_gf` should be a dictionary (unless there's only one grouping factor)")
            res_per_gf = {self.grouping_factors[0]: res_per_gf}
        yhat_r = self._get_yhat_r(design=self.rf_idx, X=X, group_ids=group_ids, res_per_gf=res_per_gf)

        # yhat for fixed-effects:
        yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))

        # combine:
        return yhat_f + yhat_r

    def get_res(self,
                X: torch.Tensor,
                y: torch.Tensor,
                group_ids: Sequence) -> Dict[str, torch.Tensor]:
        """
        Get the random-effects.
        :param X: A 2D model-matrix Tensor
        :param y: A 1D target/response Tensor.
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :return: A dictionary with grouping-factors as keys and random-effects tensors as values. These tensors have
        rows corresponding to the sorted group_ids.
        """
        X, y = self._validate_tensors(X, y)

        solver = self.solver_cls(
            X=X, y=y,
            group_ids=self._validate_group_ids(group_ids, len(self.grouping_factors)),
            design=self.rf_idx
        )
        return solver(
            fe_offset=validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx])),
            prior_precisions=self._get_prior_precisions(detach=True)
        )

    # fitting --------
    def fit_loop(self,
                 X: Union[torch.Tensor, np.ndarray],
                 y: Union[torch.Tensor, np.ndarray],
                 group_ids: Sequence,
                 optimizer: torch.optim.Optimizer,
                 **kwargs) -> Iterator[float]:
        X, y = self._validate_tensors(X, y)
        group_ids = self._validate_group_ids(group_ids, len(self.grouping_factors))

        # check whether cov is fixed. if so, can avoid passing it on each call to `closure`.
        # this is useful for subclasses (e.g. Logistic) that do expensive operations on cov
        fixed_cov = True
        for grp in optimizer.param_groups:
            for p in grp['params']:
                if self._is_cov_param(p):
                    fixed_cov = False
                    break

        # progress bar
        if not self.verbose:
            progress = None
        elif isinstance(optimizer, torch.optim.LBFGS):
            progress = tqdm(total=optimizer.param_groups[0]['max_eval'])
        else:
            progress = tqdm(total=1)

        # initialize the solver:
        if self.likelihood_requires_raneffs:
            solver = self.solver_cls(
                X=X,
                y=y,
                group_ids=group_ids,
                design=self.rf_idx,
                prior_precisions=self._get_prior_precisions(detach=True) if fixed_cov else None
            )

        # create the closure which returns the loss
        def closure():
            optimizer.zero_grad()

            loss_kwargs = {'X': X, 'y': y, 'group_ids': group_ids}
            if self.likelihood_requires_raneffs:
                yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))
                loss_kwargs['res_per_gf'] = solver(
                    fe_offset=yhat_f.detach(),
                    prior_precisions=None if fixed_cov else self._get_prior_precisions(detach=False),
                    **kwargs
                )

            loss = self.get_loss(**loss_kwargs)
            loss.backward()
            if progress:
                progress.update()
                progress.set_description(f"Epoch {epoch:,}; Loss {loss.item():.4}")
            return loss

        epoch = 0
        while True:
            if progress:
                progress.reset()
                progress.set_description(f"Epoch {epoch:,}")
            floss = optimizer.step(closure).item()
            yield floss
            epoch += 1

    def get_loss(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 res_per_gf: dict = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _validate_tensors(*args) -> Iterator:
        for arg in args:
            arg = ndarray_to_tensor(arg).to(torch.float32)
            if torch.isnan(arg).any():
                raise ValueError("`nans` in tensor")
            if torch.isinf(arg).any():
                raise ValueError("`infs` in tensor")
            yield arg

    @staticmethod
    def _validate_group_ids(group_ids: Sequence, num_grouping_factors: int) -> np.ndarray:
        group_ids = np.asanyarray(group_ids)
        if num_grouping_factors > 1:
            if len(group_ids.shape) != 2 or len(group_ids.shape[1]) != num_grouping_factors:
                raise ValueError(
                    f"There are {num_grouping_factors} grouping-factors, so `group_ids` should be 2d with 2nd "
                    f"dimension of this extent."
                )
        else:
            group_ids = validate_1d_like(group_ids)[:, None]
        return group_ids

    @staticmethod
    def _get_yhat_r(design: dict,
                    X: torch.Tensor,
                    group_ids: np.ndarray,
                    res_per_gf: dict,
                    reduce: bool = True) -> torch.Tensor:
        yhat_r = torch.empty(*group_ids.shape)
        for i, (gf, col_idx) in enumerate(design.items()):
            Xr = torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1)
            _, group_idx = np.unique(group_ids[:, i], return_inverse=True)
            betas_broad = res_per_gf[gf][group_idx]
            yhat_r[:, i] = (Xr * betas_broad).sum(1)
        if reduce:
            yhat_r = yhat_r.sum(1)
        return yhat_r


class ReSolver:
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 design: dict,
                 prior_precisions: Optional[dict] = None):
        """
        Initialize this random-effects solver

        :param X: A model-matrix Tensor
        :param y: A response/target Tensor
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param design: An ordered dictionary whose keys are grouping-factor labels and whose values are column-indices
        in `X` that are used for the random-effects of that grouping factor.
        """
        self.X, self.y = MixedEffectsModule._validate_tensors(X, y)
        self.group_ids = MixedEffectsModule._validate_group_ids(group_ids, num_grouping_factors=len(design))
        self.design = design
        self.prior_precisions = prior_precisions

        # establish static kwargs:
        self.static_kwargs_per_gf = {}
        for i, (gp, col_idx) in enumerate(self.design.items()):
            kwargs = {
                'X': torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1),
                'y': y,
                'group_ids': group_ids[:, i]
            }
            kwargs['XtX'] = torch.stack([
                Xg.t() @ Xg for Xg, in chunk_grouped_data(kwargs['X'], group_ids=kwargs['group_ids'])
            ])
            self.static_kwargs_per_gf[gp] = kwargs

    def __call__(self,
                 fe_offset: torch.Tensor,
                 max_iter: int = 200,
                 tol: float = .01,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Solve the random effects.

        :param fe_offset: The offset that comes from the fixed-effects.
        :param max_iter: The maximum number of iterations.
        :param tol: Tolerance for checking convergence.
        :param kwargs: Other keyword arguments used to create kwargs for `solve_step`
        :return: A dictionary with random-effects for each grouping-factor
        """
        assert max_iter > 0
        kwargs_per_gf = self._initialize_kwargs(fe_offset=fe_offset, **kwargs)

        self.history = []
        while True:
            if len(self.history) >= max_iter:
                # TODO: mask gradient for unconverged groups?
                if max_iter:
                    warn("TODO")
                break

            # take a step towards solving the random-effects:
            self.history.append({})
            for gf in self.design.keys():
                self.history[-1][gf] = self.solve_step(**kwargs_per_gf[gf])

            # check convergence:
            if self._check_convergence(tol=tol):
                # TODO: verbose
                # TODO: patience
                break

            # recompute offset, etc:
            kwargs_per_gf = self._update_kwargs(kwargs_per_gf, fe_offset=fe_offset)

        return self.history[-1]

    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: Sequence,
                   prior_precision: torch.Tensor,
                   **kwargs) -> torch.Tensor:
        """
        Update random-effects. For some solvers this might not be iterative given a single grouping factor
        because a closed-form solution exists (e.g. gaussian); however, this is still an iterative `step` in the case of
         multiple grouping-factors.
        """
        raise NotImplementedError

    def _initialize_kwargs(self, fe_offset: torch.Tensor, prior_precisions: Optional[dict] = None, **kwargs) -> dict:
        """
        Called at the start of the solver's __call__, this XXX

        :param fe_offset: XXX
        :param prior_precisions: XXX
        :param kwargs: XXX
        :return:
        """
        if kwargs:
            raise TypeError(f"Unexpected keyword-args:\n{set(kwargs.keys())}")

        if prior_precisions is None:
            # if we are not optimizing covariance, can avoid passing prior-precisions on each iter
            prior_precisions = self.prior_precisions

        kwargs_per_gf = {}
        for gf in self.design.keys():
            kwargs_per_gf[gf] = self.static_kwargs_per_gf[gf].copy()
            kwargs_per_gf[gf]['offset'] = fe_offset.clone()
            kwargs_per_gf[gf]['prior_precision'] = prior_precisions[gf]
        return kwargs_per_gf

    def _update_kwargs(self, kwargs_per_gf: dict, fe_offset: torch.Tensor) -> Optional[dict]:
        """
        Called on each iteration of the solver's __call__, this method recomputes the offset

        :param kwargs_per_gf: XXX
        :param fe_offset: XXX
        :return:
        """

        # update offsets:
        with torch.no_grad():
            yhat_r = MixedEffectsModule._get_yhat_r(
                design=self.design, X=self.X, group_ids=self.group_ids, res_per_gf=self.history[-1], reduce=False
            )
            out = {}
            for gf1 in kwargs_per_gf.keys():
                out[gf1] = kwargs_per_gf[gf1].copy()
                out[gf1]['offset'] = [fe_offset.clone()]
                for i, gf2 in enumerate(self.design.keys()):
                    if gf1 == gf2:
                        continue
                    out[gf1]['offset'].append(yhat_r[:, i])
                out[gf1]['offset'] = torch.sum(torch.stack(out[gf1]['offset']))

        return out

    def _check_convergence(self, tol: float) -> bool:
        if len(self.history) < 2:
            return False
        converged = True
        with torch.no_grad():
            for gf in self.design.keys():
                current_res = self.history[-1][gf]
                prev_res = self.history[-2][gf]
                changes = torch.norm(current_res - prev_res, dim=1) / torch.norm(prev_res, dim=1)
                if (changes > tol).any():
                    converged = False
                    break
        return converged
