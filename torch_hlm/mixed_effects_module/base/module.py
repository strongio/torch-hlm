from collections import OrderedDict
from typing import Union, Optional, Sequence, Iterator, Dict, Tuple, Type
from warnings import warn

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm

from ..utils import validate_1d_like, validate_tensors, validate_group_ids, get_yhat_r, get_to_kwargs
from ...covariance import Covariance


class MixedEffectsModule(torch.nn.Module):
    """
    Base-class for a torch.nn.Module that can be trained with mixed-effects

    :param fixeff_features: The column-indices of features for fixed-effects model matrix (not including
    bias/intercept); if a single integer is passed then these indices are ``range(fixeff_features)``
    :param raneff_features: If there is a single grouping factor, then this argument is like ``fixeff_features``: the
     indices in the model-matrix corresponding to random-effects. If there are multiple grouping factors, then the
     keys are grouping factor labels and the values are these indices.
    :param fixed_effects_nn: An optional ``torch.nn.Module`` for the fixed-effects. The default is to create a
     ``torch.nn.Linear(num_fixed_features, 1)``.
    :param covariance: See ``torch_hlm.covariance``. Default is 'log_cholesky'.
    :param re_scale_penalty: In some settings, optimization will fail because the variance on some random effects
     is too high (e.g. if there are a very large number of observations per group, the random-intercept might have
     high variance) which causes numerical issues. Setting `re_scale_penalty>0` can help in these cases. This
     corresponds to a half-normal prior on the random-effect std-devs, with precision = re_scale_penalty.
    :param verbose: Verbosity level; 1 (default) prints the loss on each optimization epoch; 0 disables everything,
     and 2 allows messages from the inner random-effects solver.
    """

    solver_cls: Type['ReSolver'] = None
    default_loss_type: str = None
    _nm2cls = {}

    def __init_subclass__(cls, **kwargs):
        cls._nm2cls[cls.__name__.replace('MixedEffectsModule', '').lower()] = cls

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> 'MixedEffectsModule':
        klass = cls._nm2cls.get(name)
        if klass is None:
            raise TypeError(f"Unrecognized `name`, options: {set(cls._nm2cls)}")
        return klass(*args, **kwargs)

    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Dict[str, Union[int, Sequence]],
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 covariance: str = 'log_cholesky',
                 re_scale_penalty: Union[float, dict] = 0.,
                 verbose: int = 1):

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
            if num_fixed_features:
                fixed_effects_nn = torch.nn.Linear(num_fixed_features, 1)
                with torch.no_grad():
                    fixed_effects_nn.weight[:] = .01 * torch.randn(num_fixed_features)
            else:
                fixed_effects_nn = _NoWeightModule(1)
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

        if not isinstance(covariance, dict):
            assert isinstance(covariance, str)
            covariance = {gf: covariance for gf in self.grouping_factors}
        self.covariance = torch.nn.ModuleDict()
        for gf, idx in self.rf_idx.items():
            gf_cov = covariance[gf]
            if isinstance(covariance[gf], str):
                self.covariance[gf] = Covariance.from_name(gf_cov, rank=len(idx) + 1)
            else:
                assert isinstance(gf_cov, Covariance), f"expecteed {gf_cov} to be string or `Covariance`"
                self.covariance[gf] = gf_cov

        if not isinstance(re_scale_penalty, dict):
            re_scale_penalty = {gf: float(re_scale_penalty) for gf in self.grouping_factors}
        self.re_scale_penalty = re_scale_penalty

    @property
    def residual_var(self) -> Union[torch.Tensor, float]:
        return 1.0

    @property
    def grouping_factors(self) -> Sequence:
        return list(self.rf_idx.keys())

    def _get_prior_precisions(self, detach: bool) -> Dict[str, torch.Tensor]:
        out = {}
        for gf in self.grouping_factors:
            out[gf] = self.re_distribution(gf).precision_matrix
            if detach:
                out[gf] = out[gf].detach()
        return out

    def re_distribution(self, grouping_factor: str = None, eps: float = 1e-6) -> torch.distributions.MultivariateNormal:
        if grouping_factor is None:
            if len(self.grouping_factors) > 1:
                raise ValueError("Must specify `grouping_factor`")
            grouping_factor = self.grouping_factors[0]

        covariance_matrix = self.covariance[grouping_factor]() * self.residual_var
        covariance_matrix = covariance_matrix + eps * torch.eye(len(covariance_matrix))
        try:
            dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(len(covariance_matrix)), covariance_matrix=covariance_matrix,
                validate_args=False
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
        X, = validate_tensors(X)

        if res_per_gf is None:
            if re_solve_data is None:
                raise TypeError("Must pass one of `re_solve_data`, `res_per_gf`; got neither.")
            *_, rs_group_ids = re_solve_data
            group_ids = validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))
            rs_group_ids = validate_group_ids(rs_group_ids, num_grouping_factors=len(self.grouping_factors))

            with torch.no_grad():
                # solve random-effects:
                if len(rs_group_ids):
                    res_per_gf = self.get_res(*re_solve_data)
                else:
                    # i.e. re_solve_data is empty
                    res_per_gf = {gf: torch.zeros((0, len(idx) + 1)) for gf, idx in self.rf_idx.items()}

                # there is no requirement that all groups in `group_ids` are present in `group_data`, or vice versa, so
                # need to map the re_solve output
                res_per_gf_padded = {}
                for gf_i, gf in enumerate(self.grouping_factors):
                    ugroups_target = {gid: i for i, gid in enumerate(np.unique(group_ids[:, gf_i]))}
                    ugroups_solve = {gid: i for i, gid in enumerate(np.unique(rs_group_ids[:, gf_i]))}
                    set1 = set(ugroups_solve) - set(ugroups_target)
                    if set1 and self.verbose:
                        print(f"there are {len(set1):,} groups in `re_solve_data` but not in `X`")
                    set2 = set(ugroups_target) - set(ugroups_solve)
                    if set2 and self.verbose:
                        print(f"there are {len(set2):,} groups in `X` but not in `re_solve_data`")

                    res_per_gf_padded[gf] = torch.zeros((len(ugroups_target), len(self.rf_idx[gf]) + 1))
                    for gid_target, idx_target in ugroups_target.items():
                        idx_solve = ugroups_solve.get(gid_target)
                        if idx_solve is None:
                            continue
                        res_per_gf_padded[gf][idx_target] = res_per_gf[gf][idx_solve]

                return self.forward(X=X, group_ids=group_ids, re_solve_data=None, res_per_gf=res_per_gf_padded)

        if re_solve_data is not None:
            raise TypeError("Must pass one of `re_solve_data`, `betas_per_group`; got both.")

        # validate group_ids
        group_ids = validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))

        # get yhat for raneffects:
        if not isinstance(res_per_gf, dict):
            if len(self.grouping_factors) > 1:
                raise ValueError("`res_per_gf` should be a dictionary (unless there's only one grouping factor)")
            res_per_gf = {self.grouping_factors[0]: res_per_gf}
        yhat_r = get_yhat_r(design=self.rf_idx, X=X, group_ids=group_ids, res_per_gf=res_per_gf).sum(1)

        # yhat for fixed-effects:
        yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))

        # combine:
        return yhat_f + yhat_r

    def predict_distribution_mode(
            self,
            X: torch.Tensor,
            group_ids: Sequence,
            re_solve_data: Optional[Tuple[torch.Tensor, torch.Tensor, Sequence]] = None,
            res_per_gf: Optional[Union[dict, torch.Tensor]] = None,
            **kwargs
    ) -> torch.distributions.Distribution:
        """
        Predict the posterior mode as a ``torch.distributions.Distribution``.

        :param X: Model-matrix
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param re_solve_data: A tuple of (X,y,group_ids) that will be used for establishing the random-effects.
        :param res_per_gf: Instead of passing `re_solve_data` and having the module solve the random-effects per
        group, can alternatively pass these random-effects. This should be a dictionary whose keys are grouping-factors
        and whose values are tensors. Each tensor's row corresponds to the groups for that grouping factor, in sorted
        order. If there is only one grouping factor, can pass a tensor instead of a dict.
        :param kwargs: Other arguments to pass to the distribution's init.
        :return: A ``torch.distributions.Distribution``
        """
        raise NotImplementedError

    def get_res(self,
                X: torch.Tensor,
                y: torch.Tensor,
                group_ids: Sequence,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Get the random-effects.
        :param X: A 2D model-matrix Tensor
        :param y: A 1D target/response Tensor.
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :return: A dictionary with grouping-factors as keys and random-effects tensors as values. These tensors have
        rows corresponding to the sorted group_ids.
        """
        X, y = validate_tensors(X, y)

        solver = self.solver_cls(
            X=X, y=y,
            group_ids=validate_group_ids(group_ids, len(self.grouping_factors)),
            design=self.rf_idx,
            **kwargs
        )
        return solver(
            fe_offset=validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx])),
            prior_precisions=self._get_prior_precisions(detach=True)
        )

    @property
    def fixed_cov(self) -> bool:
        # check whether cov is fixed. if so, can avoid passing it on each call to `closure`.
        # this is useful for subclasses (e.g. Logistic) that do expensive operations on cov
        fixed_cov = True
        for p in self.parameters():
            if p.requires_grad and self._is_cov_param(p):
                fixed_cov = False
                break
        return fixed_cov

    def _is_cov_param(self, param: torch.Tensor) -> bool:
        return any(param is cov_param for cov_param in self.covariance.parameters())

    # fitting --------
    def fit_loop(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: Sequence,
                 optimizer: torch.optim.Optimizer,
                 loss_type: Optional[str] = None,
                 **kwargs) -> Iterator[float]:

        X, y = validate_tensors(X, y)
        group_ids = validate_group_ids(group_ids, len(self.grouping_factors))

        # progress bar
        if not self.verbose:
            progress = None
        elif isinstance(optimizer, torch.optim.LBFGS):
            progress = tqdm(total=optimizer.param_groups[0]['max_eval'])
        else:
            progress = tqdm(total=1)

        # solver will go here:
        cache = {}

        # create the closure which returns the loss
        epoch = 0

        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(X, y, group_ids, loss_type=loss_type, cache=cache, **kwargs)
            if torch.isnan(loss):
                raise RuntimeError("Encountered `nan` loss in training.")
            loss.backward()
            if progress:
                progress.update()
                progress.set_description(f"Epoch {epoch:,}; Loss {loss.item():.4}")
            return loss

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
                 cache: Optional[dict] = None,
                 loss_type: Optional[str] = None,
                 reduce: bool = True,
                 **kwargs) -> torch.Tensor:

        if loss_type is None:
            loss_type = self.default_loss_type

        loss = self._get_loss(X=X, y=y, group_ids=group_ids, cache=cache, loss_type=loss_type, **kwargs)

        loss = loss + self._get_re_penalty()

        if reduce:
            loss = loss / len(X)
        return loss

    def _get_loss(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  group_ids: np.ndarray,
                  cache: Optional[dict] = None,
                  loss_type: Optional[str] = None,
                  **kwargs):
        if loss_type.startswith('cv'):
            return self._get_cv_loss(X=X, y=y, group_ids=group_ids, cache=cache, **kwargs)
        elif loss_type.startswith('iid'):
            return self._get_iid_loss(X=X, y=y, group_ids=group_ids, cache=cache, **kwargs)
        else:
            raise NotImplementedError(f"{type(self).__name__} does not implement type={loss_type}")

    def _get_cv_loss(self,
                     X: torch.Tensor,
                     y: torch.Tensor,
                     group_ids: np.ndarray,
                     cache: Optional[dict] = None,
                     cv: Optional[Iterator] = None,
                     **kwargs):
        """
        XXX
        :return:
        """
        cache = cache or {}

        if cv is None:
            cv = 10
        if isinstance(cv, int):
            cv = StratifiedKFold(n_splits=cv)  # TODO: random-state?

        have_solvers = [f'fold{i + 1}' in cache for i in range(cv.n_splits)]
        if not all(have_solvers):
            assert not any(have_solvers)  # that would be weird
            for i, (train_idx, test_idx) in enumerate(cv.split(X=X, y=group_ids[:, 0])):  # TODO: select grouping factor
                solver = self.solver_cls(
                    X=X[train_idx],
                    y=y[train_idx],
                    group_ids=group_ids[train_idx],
                    design=self.rf_idx,
                    prior_precisions=self._get_prior_precisions(detach=True) if self.fixed_cov else None,
                    verbose=self.verbose > 1,
                    **kwargs
                )
                cache[f'fold{i + 1}'] = (solver, test_idx)

        log_probs = []
        for k, v in cache.items():
            if not k.startswith('fold'):
                continue
            solver, test_idx = v
            res_per_gf = solver(
                fe_offset=validate_1d_like(self.fixed_effects_nn(solver.X[:, self.ff_idx])),
                prior_precisions=self._get_prior_precisions(detach=False) if solver.prior_precisions is None else None
            )
            for i in range(solver.group_ids.shape[-1]):
                assert np.array_equal(np.unique(solver.group_ids[:, i]), np.unique(group_ids[test_idx, i]))
            dist = self.predict_distribution_mode(X=X[test_idx], group_ids=group_ids[test_idx], res_per_gf=res_per_gf)
            log_probs.append(dist.log_prob(y[test_idx]).sum())
        return -torch.sum(torch.stack(log_probs))

    def _get_iid_loss(self,
                      X: torch.Tensor,
                      y: torch.Tensor,
                      group_ids: np.ndarray,
                      cache: Optional[dict] = None,
                      **kwargs) -> torch.Tensor:
        """
        If the re-cov is known and fixed, we can optimized fixed effects using likelihood of the posterior modes.

        :param X: A 2D model-matrix Tensor
        :param y: A 1D target/response Tensor.
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param cache: Dictionary of cached objects that don't need to be re-created each call (e.g. ``ReSolver``)
        :return: The log-prob for the data.
        """
        if not self.fixed_cov:
            raise RuntimeError(
                f"Cannot use `iid_loss` when optimizing covariance. Need to pass known covariances to "
                f"``{type(self).__name__}.set_re_cov()``, or need to change ``loss_type``."
            )

        X, y = validate_tensors(X, y)
        group_ids = validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))

        cache = cache or {}
        if 'solver' in cache:
            solver = cache['solver']
        else:
            solver = self.solver_cls(
                X=X,
                y=y,
                group_ids=group_ids,
                design=self.rf_idx,
                prior_precisions=self._get_prior_precisions(detach=True),
                verbose=self.verbose > 1,
                **kwargs
            )

        yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))
        res_per_gf = solver(fe_offset=yhat_f)
        dist = self.predict_distribution_mode(X=X, group_ids=group_ids, res_per_gf=res_per_gf)
        log_prob_lik = dist.log_prob(y).sum()
        return -log_prob_lik

    def _get_re_penalty(self) -> Union[float, torch.Tensor]:
        """
        In some settings, optimization will fail because the variance on some random effects
        is too high (e.g. if there are a very large number of observations per group, the random-intercept might have
        high variance) which causes numerical issues. Setting `re_scale_penalty>0` can help in these cases. This
        """
        penalties = []
        for gf, precision in self.re_scale_penalty.items():
            if not precision:
                continue
            dist = torch.distributions.HalfNormal(np.sqrt(1 / precision))
            vars = self.re_distribution(gf).covariance_matrix.diag()
            std_devs = (vars / self.residual_var).sqrt()
            penalties.append(dist.log_prob(std_devs))
        if not penalties:
            return 0.
        return -torch.cat(penalties).sum()


class _NoWeightModule(torch.nn.Module):
    def __init__(self, out_features: int = 1):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.weight = torch.empty((0, out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(len(input), -1)
