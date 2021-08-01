from collections import OrderedDict
from typing import Union, Optional, Sequence, Iterator, Dict, Tuple, Type, Collection, Callable
from warnings import warn

import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from ..utils import validate_1d_like, validate_tensors, validate_group_ids, get_yhat_r, pad_res_per_gf
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
                 re_scale_penalty: Union[float, dict] = 0.):

        super().__init__()

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

        dist = None
        try:
            dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(len(covariance_matrix)), covariance_matrix=covariance_matrix,
                validate_args=True  # TODO: profile this
            )
        except ValueError as e:
            if 'invalid' not in str(e):
                raise e

        if dist is None:
            msg = f"eps of {eps} insufficient to ensure posdef covariance matrix for gf={grouping_factor}"
            if eps < .01:
                if eps > 1e-04:
                    warn(f"{msg}, increasing 10x")
                return self.re_distribution(grouping_factor, eps=eps * 10)
            else:
                raise RuntimeError(msg)
        return dist

    # forward / re-solve -------
    def forward(self,
                X: torch.Tensor,
                group_ids: Sequence,
                re_solve_data: Optional[tuple] = None,
                res_per_gf: Optional[Union[dict, torch.Tensor]] = None,
                quiet: bool = False) -> torch.Tensor:
        """

        :param X: A 2D model-matrix Tensor
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param re_solve_data: A tuple of (X,y,group_ids[,weights]) that will be used for establishing the ran-effects.
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

                res_per_gf_padded = pad_res_per_gf(res_per_gf, group_ids, rs_group_ids, verbose=not quiet)

                return self.forward(
                    X=X, group_ids=group_ids, re_solve_data=None, res_per_gf=res_per_gf_padded, quiet=quiet
                )

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
            re_solve_data: Optional[tuple] = None,
            res_per_gf: Optional[Union[dict, torch.Tensor]] = None,
            **kwargs
    ) -> torch.distributions.Distribution:
        """
        Predict the posterior mode as a ``torch.distributions.Distribution``.

        :param X: Model-matrix
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param re_solve_data: A tuple of (X,y,group_ids[,weights]) that will be used for establishing the ran-effects.
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
                weights: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Get the random-effects.
        :param X: A 2D model-matrix Tensor
        :param y: A 1D target/response Tensor.
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param weights: Optional sample-weights.
        :return: A dictionary with grouping-factors as keys and random-effects tensors as values. These tensors have
        rows corresponding to the sorted group_ids.
        """
        X, y = validate_tensors(X, y)

        solver = self.solver_cls(
            X=X, y=y,
            group_ids=validate_group_ids(group_ids, len(self.grouping_factors)),
            design=self.rf_idx,
            weights=weights,
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
                 weights: Optional[torch.Tensor] = None,
                 loss_type: Optional[str] = None,
                 callbacks: Collection[Callable] = (),
                 **kwargs) -> Iterator[float]:

        X, y = validate_tensors(X, y)
        group_ids = validate_group_ids(group_ids, len(self.grouping_factors))

        # solver will go here:
        cache = {}

        # create the closure which returns the loss
        def closure():
            optimizer.zero_grad()
            loss = self.get_loss(X, y, group_ids, loss_type=loss_type, cache=cache, weights=weights, **kwargs)
            if torch.isnan(loss):
                raise RuntimeError("Encountered `nan` loss in training.")
            loss.backward()
            for callback in callbacks:
                callback(loss)
            return loss

        while True:
            floss = optimizer.step(closure).item()
            yield floss

    def get_loss(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 weights: Optional[torch.Tensor] = None,
                 cache: Optional[dict] = None,
                 loss_type: Optional[str] = None,
                 reduce: bool = True,
                 **kwargs) -> torch.Tensor:

        if loss_type is None:
            loss_type = self.default_loss_type

        if cache is None:
            cache = {}

        loss = self._get_loss(
            X=X, y=y, group_ids=group_ids, cache=cache, loss_type=loss_type, weights=weights, **kwargs
        )

        loss = loss + self._get_re_penalty()

        if reduce:
            loss = loss / len(X)
        return loss

    def _get_loss(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  group_ids: np.ndarray,
                  weights: Optional[torch.Tensor],
                  cache: dict,
                  loss_type: Optional[str] = None,
                  **kwargs):

        if loss_type.startswith('cv'):
            return self._get_cv_loss(
                X=X, y=y, group_ids=group_ids, cache=cache, weights=weights, **kwargs
            )
        elif loss_type.startswith('iid'):
            return self._get_iid_loss(
                X=X, y=y, group_ids=group_ids, cache=cache, weights=weights, **kwargs
            )
        else:
            raise NotImplementedError(f"{type(self).__name__} does not implement type={loss_type}")

    def _get_cv_loss(self,
                     X: torch.Tensor,
                     y: torch.Tensor,
                     group_ids: np.ndarray,
                     weights: Optional[torch.Tensor],
                     cache: dict,
                     cv_config: Optional[dict] = None,
                     **kwargs) -> torch.Tensor:
        """
        For many types of mixed-effects models, finding the posterior modes is tractable but evaluating the full
        likelihood is not. One option afforded by autograd is to optimize the covariance based on iid likelihood on
        *holdout* data. This method involves generating splits within each group of varying sizes, and optimizing the
        parameters so minimize loss on the test-splits.

        :param X: A 2D model-matrix Tensor
        :param y: A 1D target/response Tensor.
        :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
        tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
        :param cache: Dictionary of cached objects that don't need to be re-created each call (e.g. ``ReSolver``)
        :param cv_config: A dictionary with config for the cross-validation. This is experimental and subject to
        change, current options are: `n_splits` for the number of splits per each round of stratified splitting (default
         3); `quantile_width` for controlling the number of rounds of splits, and the sizes of the train/test-split on
         each (e.g. default of .20 will create perform 4 rounds of stratified splitting train/test with proportions
        `[.20/.80, .40/.60, .60/.40, .80/.20]`); and `random_state` (default None) for controlling the random-state of
         the splitter.
        :param kwargs: Other keyword args to pass to each ``ReSolver``.
        :return: The loss
        """
        if weights is None:
            weights = torch.ones_like(y)
        assert len(weights) == len(y)
        prior_precisions = self._get_prior_precisions(detach=False) if not self.fixed_cov else None
        log_probs = []
        for k, v in cache.items():
            if not isinstance(k, tuple) and isinstance(v, tuple):
                continue
            gf, q, _ = k
            solver, test_idx = v
            res_per_gf = solver(
                fe_offset=validate_1d_like(self.fixed_effects_nn(solver.X[:, self.ff_idx])),
                prior_precisions=prior_precisions
            )
            res_per_gf = pad_res_per_gf(
                res_per_gf, group_ids1=solver.group_ids, group_ids2=group_ids[test_idx], verbose=solver.verbose
            )
            dist = self.predict_distribution_mode(X=X[test_idx], group_ids=group_ids[test_idx], res_per_gf=res_per_gf)
            log_probs.append(torch.sum(dist.log_prob(y[test_idx]) * weights[test_idx]))

        if log_probs:
            return -torch.sum(torch.stack(log_probs) / len(self.grouping_factors))
        else:
            # TODO: choose breaks by over-weighting regions w/more uncertainty
            _defaults = {'quantile_width': .20, 'n_splits': 3, 'random_state': None}
            cv_config = {**_defaults, **(cv_config or {})}

            for gfi, gf in enumerate(self.grouping_factors):
                # calculate quantiles:
                _, counts = np.unique(group_ids[:, gfi], return_counts=True)
                q = max(1 / np.median(counts), cv_config['quantile_width'])
                assert q <= .5, f"Cannot do cv on this dataset, median group-size for {gf} is <2"
                quantiles = np.arange(q, 1., q)
                if 1 - quantiles[-1] < q / 2:
                    quantiles = quantiles[0:-1]

                # create splits + solvers for each:
                for q in quantiles:
                    splitter = StratifiedShuffleSplit(
                        n_splits=cv_config['n_splits'], test_size=q, random_state=cv_config['random_state']
                    )
                    for i, (train_idx, test_idx) in enumerate(splitter.split(X=X, y=group_ids[:, gfi]), 1):
                        solver = self.solver_cls(
                            X=X[train_idx],
                            y=y[train_idx],
                            group_ids=group_ids[train_idx],
                            weights=weights[train_idx],
                            design=self.rf_idx,
                            prior_precisions=self._get_prior_precisions(detach=True) if self.fixed_cov else None,
                            **kwargs
                        )
                        cache[(gf, q, i)] = (solver, test_idx)
            return self._get_cv_loss(
                X=X,
                y=y,
                group_ids=group_ids,
                weights=weights,
                cache=cache,
                cv_config=cv_config,
                **kwargs
            )

    def _get_iid_loss(self,
                      X: torch.Tensor,
                      y: torch.Tensor,
                      group_ids: np.ndarray,
                      weights: Optional[torch.Tensor],
                      cache: dict,
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
        if weights is None:
            weights = torch.ones_like(y)
        assert len(weights) == len(y)

        if 'solver' not in cache:
            cache['solver'] = self.solver_cls(
                X=X,
                y=y,
                group_ids=group_ids,
                weights=weights,
                design=self.rf_idx,
                prior_precisions=self._get_prior_precisions(detach=True),
                **kwargs
            )
        solver = cache['solver']

        yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))
        res_per_gf = solver(fe_offset=yhat_f)
        dist = self.predict_distribution_mode(X=X, group_ids=group_ids, res_per_gf=res_per_gf)
        log_prob_lik = torch.sum(dist.log_prob(y) * weights)
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
