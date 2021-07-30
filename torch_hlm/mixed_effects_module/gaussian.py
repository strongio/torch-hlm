from collections import defaultdict
from typing import Sequence, Union, Optional, Dict, Tuple

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver
from .utils import chunk_grouped_data, validate_tensors, validate_group_ids
from torch.distributions import MultivariateNormal

from torch_hlm.low_rank import LowRankMultivariateNormal


class GaussianReSolver(ReSolver):

    def __call__(self,
                 fe_offset: torch.Tensor,
                 max_iter: Optional[int] = None,
                 reltol: float = .01,
                 prior_precisions: Optional[dict] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        # TODO: option for closed form w/multiple grouping factors, instead of iterative
        return super(GaussianReSolver, self).__call__(
            fe_offset=fe_offset,
            max_iter=max_iter,
            reltol=reltol,
            prior_precisions=prior_precisions,
            **kwargs
        )

    # noinspection PyMethodOverriding
    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: np.ndarray,
                   XtX: torch.Tensor,
                   prior_precision: torch.Tensor,
                   prev_res: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param X: N*K Tensor model-matrix
        :param y: N Tensor of target/response
        :param offset: N Tensor offset (e.g. fixed-effects predictions)
        :param group_ids: 1D array of group-identifiers for each row in X/y
        :param XtX: A G*K*K batch Tensor, full of each group's model-matrix times its own transpose. Critically, it is
        assumed that the order of the batches corresponds to the sorted `group_ids`.
        :param prior_precision: A K*K Tensor with the prior-precision
        :return: A G*K Tensor with the random-effects. Each row corresponds to the sorted `group_ids`.
        """
        num_groups = len(XtX)
        num_obs, num_res = X.shape

        group_ids_seq = rankdata(group_ids, method='dense') - 1
        group_ids_broad = torch.tensor(group_ids_seq, dtype=torch.int64).unsqueeze(-1).expand(-1, num_res)

        prior_precision = prior_precision.expand(len(XtX), -1, -1).clone()

        assert y.shape == offset.shape, (y.shape, offset.shape)

        if prev_res is not None:
            offset = offset + (prev_res[group_ids_seq] * X).sum(1)

        yoff = y - offset
        Xty_els = X * yoff.unsqueeze(1)  # TODO: sample_weights
        Xty = torch.zeros(num_groups, num_res).scatter_add(0, group_ids_broad, Xty_els)

        iter_ = len(self.history) + 1
        m = 1 - (iter_ / (float(self.slow_start) + iter_))
        extra_penalty = (m ** 2) * num_obs * torch.eye(num_res)

        res = torch.solve(Xty.unsqueeze(-1), XtX + prior_precision + extra_penalty)[0].squeeze(-1)

        if prev_res is not None:
            res = prev_res + res
        return res

    def _check_if_converged(self, reltol: float) -> bool:
        if len(self.design) == 1:
            assert not self.slow_start
            # if only one grouping factor, then solution is closed form, not iterative
            return True
        return super()._check_if_converged(reltol=reltol)


class GaussianMixedEffectsModule(MixedEffectsModule):
    solver_cls = GaussianReSolver
    default_loss_type = 'closed_form'

    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Dict[str, Union[int, Sequence]],
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 covariance: str = 'log_cholesky',
                 re_scale_penalty: Union[float, dict] = 0.,
                 verbose: int = 1):
        super().__init__(
            fixeff_features=fixeff_features,
            raneff_features=raneff_features,
            fixed_effects_nn=fixed_effects_nn,
            covariance=covariance,
            re_scale_penalty=re_scale_penalty,
            verbose=verbose
        )
        self._residual_std_dev_log = torch.nn.Parameter(.01 * torch.randn(1))

    def predict_distribution_mode(
            self,
            X: torch.Tensor,
            group_ids: Sequence,
            re_solve_data: Optional[Tuple[torch.Tensor, torch.Tensor, Sequence]] = None,
            res_per_gf: Optional[Union[dict, torch.Tensor]] = None,
            **kwargs
    ) -> torch.distributions.Distribution:
        if 'validate_args' not in kwargs:
            kwargs['validate_args'] = False
        pred = self(X=X, group_ids=group_ids, re_solve_data=re_solve_data, res_per_gf=res_per_gf)
        return torch.distributions.Normal(loc=pred, scale=self.residual_var ** .5)

    @property
    def residual_var(self) -> torch.Tensor:
        return self._residual_std_dev_log.exp() ** 2

    def _get_loss(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  group_ids: np.ndarray,
                  cache: Optional[dict] = None,
                  loss_type: Optional[str] = None):
        if loss_type == 'closed_form':
            return -self._get_log_prob(X=X, y=y, group_ids=group_ids)
        return super()._get_loss(X=X, y=y, group_ids=group_ids, cache=cache, loss_type=loss_type)

    def _get_log_prob(self,
                      X: torch.Tensor,
                      y: torch.Tensor,
                      group_ids: np.ndarray) -> torch.Tensor:

        X, y = validate_tensors(X, y)
        group_ids = validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))

        if len(self.grouping_factors) > 1:
            raise NotImplementedError
        # mercifully, if there is one grouping-factor, overall prob is just product of individual probs:
        gf = self.grouping_factors[0]

        # model-mat for raneffects:
        Z = torch.cat([torch.ones((len(X), 1)), X[:, self.rf_idx[gf]]], 1)

        Xs_by_r = defaultdict(list)
        ys_by_r = defaultdict(list)
        Zs_by_r = defaultdict(list)
        for Z_g, X_g, y_g in chunk_grouped_data(Z, X[:, self.ff_idx], y, group_ids=group_ids):
            Zs_by_r[len(Z_g)].append(Z_g)
            ys_by_r[len(Z_g)].append(y_g)
            Xs_by_r[len(Z_g)].append(X_g)

        # compute log-prob per batch:
        re_dist = self.re_distribution(gf)
        log_probs = []
        for r, y_r in ys_by_r.items():
            ng = len(y_r)
            X_r = torch.stack(Xs_by_r[r])
            Z_r = torch.stack(Zs_by_r[r])

            # counter-intuitively, it's faster (for `backward()`) if we call this one per batch here (vs. calling for
            # the entire dataset above)
            loc = self.fixed_effects_nn(X_r)
            if len(loc.shape) > 2:
                loc = loc.squeeze(-1)

            # mvnorm = LowRankMultivariateNormal(
            #     loc=loc,
            #     cov_diag=self.residual_var * torch.eye(r).expand(ng, -1, -1),  # TODO: sample weights applied here?
            #     cov_factor=Z_r,
            #     cov_factor_inner=re_dist.covariance_matrix.expand(ng, -1, -1),
            #     validate_args=False
            # )
            cov_r = Z_r @ re_dist.covariance_matrix.expand(ng, -1, -1) @ Z_r.permute(0, 2, 1)
            eps_r = (self.residual_var * torch.eye(r)).expand(ng, -1, -1)
            mvnorm = MultivariateNormal(loc=loc, covariance_matrix=eps_r + cov_r)

            log_probs.append(mvnorm.log_prob(torch.stack(y_r)))

        return torch.cat(log_probs).sum()
