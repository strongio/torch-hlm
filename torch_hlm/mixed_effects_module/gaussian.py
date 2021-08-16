from collections import defaultdict
from typing import Sequence, Union, Optional, Dict, Tuple
from warnings import warn

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver
from .utils import chunk_grouped_data, validate_tensors, validate_group_ids
from torch.distributions import MultivariateNormal

from torch_hlm.low_rank import LowRankMultivariateNormal


class GaussianReSolver(ReSolver):

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 weights: Optional[torch.Tensor],
                 design: dict,
                 **kwargs):
        self.iterative = len(design) > 1
        super().__init__(X=X, y=y, group_ids=group_ids, weights=weights, design=design, **kwargs)

    def _calculate_htild_inv(self, XtX: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
        return torch.inverse(XtX + pp)

    @staticmethod
    def ilink(x: torch.Tensor) -> torch.Tensor:
        return x

    def _calculate_grad(self,
                        X: torch.Tensor,
                        y: torch.Tensor,
                        mu: torch.Tensor) -> torch.Tensor:
        _, num_res = X.shape
        return X * (y - mu).unsqueeze(-1).expand(-1, num_res)

    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: np.ndarray,
                   weights: torch.Tensor,
                   prior_precision: torch.Tensor,
                   Htild_inv: torch.Tensor,
                   prev_res: Optional[torch.Tensor],
                   iter_: int,
                   converged_mask: Optional[np.ndarray] = None,
                   XtX: Optional[torch.Tensor] = None,
                   **kwargs
                   ) -> torch.Tensor:
        if self.iterative:
            return super(GaussianReSolver, self).solve_step(
                X=X,
                y=y,
                offset=offset,
                group_ids=group_ids,
                weights=weights,
                prior_precision=prior_precision,
                Htild_inv=Htild_inv,
                prev_res=prev_res,
                converged_mask=converged_mask,
                iter_=iter_
            )
        else:
            assert converged_mask is None
            assert XtX is not None
            num_obs, num_res = X.shape
            num_groups = len(XtX)
            group_ids_seq = torch.as_tensor(rankdata(group_ids, method='dense') - 1)
            group_ids_broad = group_ids_seq.unsqueeze(-1).expand(-1, num_res)
            assert y.shape == weights.shape
            yoff = (y - offset)
            Xty_els = X * yoff.unsqueeze(1) * weights.unsqueeze(1)
            Xty = torch.zeros(num_groups, num_res).scatter_add(0, group_ids_broad, Xty_els)
            return torch.solve(Xty.unsqueeze(-1), XtX + prior_precision)[0].squeeze(-1)

    @staticmethod
    def _get_hessian(
            X: torch.Tensor,
            weights: torch.Tensor,
            mu: torch.Tensor) -> torch.Tensor:
        return weights * X.t() @ X


class GaussianMixedEffectsModule(MixedEffectsModule):
    solver_cls = GaussianReSolver

    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Dict[str, Union[int, Sequence]],
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 covariance: str = 'log_cholesky',
                 re_scale_penalty: Union[float, dict] = 0.):
        super().__init__(
            fixeff_features=fixeff_features,
            raneff_features=raneff_features,
            fixed_effects_nn=fixed_effects_nn,
            covariance=covariance,
            re_scale_penalty=re_scale_penalty,
        )
        self._residual_std_dev_log = torch.nn.Parameter(.01 * torch.randn(1))

    def _forward_to_distribution(self, pred: torch.Tensor, **kwargs) -> torch.distributions.Distribution:
        return torch.distributions.Normal(loc=pred, scale=self.residual_var ** .5, **kwargs)

    @property
    def residual_var(self) -> torch.Tensor:
        std = self._residual_std_dev_log.exp()
        if torch.isclose(std, torch.zeros_like(std)):
            eps = 0.001
            warn(f"`{type(self)}().residual_var` near-zero, adding {eps ** 2}")
            std = std + eps
        return std ** 2

    def _get_loss(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  group_ids: np.ndarray,
                  weights: Optional[torch.Tensor],
                  cache: dict,
                  loss_type: Optional[str] = None,
                  **kwargs):
        if loss_type == 'mvnorm':
            return -self._get_mvnorm_log_prob(X=X, y=y, group_ids=group_ids, weights=weights, **kwargs)
        return super()._get_loss(
            X=X, y=y, group_ids=group_ids, cache=cache, loss_type=loss_type, weights=weights, **kwargs
        )

    def _get_default_loss_type(self) -> str:
        if len(self.rf_idx) == 1:
            return 'mvnorm'
        else:
            return 'iid'

    def _get_iid_log_probs(self,
                           pred: torch.Tensor,
                           actual: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        dist = self._forward_to_distribution(pred, validate_args=False)
        dist.scale = dist.scale / torch.sqrt(weights)
        log_probs = dist.log_prob(actual)
        return log_probs

    def _get_mvnorm_log_prob(self,
                             X: torch.Tensor,
                             y: torch.Tensor,
                             group_ids: np.ndarray,
                             weights: Optional[torch.Tensor],
                             **kwargs) -> torch.Tensor:
        extra = set(kwargs)
        if extra:
            warn(f"Ignoring {extra}")

        X, y = validate_tensors(X, y)
        group_ids = validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))
        if weights is None:
            weights = torch.ones_like(y)
        assert weights.shape == y.shape

        if len(self.grouping_factors) > 1:
            raise NotImplementedError
        # mercifully, if there is one grouping-factor, overall prob is just product of individual probs:
        gf = self.grouping_factors[0]

        # model-mat for raneffects:
        Z = torch.cat([torch.ones((len(X), 1)), X[:, self.rf_idx[gf]]], 1)

        Xs_by_r = defaultdict(list)
        ys_by_r = defaultdict(list)
        Zs_by_r = defaultdict(list)
        ws_by_r = defaultdict(list)
        for Z_g, X_g, y_g, w_g in chunk_grouped_data(Z, X[:, self.ff_idx], y, weights, group_ids=group_ids):
            Zs_by_r[len(Z_g)].append(Z_g)
            ys_by_r[len(Z_g)].append(y_g)
            Xs_by_r[len(Z_g)].append(X_g)
            ws_by_r[len(Z_g)].append(w_g)

        # compute log-prob per batch:
        re_dist = self.re_distribution(gf)
        log_probs = []
        for r, y_r in ys_by_r.items():
            ng = len(y_r)
            X_r = torch.stack(Xs_by_r[r])
            Z_r = torch.stack(Zs_by_r[r])
            w_r = torch.stack(ws_by_r[r])

            # counter-intuitively, it's faster (for `backward()`) if we call this one per batch here (vs. calling for
            # the entire dataset above)
            loc = self.fixed_effects_nn(X_r)
            if len(loc.shape) > 2:
                loc = loc.squeeze(-1)

            # mvnorm = LowRankMultivariateNormal(
            #     loc=loc,
            #     cov_diag=self.residual_var * torch.eye(r).expand(ng, -1, -1)/w_r.sqrt().unsqueeze(-1),
            #     cov_factor=Z_r,
            #     cov_factor_inner=re_dist.covariance_matrix.expand(ng, -1, -1),
            #     validate_args=False
            # )
            cov_r = Z_r @ re_dist.covariance_matrix.expand(ng, -1, -1) @ Z_r.permute(0, 2, 1)
            sigma_diag_r = self.residual_var / w_r
            if torch.isnan(cov_r).any() or torch.isinf(cov_r).any():
                raise ValueError("Nans/infs in parameters")
            mvnorm = _mvnorm_eps(loc, sigma_diag_r, cov_r)

            log_probs.append(mvnorm.log_prob(torch.stack(y_r)))

        return torch.cat(log_probs).sum()


def _mvnorm_eps(loc, sigma_diag, cov, start_eps: float = .000001) -> MultivariateNormal:
    mvnorm = None
    eps = start_eps
    while mvnorm is None:
        try:
            mvnorm = MultivariateNormal(
                loc=loc, covariance_matrix=torch.diag_embed(sigma_diag + eps) + cov, validate_args=True
            )
        except (ValueError, RuntimeError):
            eps *= 10
            if eps > .01:
                raise
    return mvnorm
