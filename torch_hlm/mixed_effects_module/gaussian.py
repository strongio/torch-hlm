from collections import defaultdict
from typing import Sequence, Union, Optional, Dict

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver
from .utils import ndarray_to_tensor, chunk_grouped_data, validate_1d_like
from torch.distributions import MultivariateNormal


class GaussianReSolver(ReSolver):
    # noinspection PyMethodOverriding
    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: np.ndarray,
                   XtX: torch.Tensor,
                   prior_precision: torch.Tensor) -> torch.Tensor:
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

        yoff = y - offset
        Xty_els = X * yoff[:, None]
        Xty = torch.zeros(num_groups, num_res).scatter_add(0, group_ids_broad, Xty_els)

        # res = torch.inverse(XtX + prior_precision) @ Xty.unsqueeze(-1)
        res, _ = torch.solve(Xty.unsqueeze(-1), XtX + prior_precision)

        return res.squeeze(-1)

    def _check_convergence(self, tol: float) -> bool:
        if len(self.design) == 1:
            # if only one grouping factor, then solution is closed form, not iterative
            return True
        return super()._check_convergence(tol=tol)


class GaussianMixedEffectsModule(MixedEffectsModule):
    solver_cls = GaussianReSolver

    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Dict[str, Union[int, Sequence]],
                 fixed_effects_nn: Optional[torch.nn.Module] = None,
                 verbose: int = 1):
        super().__init__(
            fixeff_features=fixeff_features,
            raneff_features=raneff_features,
            fixed_effects_nn=fixed_effects_nn,
            verbose=verbose
        )
        self._residual_std_dev_log = torch.nn.Parameter(.01 * torch.randn(1))

    @property
    def residual_std_dev(self) -> torch.Tensor:
        return self._residual_std_dev_log.exp()

    def get_loss(self, X: torch.Tensor, y: torch.Tensor, group_ids: np.ndarray, **kwargs) -> torch.Tensor:
        if len(self.grouping_factors) > 1:
            raise NotImplementedError("`get_loss` for multiple grouping-factors not currently implemented.")
        gf = self.grouping_factors[0]

        # yhat for fixed effects:
        yhat_f = validate_1d_like(self.fixed_effects_nn(X[:, self.ff_idx]))

        # model-mat for raneffects:
        col_idx = self.rf_idx[gf]
        Z = torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1)

        # for computational efficiency, we'll group by rank so we can batch MultivariateNormal's __init__ and log_prob
        ys_by_rank = defaultdict(list)
        locs_by_rank = defaultdict(list)
        covs_by_rank = defaultdict(list)
        cov = self.re_distribution(gf).covariance_matrix
        for Z_g, yhat_f_g, y_g in chunk_grouped_data(Z, yhat_f, y, group_ids=group_ids):
            cov_g = Z_g @ cov @ Z_g.t()
            rank_g = len(cov_g)
            locs_by_rank[rank_g].append(yhat_f_g)
            covs_by_rank[rank_g].append(self.residual_std_dev * torch.eye(rank_g) + cov_g)
            ys_by_rank[rank_g].append(y_g)

        # mercifully just the product for the single-group case:
        log_probs = []
        for rank_g, y_gs in ys_by_rank.items():
            mvnorm = MultivariateNormal(
                loc=torch.stack(locs_by_rank[rank_g]), covariance_matrix=torch.stack(covs_by_rank[rank_g])
            )
            log_probs.append(mvnorm.log_prob(torch.stack(y_gs)))
        # make it so that the log-prob is roughly of the same magnitude regardless of the nobs-per-group:
        return -torch.cat(log_probs).mean() / np.median(list(ys_by_rank.keys()))
