from collections import defaultdict
from typing import Sequence, Union, Optional, Dict

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver
from .utils import chunk_grouped_data
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
                 covariance_parameterization: str = 'log_cholesky',
                 re_scale_penalty: Union[float, dict] = 0.,
                 verbose: int = 1):
        super().__init__(
            fixeff_features=fixeff_features,
            raneff_features=raneff_features,
            fixed_effects_nn=fixed_effects_nn,
            covariance_parameterization=covariance_parameterization,
            re_scale_penalty=re_scale_penalty,
            verbose=verbose
        )
        self._residual_std_dev_log = torch.nn.Parameter(.01 * torch.randn(1))

    @property
    def residual_var(self) -> torch.Tensor:
        return self._residual_std_dev_log.exp() ** 2

    def get_loss(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 res_per_gf: dict = None,
                 reduce: str = 'nobs') -> torch.Tensor:
        X, y = self._validate_tensors(X, y)
        group_ids = self._validate_group_ids(group_ids, num_grouping_factors=len(self.grouping_factors))
        if len(self.grouping_factors) > 1:
            raise NotImplementedError("`get_loss` for multiple grouping-factors not currently implemented.")
        gf = self.grouping_factors[0]

        # model-mat for raneffects:
        col_idx = self.rf_idx[gf]
        Z = torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1)

        # for computational efficiency, we'll group by nobs so we can batch MultivariateNormal's __init__ and log_prob
        Xs_by_r = defaultdict(list)
        ys_by_r = defaultdict(list)
        Zs_by_r = defaultdict(list)
        for Z_g, X_g, y_g in chunk_grouped_data(Z, X[:, self.ff_idx], y, group_ids=group_ids):
            Zs_by_r[len(Z_g)].append(Z_g)
            ys_by_r[len(Z_g)].append(y_g)
            Xs_by_r[len(Z_g)].append(X_g)

        # compute log-prob per batch:
        log_probs = []
        for r, y_r in ys_by_r.items():
            ng = len(y_r)
            X_r = torch.stack(Xs_by_r[r])
            Z_r = torch.stack(Zs_by_r[r])

            # cov:
            cov_r = Z_r @ self.re_distribution(gf).covariance_matrix.expand(ng, -1, -1) @ Z_r.permute(0, 2, 1)
            eps_r = (self.residual_var * torch.eye(r)).expand(ng, -1, -1)

            # counter-intuitively, it's faster (for `backward()`) if we call this one per batch here (vs. calling for
            # the entire dataset above)
            loc = self.fixed_effects_nn(X_r)
            if len(loc.shape) > 2:
                loc = loc.squeeze(-1)

            # mercifully, if there is one grouping-factor, overall prob is just product of individual probs:
            try:
                mvnorm = MultivariateNormal(loc=loc, covariance_matrix=eps_r + cov_r)
                log_probs.append(mvnorm.log_prob(torch.stack(y_r)))
            except RuntimeError as e:
                if 'chol' not in str(e):
                    raise e
                raise RuntimeError(
                    "Unable to evaluate log-prob. Things to try:"
                    "\n- Center/scale predictors"
                    "\n- Check for extreme-values in the response-variable"
                    "\n- Decrease the learning-rate"
                    "\n- Set `re_scale_penalty`"
                ) from e

        loss = -torch.cat(log_probs).sum()
        loss = loss + self._get_re_penalty()
        if reduce:
            loss = loss / len(X)
        return loss
