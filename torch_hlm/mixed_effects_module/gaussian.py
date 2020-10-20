from typing import Sequence, Union, Optional, Dict

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver
from .utils import ndarray_to_tensor


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
        assert not offset.requires_grad
        yoff = y - offset
        Xty_els = X * yoff[:, None]
        Xty = torch.zeros(num_groups, num_res).scatter_add(0, group_ids_broad, Xty_els)

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

    def get_loss(self,
                 predicted: torch.Tensor,
                 actual: torch.Tensor,
                 res_per_gf: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: this is h-likelihood, should be one but not only option
        actual = ndarray_to_tensor(actual)
        dist = torch.distributions.Normal(predicted, self.residual_std_dev)
        log_prob_lik = dist.log_prob(actual).sum()
        log_prob_prior = [torch.tensor(0.)]
        for gf, res in res_per_gf.items():
            log_prob_prior.append(self.re_distribution(gf).log_prob(res).sum())
        log_prob_prior = torch.stack(log_prob_prior)
        return (-log_prob_lik - log_prob_prior.sum()) / len(actual)
