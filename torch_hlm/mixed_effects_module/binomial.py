from scipy.stats import rankdata
from typing import Sequence, Optional

import torch
import numpy as np

from warnings import warn

from .base import MixedEffectsModule, ReSolver


class BinomialReSolver(ReSolver):
    iterative = True

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 weights: Optional[torch.Tensor],
                 design: dict,
                 cg: bool = True,
                 max_iter: Optional[int] = None,
                 **kwargs):
        self.cg = cg
        if max_iter is None:
            max_iter = 500 * len(design)
        super().__init__(X=X, y=y, group_ids=group_ids, weights=weights, design=design, max_iter=max_iter, **kwargs)
        if self.cg:
            self.slow_start *= 5

    def _calculate_htild_inv(self, XtX: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
        return torch.inverse(.25 * XtX + pp)

    @staticmethod
    def ilink(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _calculate_grad(self,
                        X: torch.Tensor,
                        y: torch.Tensor,
                        mu: torch.Tensor) -> torch.Tensor:
        _, num_res = X.shape
        return X * (y - mu).unsqueeze(-1).expand(-1, num_res)

    def _calculate_step(self,
                        grad: torch.Tensor,
                        Htild_inv: torch.Tensor,
                        X: torch.Tensor = None,
                        group_ids_seq: torch.Tensor = None,
                        mu: torch.Tensor = None,
                        weights: torch.Tensor = None,
                        prior_precision: torch.Tensor = None,
                        cg: Optional[torch.Tensor] = None,
                        **kwargs):

        step = super(BinomialReSolver, self)._calculate_step(grad=grad, Htild_inv=Htild_inv)
        if self.cg and (cg is None or cg.any()):
            ngroups = len(step)
            if cg is None:
                cg = torch.ones(ngroups, dtype=torch.bool)
            cg_mask2 = cg[group_ids_seq]
            step_size = torch.ones((ngroups, 1))
            step_size[cg] = self._cg_step_size(
                grad=grad[cg],
                step=step[cg],
                X=X[cg_mask2],
                group_ids_seq=torch.as_tensor(rankdata(group_ids_seq[cg_mask2], method='dense') - 1),
                mu=mu[cg_mask2],
                weights=weights[cg_mask2],
                prior_precision=prior_precision[cg]
            ).unsqueeze(-1)
        else:
            step_size = 1
        return step * step_size

    def _cg_step_size(self,
                      grad: torch.Tensor,
                      step: torch.Tensor,
                      X: torch.Tensor = None,
                      group_ids_seq: torch.Tensor = None,
                      mu: torch.Tensor = None,
                      weights: torch.Tensor = None,
                      prior_precision: torch.Tensor = None):
        # g.T @ u
        numer = (grad * step).sum(1)
        nz_grad_idx = np.where(numer != 0)  # when gradient is zero, we'll get 0./0. errors if we try to update

        # u.T @ pp @ u
        denom1 = ((prior_precision @ step.unsqueeze(-1)).permute(0, 2, 1) @ step.unsqueeze(-1))
        denom1 = denom1.squeeze(-1).squeeze(-1)

        # sum[ var[i]*(u.T @ x[i])**2 ]
        var = (mu * (1. - mu))
        Xstep = (X * step[group_ids_seq]).sum(1)
        denom2 = torch.zeros_like(numer).scatter_add(0, group_ids_seq, var * weights * (Xstep ** 2))
        denom = (denom1[nz_grad_idx] + denom2[nz_grad_idx].squeeze(-1))

        # hessian = torch.stack([self._get_hessian(Xg,wg,mug) + prior_precision[0] for Xg, wg, mug in
        #                        chunk_grouped_data(X, weights, mu, group_ids=group_ids_seq)])
        # stepu = step.unsqueeze(-1)
        # denom = (stepu.permute(0, 2, 1) @ hessian @ stepu).squeeze(-1).squeeze(-1)[nz_grad_idx]

        # cj:
        step_size = torch.zeros_like(numer)
        step_size[nz_grad_idx] = numer[nz_grad_idx] / denom
        return step_size

    @staticmethod
    def _get_hessian(
            X: torch.Tensor,
            weights: torch.Tensor,
            mu: torch.Tensor) -> torch.Tensor:
        var = (mu * (1. - mu))
        return var * weights * X.t() @ X


class BinomialMixedEffectsModule(MixedEffectsModule):
    solver_cls = BinomialReSolver

    def _get_default_loss_type(self) -> str:
        if len(self.grouping_factors) == 1:
            return 'mc'
        else:
            return 'iid'

    def _forward_to_distribution(self, pred: torch.Tensor, **kwargs) -> torch.distributions.Distribution:
        return torch.distributions.Binomial(logits=pred, **kwargs)

    def _get_iid_log_probs(self,
                           pred: torch.Tensor,
                           actual: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
        if not torch.logical_and(0 <= actual, 1 >= actual).all():
            raise ValueError("expected Binomial values to be 0-1, did you pass counts instead of proportions?")
        weights_gr1 = weights > 1
        if weights_gr1.any():
            if torch.logical_or(actual[weights_gr1] == 0, actual[weights_gr1] == 1).all():
                warn("All values are exactly 0 or 1; did you pass counts instead of probabilities?")
        dist = self._forward_to_distribution(pred, total_count=weights, validate_args=False)
        log_probs = dist.log_prob(actual * weights)
        return log_probs
