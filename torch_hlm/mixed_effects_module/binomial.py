from typing import Sequence, Optional, Union, Tuple

import torch
import numpy as np

from .base import MixedEffectsModule, ReSolver


class BinomialReSolver(ReSolver):
    iterative = True

    def __init__(self, *args, cg: Optional[bool] = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg = cg

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
                        **kwargs):

        step = super(BinomialReSolver, self)._calculate_step(grad=grad, Htild_inv=Htild_inv)

        var = (mu * (1. - mu)) * weights
        if self.cg:
            numer = (grad * step).sum(1)
            Xstep = (X * step[group_ids_seq]).sum(1)
            denom1 = torch.zeros_like(numer).scatter_add(0, group_ids_seq, (var * Xstep ** 2))
            denom2 = ((prior_precision @ step.unsqueeze(-1)).permute(0, 2, 1) @ step.unsqueeze(-1)).squeeze(-1)
            step_size = torch.zeros_like(numer)
            nz_grad_idx = np.where(numer != 0)  # when gradient is zero, we'll get 0./0. errors if we try to update
            step_size[nz_grad_idx] = numer[nz_grad_idx] / (denom1[nz_grad_idx] + denom2[nz_grad_idx].squeeze(-1))
            step_size = step_size.unsqueeze(-1)
        else:
            var_sums = torch.zeros(len(prior_precision)).scatter_add(0, group_ids_seq, var)
            # TODO: memoize
            group_counts = torch.zeros(len(prior_precision)).scatter_add(0, group_ids_seq, torch.ones_like(var))
            step_size = (var_sums / group_counts).unsqueeze(-1)
        return step_size * step

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
        dist = self._forward_to_distribution(pred, total_count=weights, validate_args=False)
        log_probs = dist.log_prob(actual * weights)
        return log_probs
