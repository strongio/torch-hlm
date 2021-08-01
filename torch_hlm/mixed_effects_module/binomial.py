from typing import Sequence, Optional, Union, Dict, Tuple

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver

from .utils import validate_1d_like, validate_tensors, validate_group_ids, get_yhat_r


class BinomialReSolver(ReSolver):
    iterative = True

    def __init__(self, *args, cg: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if cg is None:
            cg = len(self.design) == 1
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

        if self.cg:
            numer = (grad * step).sum(1)
            var = (mu * (1. - mu)) / weights
            Xstep = (X * step[group_ids_seq]).sum(1)
            denom1 = torch.zeros_like(numer).scatter_add(0, torch.tensor(group_ids_seq), (var * Xstep ** 2))
            denom2 = ((prior_precision @ step.unsqueeze(-1)).permute(0, 2, 1) @ step.unsqueeze(-1))
            step_size = torch.zeros_like(numer)
            nz_grad_idx = np.where(numer != 0)  # when gradient is zero, we'll get 0./0. errors if we try to update
            step_size[nz_grad_idx] = numer[nz_grad_idx] / (denom1[nz_grad_idx] + denom2[nz_grad_idx].squeeze())
            step_size = step_size.unsqueeze(-1)
        else:
            step_size = .25
        return step_size * step


class BinomialMixedEffectsModule(MixedEffectsModule):
    solver_cls = BinomialReSolver
    default_loss_type = 'iid'  # TODO

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
        return torch.distributions.Binomial(logits=pred, **kwargs)
