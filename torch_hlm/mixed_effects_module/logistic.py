from typing import Sequence, Optional, Iterable, Union, Dict
from warnings import warn

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, ReSolver

from .utils import ndarray_to_tensor


class LogisticReSolver(ReSolver):
    prev_res_per_gf = None

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 design: dict,
                 prior_precisions: Optional[dict] = None):
        super().__init__(X=X, y=y, group_ids=group_ids, design=design, prior_precisions=prior_precisions)

        # precompute Htild_inv
        if prior_precisions is not None:
            for gf, kwargs in self.static_kwargs_per_gf.items():
                XtX = kwargs.pop('XtX')
                pp = prior_precisions[gf].detach().expand(len(XtX), -1, -1)
                kwargs['Htild_inv'] = torch.inverse(-XtX - pp)

    def _initialize_kwargs(self, fe_offset: torch.Tensor, prior_precisions: Optional[dict] = None, **kwargs) -> dict:
        kwargs_per_gf = super()._initialize_kwargs(fe_offset=fe_offset, prior_precisions=prior_precisions)
        if prior_precisions is not None:
            # Htild_inv was not precomputed, compute it here
            for gf, kwargs in kwargs_per_gf.items():
                XtX = kwargs.pop('XtX')
                pp = prior_precisions[gf].expand(len(XtX), -1, -1)
                kwargs['Htild_inv'] = torch.inverse(-XtX - pp)
        return kwargs_per_gf

    def __call__(self,
                 fe_offset: torch.Tensor,
                 max_iter: int = 200,
                 tol: float = .01,
                 **kwargs) -> Dict[str, torch.Tensor]:
        res_per_gf = super()(fe_offset=fe_offset, max_iter=max_iter, tol=tol, **kwargs)
        self.prev_res_per_gf = {k: v.detach() for k, v in res_per_gf.items()}
        return res_per_gf

    def _update_kwargs(self, kwargs_per_gf: dict, fe_offset: torch.Tensor, tol: float) -> dict:
        kwargs_per_gf = super()._update_kwargs(kwargs_per_gf=kwargs_per_gf, fe_offset=fe_offset, tol=tol)
        for gf, kwargs in kwargs_per_gf.items():
            if self.history:
                kwargs_per_gf['prev_res'] = self.history[-1][gf].detach()
            elif self.prev_res_per_gf:
                # TODO: jitter?
                kwargs_per_gf['prev_res'] = self.prev_res_per_gf[gf]
            else:
                kwargs_per_gf['prev_res'] = None
        return kwargs_per_gf

    # noinspection PyMethodOverriding
    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: Sequence,
                   prev_res: torch.Tensor,
                   prior_precision: torch.Tensor,
                   Htild_inv: torch.Tensor,
                   cg: Union[bool, str] = 'detach') -> torch.Tensor:
        """
        :param X: N*K model-mat
        :param y: N vector
        :param group_ids: N vector
        :param offset: N vector of offsets for y (e.g. predictions from fixed-effects).
        :param prior_precision: K*K precision matrix
        :param Htild_inv: Approximate hessian, inverted.
        :param cg: Use conjugate-gradient step? Default is to use, but detach the autograd gradient.
        :return: A G*K tensor of REs
        """
        _, num_res = X.shape
        num_groups = len(Htild_inv)

        if prev_res is None:
            prev_res = .01 * torch.randn(num_groups, num_res)

        prior_precision = prior_precision.expand(num_groups, -1, -1)
        group_ids_seq = rankdata(group_ids, method='dense') - 1

        group_ids_broad = torch.tensor(group_ids_seq, dtype=torch.int64).unsqueeze(-1).expand(-1, num_res)

        prev_betas_broad = prev_res[group_ids_seq]
        # predictions:
        yhat = ((X * prev_betas_broad).sum(1) + offset)
        prob = 1.0 / (1.0 + (-yhat).exp())

        # grad:
        grad_els = X * (prob - y).unsqueeze(-1).expand(-1, num_res)
        grad_no_pen = torch.zeros_like(prev_res).scatter_add(0, group_ids_broad, grad_els)
        grad = grad_no_pen + (prior_precision @ prev_res.unsqueeze(-1)).squeeze(-1)

        # step direction
        step_direction = (Htild_inv @ grad.unsqueeze(-1)).squeeze(-1)

        # step size:
        if cg:
            disable_grad = str(cg).lower().startswith('detach')
            with torch.set_grad_enabled(not disable_grad):
                numer = (grad * step_direction).sum(1)
                p1p = (prob * (1. - prob))
                Xstep = (X * step_direction[group_ids_seq]).sum(1)
                denom1 = torch.zeros_like(numer).scatter_add(0, group_ids_broad[:, 0], (p1p * Xstep ** 2))
                denom2 = \
                    ((prior_precision @ step_direction.unsqueeze(-1)).permute(0, 2, 1) @ step_direction.unsqueeze(-1))
                step_size = torch.zeros_like(numer)
                nz_grad_idx = np.where(numer != 0)  # when gradient is zero, we'll get 0./0. errors if we try to update
                step_size[nz_grad_idx] = numer[nz_grad_idx] / -(denom1[nz_grad_idx] + denom2[nz_grad_idx].squeeze())
                step_size = step_size.unsqueeze(-1)
        else:
            step_size = 1.

        # TODO: slow-start
        return prev_res + step_direction * step_size


class LogisticMixedEffectsModule(MixedEffectsModule):
    solver_cls = LogisticReSolver

    def get_loss(self,
                 predicted: torch.Tensor,
                 actual: torch.Tensor,
                 res_per_gf: Dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO: this likelihood is not appropriate for optimizing covariance
        actual = ndarray_to_tensor(actual)
        log_prob_lik = -torch.nn.BCEWithLogitsLoss(reduction='sum')(predicted, actual)
        log_prob_prior = [torch.tensor(0.)]
        for gf, res in res_per_gf.items():
            log_prob_prior.append(self.re_distribution(gf).log_prob(res).sum())
        log_prob_prior = torch.stack(log_prob_prior)
        return (-log_prob_lik - log_prob_prior.sum()) / len(actual)
