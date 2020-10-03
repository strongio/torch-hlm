from typing import Sequence, Optional, Iterable, Union
from warnings import warn

import torch
import numpy as np

from scipy.stats import rankdata

from .base import MixedEffectsModule, _Evaluator


class _LogisticEvaluator(_Evaluator):
    prev_res_per_group = None

    def __init__(self,
                 module: 'MixedEffectsModule',
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: Sequence,
                 optimizer: torch.optim.Optimizer,
                 warm_start_jitter: float,
                 **kwargs):
        self.warm_start_jitter = warm_start_jitter
        super().__init__(module=module, X=X, y=y, group_ids=group_ids, optimizer=optimizer, **kwargs)

    def _get_res_per_group(self, **kwargs) -> torch.Tensor:
        if self.prev_res_per_group is not None:
            kwargs['inits'] = self.prev_res_per_group
            kwargs['inits'] += self.warm_start_jitter * torch.randn_like(self.prev_res_per_group)
        res_per_group = super()._get_res_per_group(**kwargs)
        self.prev_res_per_group = res_per_group.detach()
        return res_per_group


class LogisticMixedEffectsModule(MixedEffectsModule):
    evaluator_cls = _LogisticEvaluator

    def fit_loop(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: Sequence,
                 optimizer: torch.optim.Optimizer,
                 warm_start_jitter: float = .01,
                 re_solve_kwargs: Optional[dict] = None,
                 **kwargs) -> Iterable[float]:

        # if any covariance parameter is being optimized, then we can't precompute Htild^-1
        precompute_htild_inv = True
        for grp in optimizer.param_groups:
            for p in grp:
                if self._is_cov_param(p):
                    precompute_htild_inv = False
                    break
        if self.verbose:
            print(f"Optimizer {'does not include' if precompute_htild_inv else 'includes'} covariance params")

        re_solve_kwargs = re_solve_kwargs or {}
        if 'verbose' not in re_solve_kwargs:
            re_solve_kwargs['verbose'] = self.verbose > 1
        return super().fit_loop(
            X=X,
            y=y,
            group_ids=group_ids,
            optimizer=optimizer,
            warm_start_jitter=warm_start_jitter,
            precompute_htild_inv=precompute_htild_inv,
            re_solve_kwargs=re_solve_kwargs
        )

    def _static_re_solve_kwargs(self, group_data: dict, precompute_htild_inv: bool = True) -> dict:
        kwargs = super()._static_re_solve_kwargs(group_data)
        if precompute_htild_inv:
            XtX = kwargs.pop('XtX')
            pp = self.re_distribution().precision_matrix.detach().expand(len(XtX), -1, -1)
            kwargs['Htild_inv'] = torch.inverse(-XtX - pp)
        return kwargs

    def _re_solve_kwargs(self, group_data: dict, static_kwargs: Optional[dict] = None) -> dict:
        kwargs = super()._re_solve_kwargs(group_data, static_kwargs=static_kwargs)
        if 'Htild_inv' not in static_kwargs:
            XtX = kwargs.pop('XtX')
            pp = self.re_distribution().precision_matrix.expand(len(XtX), -1, -1)
            kwargs['Htild_inv'] = torch.inverse(-XtX - pp)
        return kwargs

    def get_loss(self, predicted: torch.Tensor, actual: torch.Tensor, re_betas: torch.Tensor) -> torch.Tensor:
        # TODO: this is h-likelihood, should be one but not only option
        if isinstance(actual, np.ndarray):
            actual = torch.from_numpy(actual.astype('float32'))
        log_prob1 = -torch.nn.BCEWithLogitsLoss(reduction='sum')(predicted, actual)
        log_prob2 = self.re_distribution().log_prob(re_betas).sum()
        return (-log_prob1 - log_prob2) / len(actual)

    # noinspection PyMethodOverriding
    @staticmethod
    def _re_solve(X: torch.Tensor,
                  y: torch.Tensor,
                  group_ids: Sequence,
                  offset: torch.Tensor,
                  prior_precision: torch.Tensor,
                  Htild_inv: torch.Tensor,
                  inits: Optional[torch.Tensor] = None,
                  slow_start: Optional[int] = None,
                  tol: float = .01,
                  max_iter: int = 200,
                  verbose: bool = False,
                  cg: Union[bool, str] = 'detach',
                  return_history: bool = False,
                  **kwargs):
        """
        :param X: N*K model-mat
        :param y: N vector
        :param group_ids: N vector
        :param offset: N vector of fixed-effects predictions.
        :param prior_precision: G*K*K
        :param Htild_inv: Approximate hessian, inverted.
        :param inits: G*K, where rows correspond to sorted group_ids. If `None`, then use normal(0,.01) inits
        :param slow_start: Convergence is improved by setting `step *= (iter / (iter + slow_start)`. Default is 5 if
        no inits are passed else 2.
        :param tol: The tolerance for the *biggest* change (across groups) from iter-to-iter
        :param max_iter: Maximum iterations
        :param verbose: Print on each iteration?
        :param cg: Use conjugate-gradient step? Default is to use, but detach the autograd gradient.
        :param return_history: If `True`, then returns a I*G*K tensor with the history of the REs over iterations I.
        :return: A G*K tensor of REs (unless `return_history=True`)
        """

        group_ids = np.asanyarray(group_ids)
        group_ids_seq = rankdata(group_ids, method='dense') - 1
        num_obs, num_coefs = X.shape
        num_groups = len(Htild_inv)

        if slow_start is None:
            slow_start = 5 if inits is None else 2
        if inits is None:
            inits = .01 * torch.randn((num_groups, num_coefs))
        else:
            assert not inits.requires_grad
        if kwargs:
            warn(f"Unused kwargs:{set(kwargs)}")

        step_history = []
        res = inits.clone()

        already_below_tol = torch.zeros(num_groups, dtype=torch.bool)
        unconverged_gidx = np.arange(num_groups)
        unconverged_bidx = np.arange(num_obs)
        iter_ = 1
        while True:
            if iter_ > max_iter:
                # TODO: mask gradient for unconverged groups?
                if max_iter:
                    warn(
                        f"Reached {max_iter:,} iters without convergence ({num_pending:,} groups above tol of {tol})."
                    )
                break

            # calculate the step for any unconverged groups:
            step = torch.zeros_like(res)
            step[unconverged_gidx] = LogisticMixedEffectsModule._logistic_solve_step(
                X=X[unconverged_bidx],
                y=y[unconverged_bidx],
                Htild_inv=Htild_inv[unconverged_gidx],
                prior_precision=prior_precision,
                group_ids=group_ids[unconverged_bidx],
                offset=offset[unconverged_bidx],
                prev_res=res[unconverged_gidx].detach(),
                cg=cg
            )

            # check if we've converged below tol for two iterations:
            with torch.no_grad():
                if (step != step).any():
                    raise RuntimeError("`nans` in re_solve")
                changes = torch.norm(step, dim=1) / torch.norm(res, dim=1)
                pending = (changes > tol) | ~already_below_tol  # patience
                already_below_tol = (changes < tol)
                num_pending = pending.sum().item()
                unconverged_gidx = np.where(pending)[0]
                unconverged_bidx = np.where(pending[group_ids_seq])[0]

            if verbose:
                print(f"Iter {iter_}: {num_pending:,} groups above tol ({tol}), max-change {changes.max().item()}")

            # calculate res for next iter:
            step = step * (iter_ / (float(slow_start) + iter_))
            step_history.append(step)
            res = inits.clone() + torch.sum(torch.stack(step_history, 0), 0)

            if not num_pending:
                break

            iter_ += 1

        if return_history:
            return inits[None, :, :] + torch.stack(step_history, 0)
        else:
            return res

    @staticmethod
    def _logistic_solve_step(X: torch.Tensor,
                             y: torch.Tensor,
                             Htild_inv: torch.Tensor,
                             prior_precision: torch.Tensor,
                             group_ids: Sequence,
                             offset: torch.Tensor,
                             prev_res: torch.Tensor,
                             cg: Union[bool, str]) -> torch.Tensor:
        prior_precision = prior_precision.expand(len(Htild_inv), -1, -1)
        group_ids_seq = rankdata(group_ids, method='dense') - 1
        _, num_betas = X.shape
        group_ids_broad = torch.tensor(group_ids_seq, dtype=torch.int64).unsqueeze(-1).expand(-1, num_betas)

        prev_betas_broad = prev_res[group_ids_seq]
        # predictions:
        yhat = ((X * prev_betas_broad).sum(1) + offset)
        prob = 1.0 / (1.0 + (-yhat).exp())

        # grad:
        grad_els = X * (prob - y).unsqueeze(-1).expand(-1, num_betas)
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

        #
        return step_direction * step_size
