import time
from tqdm.auto import tqdm

from typing import Optional, Sequence, Dict, Iterable, Tuple
from warnings import warn

import torch
import numpy as np
from scipy.stats import rankdata

from ..utils import chunk_grouped_data, validate_group_ids, validate_tensors
from ... import options


class ReSolver:
    """
    :param X: A model-matrix Tensor
    :param y: A response/target Tensor
    :param group_ids: A sequence which assigns each row in X to its group(s) -- e.g. a 2d ndarray or a list of
    tuples of group-labels. If there is only one grouping factor, this can be a 1d ndarray or a list of group-labels
    :param design: An ordered dictionary whose keys are grouping-factor labels and whose values are column-indices
    in `X` that are used for the random-effects of that grouping factor.
    :param prior_precisions: A dictionary with the precision matrices for each grouping factor. Should only be
    passed here if this solver is not being used in a context where the precision matrices are optimized. If the
    precision-matrices are being fed into an optimizer, then these should instead be passed to __call__
    """
    _warm_start_jitter = .01
    iterative: bool

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 weights: Optional[torch.Tensor],
                 design: dict,
                 slow_start: bool = True,
                 prior_precisions: Optional[dict] = None,
                 verbose: bool = False,
                 max_iter: Optional[int] = None,
                 min_iter: int = 5,
                 tol: float = .001):

        self.X, self.y = validate_tensors(X, y)
        if weights is None:
            weights = torch.ones_like(self.y)
        assert len(weights) == len(self.y)
        self.weights = weights
        self.group_ids = validate_group_ids(group_ids, num_grouping_factors=len(design))
        self.design = design
        self.prior_precisions = prior_precisions
        self.verbose = verbose

        if max_iter is None:
            max_iter = 200 * len(self.design)
        assert max_iter >= min_iter
        assert min_iter > 0
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.tol = tol

        self.slow_start = len(design) * slow_start * self.iterative

        self._warm_start = {}

        # establish static kwargs:
        self.static_kwargs_per_gf = {}
        for i, (gf, col_idx) in enumerate(self.design.items()):
            kwargs = {
                'X': torch.cat([torch.ones((len(self.X), 1)), self.X[:, col_idx]], 1),
                'y': self.y,
                'group_ids': self.group_ids[:, i],
                'weights': self.weights
            }

            kwargs['XtX'] = torch.stack([
                wg * Xg.t() @ Xg for Xg, wg in chunk_grouped_data(kwargs['X'], weights, group_ids=kwargs['group_ids'])
            ])

            if prior_precisions is not None:
                XtX = kwargs['XtX']
                pp = prior_precisions[gf].detach().expand(len(XtX), -1, -1)
                # TODO: use torch.cholesky -> cholesky_solve
                kwargs['Htild_inv'] = self._calculate_htild_inv(XtX, pp)

            self.static_kwargs_per_gf[gf] = kwargs

    def _calculate_htild_inv(self, XtX: torch.Tensor, pp: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _initialize_kwargs(self, prior_precisions: Optional[dict] = None) -> dict:
        """
        Called at the start of the solver's __call__, this XXX

        :param prior_precisions: XXX
        :return:
        """

        if prior_precisions is None:
            assert self.prior_precisions is not None
            # if we are not optimizing covariance, can avoid passing prior-precisions on each iter
            prior_precisions = self.prior_precisions

        kwargs_per_gf = {}
        for gf in self.design.keys():
            kwargs = kwargs_per_gf[gf] = self.static_kwargs_per_gf[gf].copy()
            kwargs['prior_precision'] = prior_precisions[gf]
            kwargs['iter_'] = 1

            if self._warm_start:
                # when this solver is used inside of MixedEffectsModule.fit_loop, we create an instance then
                # call it on each optimizer step. we re-use the solutions from the last step as a warm start
                with torch.no_grad():
                    kwargs['prev_res'] = self._warm_start['res_per_gf'][gf]
                    kwargs['prev_res'] += (self._warm_start_jitter * torch.randn_like(kwargs['prev_res']))
            else:
                kwargs['prev_res'] = None

            if 'Htild_inv' not in kwargs:
                # TODO: use torch.cholesky -> cholesky_solve
                # Htild_inv was not precomputed, compute it here
                XtX = kwargs['XtX']
                pp = prior_precisions[gf].expand(len(XtX), -1, -1)
                kwargs['Htild_inv'] = self._calculate_htild_inv(XtX, pp)

        return kwargs_per_gf

    def __call__(self,
                 fe_offset: torch.Tensor,
                 prior_precisions: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        kwargs_per_gf = self._initialize_kwargs(prior_precisions=prior_precisions)

        if self._warm_start:
            offsets = {k: v + self._warm_start_jitter * torch.randn_like(v)
                       for k, v in self._warm_start['prev_offsets'].items()}
            offsets['_fixed'] = fe_offset
        else:
            offsets = {'_fixed': fe_offset}

        prog = None
        if isinstance(self.verbose, str) and self.verbose.startswith('prog') and self.iterative:
            prog = tqdm(delay=20, total=self.max_iter)

        _cache = {gf: {} for gf in self.grouping_factors}
        changes_history = []
        history = []
        while True:
            history.append({})
            for gf in self.grouping_factors:
                gf_kwargs = kwargs_per_gf[gf]

                # recompute offset:
                gf_kwargs['offset'] = torch.sum(torch.stack([v for gf2, v in offsets.items() if gf2 != gf], 1), 1)

                # solve for gf:
                history[-1][gf] = gf_res = gf_kwargs['prev_res'] = self.solve_step(**gf_kwargs)
                if not self.iterative or len(self.design) == 1:
                    continue

                # recompute offset for other gfs:
                if 'group_idx' not in _cache[gf]:
                    _, _cache[gf]['group_idx'] = np.unique(gf_kwargs['group_ids'], return_inverse=True)
                offsets[gf] = (gf_kwargs['X'] * gf_res[_cache[gf]['group_idx']]).sum(1)

            # check convergence:
            if not self.iterative:
                break
            changes = dict(self._calculate_changes(history))
            changes_history.append(changes)
            if self._check_convergence(changes_history, iter_=len(history), prog=prog):
                break

            self._update_kwargs(kwargs_per_gf, changes_history)

        res_per_gf = history[-1]
        self._warm_start = {
            'res_per_gf': {gf: v.detach() for gf, v in res_per_gf.items()},
            'prev_offsets': {k: v.detach() for k, v in offsets.items() if k != '_fixed'}
        }

        return res_per_gf

    def _update_kwargs(self, kwargs_per_gf: dict, changes_history: Sequence[dict]) -> None:
        for gf, gf_kwargs in kwargs_per_gf.items():
            gf_kwargs['iter_'] += 1

        # if only a single grouping factor, can mask converged:
        if len(self.grouping_factors) == 1:
            gf = self.grouping_factors[0]
            gf_changes = changes_history[-1].get(gf)
            if gf_changes is not None:
                kwargs_per_gf[gf]['converged_mask'] = gf_changes <= self.tol

    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   group_ids: np.ndarray,
                   weights: torch.Tensor,
                   offset: torch.Tensor,
                   prior_precision: torch.Tensor,
                   Htild_inv: torch.Tensor,
                   prev_res: Optional[torch.Tensor],
                   iter_: int,
                   converged_mask: Optional[np.ndarray] = None,
                   **kwargs
                   ) -> torch.Tensor:
        """
        :param X: N*K model-mat
        :param y: N vector
        :param group_ids: N vector
        :param offset: N vector of offsets for y (e.g. predictions from fixed-effects).
        :param prior_precision: K*K precision matrix
        :param Htild_inv: [Approximate] hessian, inverted.
        :param prev_res: REs estimated from previous round
        :return: A G*K tensor of REs
        """
        num_groups = len(Htild_inv)
        num_obs, num_res = X.shape

        if prev_res is None:
            prev_res = .01 * torch.randn(num_groups, num_res)

        prior_precision = prior_precision.expand(num_groups, -1, -1)
        # we do this every call b/c we might have masked out converged groups
        group_ids_seq = torch.as_tensor(rankdata(group_ids, method='dense') - 1)

        if converged_mask is not None and converged_mask.any():
            out = prev_res.clone()
            converged_mask2 = converged_mask[group_ids_seq]
            out[~converged_mask] = self.solve_step(
                X=X[~converged_mask2],
                y=y[~converged_mask2],
                group_ids=group_ids[~converged_mask2],
                weights=weights[~converged_mask2],
                offset=offset[~converged_mask2],
                prior_precision=prior_precision[~converged_mask],
                Htild_inv=Htild_inv[~converged_mask],
                prev_res=prev_res[~converged_mask],
                iter_=iter_,
                **kwargs
            )
            return out

        assert y.shape == offset.shape, (y.shape, offset.shape)

        # predictions:
        prev_res_broad = prev_res[group_ids_seq]
        yhat = (X * prev_res_broad).sum(1) + offset
        mu = self.ilink(yhat)

        # grad:
        grad_els = self._calculate_grad(X, y, mu)

        if self.verbose:
            _num_clamped = (grad_els.abs() > options['re_solver_grad_clamp']).sum().item()
            if _num_clamped:
                print(f"Clamped grad for {_num_clamped:,} elements to abs<={options['re_solver_grad_clamp']}")
                # print(step[step.abs() > options['re_solver_grad_clamp']])
        grad_els = grad_els.clamp(-options['re_solver_grad_clamp'],
                                  options['re_solver_grad_clamp']) * weights.unsqueeze(-1)

        group_ids_broad = group_ids_seq.unsqueeze(-1).expand(-1, num_res)
        grad_no_pen = torch.zeros_like(prev_res).scatter_add(0, group_ids_broad, grad_els)
        grad = grad_no_pen - (prior_precision @ prev_res.unsqueeze(-1)).squeeze(-1)

        # step:
        step = self._calculate_step(
            X=X,
            group_ids_seq=group_ids_seq,
            mu=mu,
            weights=weights,
            grad=grad,
            Htild_inv=Htild_inv,
            prior_precision=prior_precision,
            **kwargs
        )

        step = step * iter_ / (iter_ + float(self.slow_start))

        return prev_res + step

    @staticmethod
    def ilink(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _calculate_grad(self,
                        X: torch.Tensor,
                        y: torch.Tensor,
                        mu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def grouping_factors(self) -> Sequence[str]:
        return list(self.design)

    def _calculate_step(self, grad: torch.Tensor, Htild_inv: torch.Tensor, **kwargs):
        return (Htild_inv @ grad.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _get_hessian(
            X: torch.Tensor,
            weights: torch.Tensor,
            mu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _check_convergence(self, changes_history: Sequence[dict], iter_: int, prog: Optional[tqdm]) -> bool:
        converged = iter_ >= self.min_iter
        _verbose = {}
        for gf, gf_changes in changes_history[-1].items():
            num_over = (gf_changes > self.tol).sum().item()
            if num_over:
                converged = False
            _verbose[gf] = num_over
        if prog is not None:
            prog.update()
        elif self.verbose:
            print(f"{type(self).__name__} - Iter {iter_} - Ngroups Changes>{self.tol} {_verbose}")

        if converged:
            return True
        elif iter_ >= self.max_iter:
            for gf, gf_changes in changes_history[-1].items():
                unconverged_mask = (gf_changes > self.tol)
                if unconverged_mask.any():
                    warn(f"There are {unconverged_mask.sum().item():,} unconverged "
                         f"groups for {gf}, max-relchange {gf_changes.max()}")
            return True
        return False

    @torch.no_grad()
    def _calculate_changes(self, history: Sequence[dict]) -> Iterable[Tuple[str, torch.Tensor]]:
        if len(history) < 2:
            return {}
        for gf in history[-1].keys():
            current_res = history[-1][gf]
            prev_res = history[-2][gf]
            abs_changes = torch.norm(current_res - prev_res, dim=1)
            if torch.isinf(abs_changes).any() or torch.isnan(abs_changes).any():
                raise RuntimeError(f"{type(self).__name__}: Optimization failed; nan/inf values")
            # rel_changes = abs_changes / prev_res.abs()
            yield gf, abs_changes
