from random import shuffle
from typing import Optional, Sequence, Dict, Iterable, Tuple, Callable
from warnings import warn

import torch
import numpy as np
from scipy.stats import rankdata

from ..utils import chunk_grouped_data, validate_group_ids, validate_tensors, get_yhat_r


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
    iterative: bool

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 design: dict,
                 slow_start: bool = True,
                 prior_precisions: Optional[dict] = None,
                 verbose: bool = False,
                 max_iter: Optional[int] = None,
                 min_iter: int = 5,
                 reltol: float = .01):

        self.X, self.y = validate_tensors(X, y)
        self.group_ids = validate_group_ids(group_ids, num_grouping_factors=len(design))
        self.design = design
        self.prior_precisions = prior_precisions
        self.verbose = verbose

        if max_iter is None:
            max_iter = 200 * len(self.design)
        assert max_iter > 0
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.reltol = reltol

        self.slow_start = len(design) * slow_start * self.iterative

        self._warm_start = {}

        # establish static kwargs:
        self.static_kwargs_per_gf = {}
        for i, (gf, col_idx) in enumerate(self.design.items()):
            kwargs = {
                'X': torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1),
                'y': y,
                'group_ids': group_ids[:, i]
            }
            # TODO: sample_weights
            kwargs['XtX'] = torch.stack([
                Xg.t() @ Xg for Xg, in chunk_grouped_data(kwargs['X'], group_ids=kwargs['group_ids'])
            ])

            if prior_precisions is not None:
                XtX = kwargs['XtX']
                pp = prior_precisions[gf].detach().expand(len(XtX), -1, -1)
                # TODO: avoid inverse
                kwargs['Htild_inv'] = torch.inverse(XtX + pp)

            self.static_kwargs_per_gf[gf] = kwargs

    def __call__(self,
                 fe_offset: torch.Tensor,
                 prior_precisions: Optional[dict] = None) -> Dict[str, torch.Tensor]:
        """
        Solve the random effects.

        :param fe_offset: The offset that comes from the fixed-effects.
        :param max_iter: The maximum number of iterations.
        :param reltol: Relative (change/prev_val) tolerance for checking convergence.
        :param prior_precisions: A dictionary with the precision matrices for each grouping factor.
        :param kwargs: Other keyword arguments to `solve_step`
        :return: A dictionary with random-effects for each grouping-factor
        """

        kwargs_per_gf = self._initialize_kwargs(fe_offset=fe_offset, prior_precisions=prior_precisions)

        self.history = []
        while True:
            # take a step towards solving the random-effects:
            self.history.append({})
            for gf in self._shuffled_gfs():
                self.history[-1][gf] = self.solve_step(**kwargs_per_gf[gf])

            # check convergence:
            if not self.iterative:
                break

            if len(self.history) > 1:
                convergence = dict(self._check_convergence())
                converged = len(self.history) >= self.min_iter
                _verbose = {}
                for gf, changes in convergence.items():
                    # TODO: patience
                    if (changes > self.reltol).any():
                        converged = False
                    _verbose[gf] = changes.max().item()
                if self.verbose:
                    print(f"{type(self).__name__} - Iter {len(self.history)} - Max. Relchanges {_verbose}")
                if converged:
                    break

            # recompute offset, etc:
            kwargs_per_gf = self._update_kwargs(kwargs_per_gf, fe_offset=fe_offset)

            if len(self.history) >= self.max_iter:
                for gf, changes in convergence.items():
                    unconverged = (changes > self.reltol).sum()
                    if unconverged:
                        warn(f"There are {unconverged:,} unconverged for {gf}, max-relchange {changes.max()}")
                break

        res_per_gf = self.history[-1]
        self._warm_start = {
            'res_per_gf': {k: v.detach() for k, v in res_per_gf.items()},
            're_offsets': {k: v[1:].detach().sum(0) for k, v in self._recompute_offsets(fe_offset, res_per_gf).items()}
        }

        return res_per_gf

    def _shuffled_gfs(self) -> Sequence[str]:
        gfs = list(self.design.keys())
        shuffle(gfs)
        return gfs

    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: Sequence,
                   prior_precision: torch.Tensor,
                   Htild_inv: torch.Tensor,
                   prev_res: Optional[torch.Tensor],
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
        group_ids_seq = rankdata(group_ids, method='dense') - 1

        assert y.shape == offset.shape, (y.shape, offset.shape)

        # predictions:
        prev_res_broad = prev_res[group_ids_seq]
        yhat = (X * prev_res_broad).sum(1) + offset
        mu = self.ilink(yhat)

        # grad:
        group_ids_broad = torch.tensor(group_ids_seq, dtype=torch.int64).unsqueeze(-1).expand(-1, num_res)
        grad_els = self.calculate_grad(X, y, mu)  # X * (y - mu).unsqueeze(-1).expand(-1, num_res)
        grad_no_pen = torch.zeros_like(prev_res).scatter_add(0, group_ids_broad, grad_els)
        grad = grad_no_pen - (prior_precision @ prev_res.unsqueeze(-1)).squeeze(-1)

        # step:
        step = self._calculate_step(
            X=X, group_ids_seq=group_ids_seq, mu=mu, grad=grad, Htild_inv=Htild_inv, prior_precision=prior_precision
        )
        iter_ = len(self.history) + 1
        step = step * (iter_ / (float(self.slow_start) + iter_))

        return prev_res + step

    @staticmethod
    def ilink(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def calculate_grad(self,
                       X: torch.Tensor,
                       y: torch.Tensor,
                       mu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _calculate_step(self,
                        X: torch.Tensor,
                        group_ids_seq: np.ndarray,
                        mu: torch.Tensor,
                        grad: torch.Tensor,
                        Htild_inv: torch.Tensor,
                        prior_precision: torch.Tensor):
        return (Htild_inv @ grad.unsqueeze(-1)).squeeze(-1)

    def _initialize_kwargs(self, fe_offset: torch.Tensor, prior_precisions: Optional[dict] = None) -> dict:
        """
        Called at the start of the solver's __call__, this XXX

        :param fe_offset: XXX
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
            kwargs['prev_res'] = None
            kwargs['offset'] = fe_offset

            if self._warm_start:
                # when this solver is used inside of MixedEffectsModule.fit_loop, we create an instance then
                # call it on each optimizer step. we re-use the solutions from the last step as a warm start
                with torch.no_grad():
                    kwargs['prev_res'] = self._warm_start['res_per_gf'][gf]
                    kwargs['prev_res'] += (.01 * torch.randn_like(kwargs['prev_res']))
                    kwargs['offset'] = kwargs['offset'] + self._warm_start['re_offsets'][gf]
                    kwargs['offset'] += (.01 * torch.randn_like(kwargs['offset']))

            if prior_precisions is not None:
                # TODO: use solve
                # Htild_inv was not precomputed, compute it here
                XtX = kwargs['XtX']
                pp = prior_precisions[gf].expand(len(XtX), -1, -1)
                kwargs['Htild_inv'] = torch.inverse(XtX + pp)

        return kwargs_per_gf

    def _update_kwargs(self, kwargs_per_gf: dict, fe_offset: torch.Tensor) -> Optional[dict]:
        """
        Called on each iteration of the solver's __call__, this method recomputes the offset

        :param kwargs_per_gf: XXX
        :param fe_offset: XXX
        :return:
        """
        assert self.history
        res_per_gf = self.history[-1]
        with torch.no_grad():  # TODO: in testing, `no_grad` was critical; unclear why
            new_offsets = self._recompute_offsets(fe_offset, res_per_gf)
        out = {}
        for gf in kwargs_per_gf.keys():
            out[gf] = kwargs_per_gf[gf].copy()
            out[gf]['offset'] = new_offsets[gf].sum(0)
            out[gf]['prev_res'] = res_per_gf[gf]
        return out

    def _recompute_offsets(self,
                           fe_offset: torch.Tensor,
                           res_per_gf: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        res_per_gf = {k: res_per_gf[k] for k in self.design}  # ordering matters b/c of enumerate below
        yhat_r = get_yhat_r(design=self.design, X=self.X, group_ids=self.group_ids, res_per_gf=res_per_gf)
        out = {}
        for gf1 in self.design.keys():
            out[gf1] = [fe_offset]
            for i, gf2 in enumerate(self.design.keys()):
                if gf1 == gf2:
                    continue
                out[gf1].append(yhat_r[:, i])
            out[gf1] = torch.stack(out[gf1])
        return out

    @torch.no_grad()
    def _check_convergence(self) -> Iterable[Tuple[str, torch.Tensor]]:
        assert len(self.history) > 1
        for gf in self.design.keys():
            current_res = self.history[-1][gf]
            prev_res = self.history[-2][gf]
            changes = torch.norm(current_res - prev_res, dim=1) / torch.norm(prev_res, dim=1)
            if torch.isinf(changes).any() or torch.isnan(changes).any():
                raise RuntimeError("Convergence failed; nan/inf values")
            yield gf, changes
