from random import shuffle
from typing import Optional, Sequence, Dict, Iterable, Tuple
from warnings import warn

import torch
import numpy as np

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
    _prev_res_per_gf = None
    _prev_re_offsets = None

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 design: dict,
                 prior_precisions: Optional[dict] = None,
                 slow_start: Optional[float] = None,
                 verbose: bool = False):

        self.X, self.y = validate_tensors(X, y)
        self.group_ids = validate_group_ids(group_ids, num_grouping_factors=len(design))
        self.design = design
        self.prior_precisions = prior_precisions
        self.verbose = verbose

        if slow_start is None:
            slow_start = 2 * (len(self.design) - 1)
        self.slow_start = slow_start

        # establish static kwargs:
        self.static_kwargs_per_gf = {}
        for i, (gp, col_idx) in enumerate(self.design.items()):
            kwargs = {
                'X': torch.cat([torch.ones((len(X), 1)), X[:, col_idx]], 1),
                'y': y,
                'group_ids': group_ids[:, i]
            }
            # TODO: sample_weights
            kwargs['XtX'] = torch.stack([
                Xg.t() @ Xg for Xg, in chunk_grouped_data(kwargs['X'], group_ids=kwargs['group_ids'])
            ])
            self.static_kwargs_per_gf[gp] = kwargs

    def __call__(self,
                 fe_offset: torch.Tensor,
                 max_iter: Optional[int] = None,
                 min_iter: int = 5,
                 reltol: float = .01,
                 prior_precisions: Optional[dict] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Solve the random effects.

        :param fe_offset: The offset that comes from the fixed-effects.
        :param max_iter: The maximum number of iterations.
        :param reltol: Relative (change/prev_val) tolerance for checking convergence.
        :param prior_precisions: A dictionary with the precision matrices for each grouping factor.
        :param kwargs: Other keyword arguments to `solve_step`
        :return: A dictionary with random-effects for each grouping-factor
        """
        if max_iter is None:
            max_iter = 200 * len(self.design)
        assert max_iter > 0
        kwargs_per_gf = self._initialize_kwargs(fe_offset=fe_offset, prior_precisions=prior_precisions)

        self.history = []
        while True:
            if len(self.history) >= max_iter:
                if max_iter:
                    for gf, changes in self._check_convergence():
                        unconverged = (changes > reltol).sum()
                        if unconverged:
                            warn(f"There are {unconverged:,} unconverged for {gf}, max-relchange {changes.max()}")
                break

            # take a step towards solving the random-effects:
            self.history.append({})
            for gf in self._shuffled_gfs():
                self.history[-1][gf] = self.solve_step(**kwargs_per_gf[gf], **kwargs)
            # print(self.history[-1])

            # check convergence:
            if self._check_if_converged(reltol, min_iter):
                break
                # TODO: if only one grouping factor, drop converged groups

            # recompute offset, etc:
            kwargs_per_gf = self._update_kwargs(kwargs_per_gf, fe_offset=fe_offset)

        res_per_gf = self.history[-1]
        self._prev_res_per_gf = {k: v.detach() for k, v in res_per_gf.items()}
        self._prev_re_offsets = {
            k: v[1:].detach().sum(0) for k, v in self._recompute_offsets(fe_offset, res_per_gf).items()
        }

        # print(len(self.history))
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
                   prev_res: Optional[torch.Tensor],
                   **kwargs) -> torch.Tensor:
        """
        Update random-effects. For some solvers this might not be iterative given a single grouping factor
        because a closed-form solution exists (e.g. gaussian); however, this is still an iterative `step` in the case of
         multiple grouping-factors.
        """
        raise NotImplementedError

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

            if self._prev_res_per_gf is not None:
                # when this solver is used inside of MixedEffectsModule.fit_loop, we create an instance then
                # call it on each optimizer step. we re-use the solutions from the last step as a warm start
                kwargs['prev_res'] = self._prev_res_per_gf[gf]
                kwargs['offset'] = kwargs['offset'] + self._prev_re_offsets[gf]
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
    def _check_if_converged(self, reltol: float, min_iter: int) -> bool:
        if len(self.history) < 2:
            return False
        converged = len(self.history) >= min_iter
        _verbose = {}
        for gf, changes in self._check_convergence():
            # TODO: patience
            if torch.isinf(changes).any() or torch.isnan(changes).any():
                raise RuntimeError("Convergence failed; nan/inf values")
            if (changes > reltol).any():
                converged = False
                if not self.verbose:
                    break
            if self.verbose:
                _verbose[gf] = changes.max().item()
        if self.verbose:
            print(f"{type(self).__name__} - Iter {len(self.history)} - Max. Relchanges {_verbose}")
        return converged

    def _check_convergence(self) -> Iterable[Tuple[str, torch.Tensor]]:
        assert len(self.history) >= 2
        with torch.no_grad():
            for gf in self.design.keys():
                current_res = self.history[-1][gf]
                prev_res = self.history[-2][gf]
                changes = torch.norm(current_res - prev_res, dim=1) / torch.norm(prev_res, dim=1)
                yield gf, changes
