from typing import Optional, Sequence, Dict
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

    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor,
                 group_ids: np.ndarray,
                 design: dict,
                 prior_precisions: Optional[dict] = None):

        self.X, self.y = validate_tensors(X, y)
        self.group_ids = validate_group_ids(group_ids, num_grouping_factors=len(design))
        self.design = design
        self.prior_precisions = prior_precisions

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
                 max_iter: int = 200,
                 tol: float = .01,
                 prior_precisions: Optional[dict] = None,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Solve the random effects.

        :param fe_offset: The offset that comes from the fixed-effects.
        :param max_iter: The maximum number of iterations.
        :param tol: Tolerance for checking convergence.
        :param prior_precisions: A dictionary with the precision matrices for each grouping factor.
        :param kwargs: Other keyword arguments to `solve_step`
        :return: A dictionary with random-effects for each grouping-factor
        """
        assert max_iter > 0
        kwargs_per_gf = self._initialize_kwargs(fe_offset=fe_offset, prior_precisions=prior_precisions)

        self.history = []
        while True:
            if len(self.history) >= max_iter:
                # TODO: mask gradient for unconverged groups?
                if max_iter:
                    warn("TODO")
                break

            # take a step towards solving the random-effects:
            self.history.append({})
            for gf in self.design.keys():
                self.history[-1][gf] = self.solve_step(**kwargs_per_gf[gf], **kwargs)

            # check convergence:
            if self._check_convergence(tol=tol):
                # TODO: verbose
                # TODO: patience
                # TODO: if only one grouping factor, drop converged groups
                break

            # recompute offset, etc:
            kwargs_per_gf = self._update_kwargs(kwargs_per_gf, fe_offset=fe_offset)

        return self.history[-1]

    def solve_step(self,
                   X: torch.Tensor,
                   y: torch.Tensor,
                   offset: torch.Tensor,
                   group_ids: Sequence,
                   prior_precision: torch.Tensor,
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
            kwargs_per_gf[gf] = self.static_kwargs_per_gf[gf].copy()
            kwargs_per_gf[gf]['offset'] = fe_offset.clone()
            kwargs_per_gf[gf]['prior_precision'] = prior_precisions[gf]
        return kwargs_per_gf

    def _update_kwargs(self, kwargs_per_gf: dict, fe_offset: torch.Tensor) -> Optional[dict]:
        """
        Called on each iteration of the solver's __call__, this method recomputes the offset

        :param kwargs_per_gf: XXX
        :param fe_offset: XXX
        :return:
        """

        # update offsets:
        with torch.no_grad():
            yhat_r = get_yhat_r(design=self.design, X=self.X, group_ids=self.group_ids, res_per_gf=self.history[-1])
            out = {}
            for gf1 in kwargs_per_gf.keys():
                out[gf1] = kwargs_per_gf[gf1].copy()
                out[gf1]['offset'] = [fe_offset.clone()]
                for i, gf2 in enumerate(self.design.keys()):
                    if gf1 == gf2:
                        continue
                    out[gf1]['offset'].append(yhat_r[:, i])
                out[gf1]['offset'] = torch.sum(torch.stack(out[gf1]['offset']), 0)

        return out

    def _check_convergence(self, tol: float) -> bool:
        if len(self.history) < 2:
            return False
        converged = True
        with torch.no_grad():
            for gf in self.design.keys():
                current_res = self.history[-1][gf]
                prev_res = self.history[-2][gf]
                changes = torch.norm(current_res - prev_res, dim=1) / torch.norm(prev_res, dim=1)
                if (changes > tol).any():
                    converged = False
                    break
        return converged
