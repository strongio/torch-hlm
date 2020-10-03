from typing import Sequence, Union, Optional

import torch

from .base import MixedEffectsModule
from .utils import ndarray_to_tensor


class GaussianMixedEffectsModule(MixedEffectsModule):
    def __init__(self,
                 fixeff_features: Union[int, Sequence],
                 raneff_features: Optional[Union[int, Sequence]],
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

    @staticmethod
    def _re_solve(X: torch.Tensor,
                  y: torch.Tensor,
                  XtX: torch.Tensor,
                  offset: torch.Tensor,
                  prior_precision: torch.Tensor,
                  **kwargs) -> torch.Tensor:
        yoff = y - offset
        return torch.solve(torch.sum(X * yoff[:, None], 1).unsqueeze(-1), XtX + prior_precision)

    def get_loss(self, predicted: torch.Tensor, actual: torch.Tensor, re_betas: torch.Tensor) -> torch.Tensor:
        # TODO: this is h-likelihood, should be one but not only option
        actual = ndarray_to_tensor(actual)
        dist = torch.distributions.Normal(predicted, self.residual_std_dev)
        log_prob1 = dist.log_prob(actual).sum()
        log_prob2 = self.re_distribution().log_prob(re_betas).sum()
        return (-log_prob1 - log_prob2) / len(actual)
