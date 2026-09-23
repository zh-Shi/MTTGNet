from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-6

    def fit(self, x: np.ndarray) -> "StandardScaler":
        if x.ndim != 2:
            raise ValueError("x must have shape [T,N]")

        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std = np.maximum(self.std, self.eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self._check()
        return (x - self.mean) / self.std

    def inverse_target(
        self,
        x: np.ndarray,
        target_index: int,
    ) -> np.ndarray:
        self._check()
        return (
            x * self.std[target_index]
            + self.mean[target_index]
        )

    def inverse_dist_params(
        self,
        dist_params: np.ndarray,
        target_index: int,
    ) -> np.ndarray:
        """Inverse-transform SHASH parameters [..., 4] to the original scale.

        The SHASH family is closed under affine transformations:
            Y = a X + b  ⇒  (mu', sigma', gamma', tau') =
                (a*mu + b, a*sigma, gamma, tau).
        So only the location and scale are rescaled; skewness (gamma) and tail
        index (tau) are invariant.  ``dist_params`` is mutated in place.
        """
        self._check()
        d = np.asarray(dist_params).copy()
        d[..., 0] = d[..., 0] * self.std[target_index] + self.mean[target_index]
        d[..., 1] = d[..., 1] * self.std[target_index]
        return d

    def _check(self) -> None:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
