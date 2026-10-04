"""
Mixture model for learning the distribution of number of points n from training data.
Uses Gaussian Mixture Model (GMM) to handle multiple peaks in the distribution.
"""
import os
import warnings
from typing import Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


class NDistributionMixture:
    """
    Mixture model for learning and sampling the distribution of number of points n.

    Uses Gaussian Mixture Model (GMM) to fit the distribution of n values from training data.
    Supports both continuous fitting and discrete sampling (since n must be integers).
    """

    def __init__(
        self,
        n_components: int = 5,
        random_state: Optional[int] = None,
        sampling_seed: Optional[int] = None,
    ):
        """
        Initialize the mixture model.

        Args:
            n_components: Number of Gaussian components in the mixture model
            random_state: Random seed for reproducibility (used for fitting)
            sampling_seed: Random seed for sampling. If None, uses OS entropy.
        """
        self.n_components = int(n_components)
        self.random_state = random_state
        self.gmm: Optional[GaussianMixture] = None
        self.min_n: int = 1
        self.max_n: int = 1
        self.fitted: bool = False

        if sampling_seed is None:
            seed = int.from_bytes(os.urandom(8), "little", signed=False)
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng(int(sampling_seed))

    def fit(self, train_n: torch.Tensor) -> None:
        """
        Fit the mixture model to training data n values.

        Args:
            train_n: Tensor of shape [N] containing number of points for each training sample
        """
        n_values = train_n.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if n_values.size == 0:
            raise ValueError("train_n is empty.")

        self.min_n = int(np.min(n_values))
        self.max_n = int(np.max(n_values))

        n_values_2d = n_values.reshape(-1, 1)
        unique_count = len(np.unique(n_values))
        n_components = max(1, min(self.n_components, unique_count))

        self.gmm = GaussianMixture(
            n_components=n_components,
            random_state=self.random_state,
            max_iter=200,
            covariance_type="full",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gmm.fit(n_values_2d)

        self.fitted = True

    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if not self.fitted or self.gmm is None:
            raise RuntimeError("Model must be fitted before sampling. Call fit() first.")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        weights = np.asarray(self.gmm.weights_, dtype=np.float64)
        weights_sum = weights.sum()
        if not np.isfinite(weights_sum) or weights_sum <= 0:
            raise RuntimeError("Invalid GMM weights.")
        weights = weights / weights_sum

        K = weights.shape[0]
        comp = self._rng.choice(K, size=num_samples, p=weights)

        means = np.asarray(self.gmm.means_[:, 0], dtype=np.float64)
        vars_ = np.asarray(self.gmm.covariances_[:, 0, 0], dtype=np.float64)
        stds = np.sqrt(np.maximum(vars_, 1e-12))

        x = self._rng.normal(loc=means[comp], scale=stds[comp], size=num_samples)

        n = np.rint(x).astype(np.int64)
        n = np.clip(n, self.min_n, self.max_n)

        out = torch.from_numpy(n).long()
        if device is not None:
            out = out.to(device)
        return out

    def get_params(self) -> dict:
        """
        Get the parameters of the fitted model.

        Returns:
            Dictionary containing model parameters
        """
        if not self.fitted or self.gmm is None:
            raise RuntimeError("Model must be fitted before getting parameters.")

        return {
            "means": self.gmm.means_,
            "covariances": self.gmm.covariances_,
            "weights": self.gmm.weights_,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "n_components": int(self.gmm.n_components),
            "covariance_type": self.gmm.covariance_type,
        }

    def load_params(self, params: dict) -> None:
        """
        Load model parameters (for saving/loading).

        Args:
            params: Dictionary containing model parameters
        """
        self.min_n = int(params["min_n"])
        self.max_n = int(params["max_n"])

        n_components = int(params["n_components"])
        covariance_type = params.get("covariance_type", "full")

        self.gmm = GaussianMixture(
            n_components=n_components,
            random_state=self.random_state,
            max_iter=1,
            covariance_type=covariance_type,
        )

        self.gmm.weights_ = np.asarray(params["weights"])
        self.gmm.means_ = np.asarray(params["means"])
        self.gmm.covariances_ = np.asarray(params["covariances"])

        self.gmm.precisions_cholesky_ = None
        self.gmm.converged_ = True
        self.gmm.n_features_in_ = 1

        self.fitted = True
