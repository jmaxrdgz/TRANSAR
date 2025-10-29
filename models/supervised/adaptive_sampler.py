import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler
from typing import Dict, List, Optional


class AdaptiveSampler:
    """
    Adaptive Sampling Scheduler for TRANSAR based on curriculum learning.

    Implements the adaptive sampling strategy from Section 2.2 of the TRANSAR paper,
    which dynamically adjusts the target class distribution during training.

    The scheduler transitions from an imbalanced distribution (over-sampling minority classes)
    to a balanced distribution using:
    - g(t): Curriculum scheduler function (linear, cosine, exponential)
    - h(t): Performance-based regularizer (F1 score)

    Args:
        dataset: Training dataset with get_sample_classes() method
        num_classes: Number of classes (typically 2 for foreground/background)
        scheduler_type: Type of g(t) function ('linear', 'cosine', 'exponential')
        alpha: Weight for balancing g(t) and h(t), default 0.8 as in paper
        total_epochs: Total number of training epochs
        device: Device for tensor computations
    """

    def __init__(
        self,
        dataset,
        num_classes: int = 2,
        scheduler_type: str = 'cosine',
        alpha: float = 0.8,
        total_epochs: int = 100,
        device: str = 'cpu'
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.scheduler_type = scheduler_type
        self.alpha = alpha
        self.total_epochs = total_epochs
        self.device = device

        # Compute class distribution from dataset
        self.class_counts = self.compute_class_distribution()

        # Compute d_train (initial imbalanced distribution)
        self.d_train = self.compute_d_train()

        # Current target distribution (initialized to d_train)
        self.d_target = self.d_train.clone()

        # Track previous F1 score
        self.prev_f1_score = 0.0

    def compute_class_distribution(self) -> Dict[int, int]:
        """
        Compute the cardinality C_i for each class.

        For binary classification (foreground/background):
        - Class 0: background (samples with no objects)
        - Class 1: foreground (samples with objects)

        Returns:
            Dictionary mapping class_id to count
        """
        class_counts = {i: 0 for i in range(self.num_classes)}

        for idx in range(len(self.dataset)):
            # Get classes present in this sample
            sample_classes = self.dataset.get_sample_classes(idx)

            if len(sample_classes) == 0:
                # No objects = background
                class_counts[0] += 1
            else:
                # Has objects = foreground
                class_counts[1] += 1

        return class_counts

    def compute_d_train(self) -> torch.Tensor:
        """
        Compute the training distribution d_train.

        For each class i:
            d_train[i] = 1 - C_i / C_max

        This over-samples minority classes by giving them higher weights.

        Returns:
            Tensor of shape [num_classes] with training distribution
        """
        C_max = max(self.class_counts.values())
        d_train = torch.zeros(self.num_classes, device=self.device)

        for i in range(self.num_classes):
            C_i = self.class_counts.get(i, 0)
            d_train[i] = 1.0 - (C_i / C_max)

        return d_train

    def compute_d_target(self, epoch: int, f1_score: Optional[float] = None) -> torch.Tensor:
        """
        Compute target distribution d_target at epoch t using Equation 2 from paper.

        d_target_i(t) = (d_train_i)^(α*g(t) + (1-α)*h(t))  for t > 0
        d_target_i(0) = d_train_i                          for t = 0

        Args:
            epoch: Current epoch number (0-indexed)
            f1_score: F1 score from validation for h(t). If None, uses previous score.

        Returns:
            Tensor of shape [num_classes] with target distribution
        """
        if epoch == 0:
            return self.d_train.clone()

        # Update F1 score if provided
        if f1_score is not None:
            self.prev_f1_score = f1_score

        # Compute g(t) - curriculum scheduler
        g_t = self._scheduler_function_g(epoch)

        # Compute h(t) - performance regularizer
        h_t = self._performance_function_h(self.prev_f1_score)

        # Compute combined exponent
        exponent = self.alpha * g_t + (1 - self.alpha) * h_t

        # Apply power function to d_train
        d_target = torch.pow(self.d_train, exponent)

        return d_target

    def _scheduler_function_g(self, epoch: int) -> float:
        """
        Compute the curriculum scheduler function g(t) ∈ [0, 1].

        Maps from 0 (start) to 1 (end) over total_epochs.
        - At t=0: g(0)=0, favoring imbalanced distribution
        - At t=T: g(T)=1, favoring balanced distribution

        Args:
            epoch: Current epoch number

        Returns:
            Scheduler value in [0, 1]
        """
        # Normalize epoch to [0, 1]
        t = min(epoch / max(self.total_epochs - 1, 1), 1.0)

        if self.scheduler_type == 'linear':
            return t

        elif self.scheduler_type == 'cosine':
            # Cosine annealing: slow start, fast middle, slow end
            return (1 - np.cos(np.pi * t)) / 2

        elif self.scheduler_type == 'exponential':
            # Exponential growth: slow start, accelerating end
            return np.exp(t) - 1

        else:
            raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")

    def _performance_function_h(self, f1_score: float) -> float:
        """
        Compute the performance regularizer function h(t) ∈ [0, 1].

        Based on model F1 score to dynamically prioritize classes that lead to false detections.
        Higher F1 means better performance, so we move toward balanced sampling faster.

        Args:
            f1_score: F1 score from validation (0 to 1)

        Returns:
            Performance value in [0, 1]
        """
        # Clip F1 score to valid range
        f1_score = np.clip(f1_score, 0.0, 1.0)

        # Use F1 score directly as h(t)
        # High F1 → h(t) close to 1 → move toward balanced distribution
        # Low F1 → h(t) close to 0 → stay with imbalanced distribution
        return f1_score

    def compute_loss_weights(self, d_target: torch.Tensor) -> torch.Tensor:
        """
        Compute per-class loss weights w(t) based on target distribution.

        w(t) = max{d_train / d_target(t), 1}  (element-wise)

        This re-weights the loss to account for changing sampling distribution.

        Args:
            d_target: Target distribution for current epoch

        Returns:
            Tensor of shape [num_classes] with loss weights
        """
        # Avoid division by zero
        d_target_safe = torch.clamp(d_target, min=1e-8)

        # Compute ratio
        weights = self.d_train / d_target_safe

        # Apply max with 1
        weights = torch.maximum(weights, torch.ones_like(weights))

        return weights

    def get_sample_weights(self, d_target: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Each sample is assigned a weight based on its class membership.

        Args:
            d_target: Target distribution for current epoch

        Returns:
            Tensor of shape [num_samples] with sample weights
        """
        sample_weights = torch.zeros(len(self.dataset), dtype=torch.float32)

        for idx in range(len(self.dataset)):
            sample_classes = self.dataset.get_sample_classes(idx)

            if len(sample_classes) == 0:
                # Background sample
                sample_weights[idx] = d_target[0].item()
            else:
                # Foreground sample
                sample_weights[idx] = d_target[1].item()

        return sample_weights

    def get_weighted_sampler(
        self,
        epoch: int,
        f1_score: Optional[float] = None,
        num_samples: Optional[int] = None
    ) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for the current epoch.

        Args:
            epoch: Current epoch number
            f1_score: F1 score from validation
            num_samples: Number of samples to draw. If None, uses dataset length.

        Returns:
            WeightedRandomSampler instance
        """
        # Compute target distribution for this epoch
        self.d_target = self.compute_d_target(epoch, f1_score)

        # Compute sample weights
        sample_weights = self.get_sample_weights(self.d_target)

        # Create sampler
        if num_samples is None:
            num_samples = len(self.dataset)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True  # Allow replacement for proper weighting
        )

        return sampler

    def get_current_loss_weights(self) -> torch.Tensor:
        """
        Get loss weights for the current target distribution.

        Returns:
            Tensor of shape [num_classes] with loss weights
        """
        return self.compute_loss_weights(self.d_target)

    def get_distribution_info(self) -> Dict[str, any]:
        """
        Get information about current sampling distribution for logging.

        Returns:
            Dictionary with distribution statistics
        """
        return {
            'class_counts': self.class_counts,
            'd_train': self.d_train.cpu().numpy(),
            'd_target': self.d_target.cpu().numpy(),
            'loss_weights': self.get_current_loss_weights().cpu().numpy(),
            'f1_score': self.prev_f1_score
        }
