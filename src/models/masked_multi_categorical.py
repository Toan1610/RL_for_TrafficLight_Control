"""
Masked Multi-Categorical Action Distribution for Discrete Cycle Adjustment.

This module implements a custom action distribution for MultiDiscrete action spaces
with action masking support. It is designed for the Discrete Cycle Adjustment
approach where each traffic phase gets an independent discrete action:
    Action 0 = decrease green time (-5s)
    Action 1 = keep green time (0s)
    Action 2 = increase green time (+5s)

Action Masking:
- Invalid phases (determined by FRAP PhaseStandardizer) are forced to select
  action 1 (keep/no change) by setting logits to [-inf, 0, -inf].
- This ensures the model never wastes gradient on invalid phases.

Usage:
    from src.models.masked_multi_categorical import register_masked_multi_categorical
    register_masked_multi_categorical()
    # In PPO config:
    config.training(model={"custom_action_dist": "masked_multi_categorical"})
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, TYPE_CHECKING

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType, ModelConfigDict

if TYPE_CHECKING:
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


# Number of standard phases
NUM_STANDARD_PHASES = 8

# Number of discrete actions per phase: 0=decrease, 1=keep, 2=increase
NUM_ACTIONS_PER_PHASE = 3

# Large negative value for masking
MASK_VALUE = -1e9

# Index of "keep/no-change" action
KEEP_ACTION_IDX = 1


class TorchMaskedMultiCategorical(TorchDistributionWrapper):
    """Masked Multi-Categorical distribution for MultiDiscrete action spaces.

    For each of the 8 standard phases, maintains an independent categorical
    distribution over {decrease, keep, increase}. Invalid phases are masked
    to always select "keep" (no change).

    The model must store action_mask in self._last_action_mask before
    the distribution is created (same as MaskedSoftmax).

    Model output: [batch, NUM_STANDARD_PHASES * NUM_ACTIONS_PER_PHASE]
    = [batch, 24] logits, split into 8 groups of 3.
    """

    @override(ActionDistribution)
    def __init__(
        self,
        inputs: TensorType,
        model: "TorchModelV2",
        *,
        action_space=None,
    ):
        super().__init__(inputs, model)

        self.num_phases = NUM_STANDARD_PHASES
        self.num_actions = NUM_ACTIONS_PER_PHASE

        # Split inputs into per-phase logits: [batch, 8, 3]
        batch_size = inputs.shape[0]
        self.all_logits = inputs.view(batch_size, self.num_phases, self.num_actions)

        # Get action mask from model
        if hasattr(model, '_last_action_mask') and model._last_action_mask is not None:
            self.action_mask = model._last_action_mask.to(inputs.device)
            if self.action_mask.dim() == 1:
                self.action_mask = self.action_mask.unsqueeze(0).expand(batch_size, -1)
        else:
            self.action_mask = torch.ones(batch_size, self.num_phases, device=inputs.device)

        # Apply mask: for invalid phases, force logits to select "keep"
        self.masked_logits = self._apply_mask(self.all_logits)

        # Compute per-phase log-probabilities
        self.log_probs = F.log_softmax(self.masked_logits, dim=-1)  # [B, 8, 3]

        self.last_sample = None

    def _apply_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply action mask to logits.

        For invalid phases (mask=0), set logits to force "keep" action:
        [-inf, 0, -inf] so softmax gives [0, 1, 0].
        """
        # mask_expanded: [B, 8, 1]
        mask = self.action_mask.unsqueeze(-1)  # [B, 8, 1]

        # For invalid phases: replace logits with [-inf, 0, -inf]
        keep_only = torch.full_like(logits, MASK_VALUE)
        keep_only[..., KEEP_ACTION_IDX] = 0.0

        # Where mask=1 use original logits, where mask=0 use keep_only
        masked = logits * mask + keep_only * (1.0 - mask)
        return masked

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        """Return the mode (argmax) of each phase's categorical distribution."""
        # [B, 8]
        actions = torch.argmax(self.masked_logits, dim=-1)
        self.last_sample = actions
        return actions

    @override(ActionDistribution)
    def sample(self) -> TensorType:
        """Sample from each phase's categorical distribution independently."""
        # Gumbel-max trick for differentiable sampling
        uniform = torch.rand_like(self.masked_logits).clamp(1e-8, 1.0 - 1e-8)
        gumbels = -torch.log(-torch.log(uniform))
        actions = torch.argmax(self.masked_logits + gumbels, dim=-1)  # [B, 8]
        self.last_sample = actions
        return actions

    @override(ActionDistribution)
    def logp(self, actions: TensorType) -> TensorType:
        """Compute log-probability of given actions.

        Args:
            actions: [batch, 8] integer actions (0, 1, or 2 for each phase)
        """
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        # Gather log-prob for each phase's chosen action
        # log_probs: [B, 8, 3], actions: [B, 8] -> gather on dim=2
        log_p = self.log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B, 8]

        # Sum log-probs across all phases (independent categoricals)
        total_log_p = log_p.sum(dim=-1)  # [B]
        return total_log_p

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        """Return log probability of the last sampled action."""
        assert self.last_sample is not None, "Must call sample() first"
        return self.logp(self.last_sample)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        """Compute entropy as sum of per-phase categorical entropies."""
        # H = -sum(p * log(p)) for each phase, then sum across phases
        probs = torch.exp(self.log_probs)  # [B, 8, 3]
        per_phase_entropy = -(probs * self.log_probs).sum(dim=-1)  # [B, 8]

        # Only count entropy for valid phases
        masked_entropy = per_phase_entropy * self.action_mask  # [B, 8]
        total_entropy = masked_entropy.sum(dim=-1)  # [B]
        return total_entropy

    @override(ActionDistribution)
    def kl(self, other: "TorchMaskedMultiCategorical") -> TensorType:
        """Compute KL divergence KL(self || other)."""
        probs_self = torch.exp(self.log_probs)  # [B, 8, 3]
        log_probs_other = other.log_probs  # [B, 8, 3]

        # KL per phase = sum(p * (log(p) - log(q)))
        per_phase_kl = (probs_self * (self.log_probs - log_probs_other)).sum(dim=-1)  # [B, 8]

        # Only count KL for valid phases
        masked_kl = per_phase_kl * self.action_mask  # [B, 8]
        total_kl = masked_kl.sum(dim=-1)  # [B]
        return total_kl

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space,
        model_config: ModelConfigDict,
    ) -> Union[int, np.ndarray]:
        """Return required model output size = 8 * 3 = 24."""
        return NUM_STANDARD_PHASES * NUM_ACTIONS_PER_PHASE


def register_masked_multi_categorical():
    """Register the Masked Multi-Categorical distribution with RLlib."""
    from ray.rllib.models import ModelCatalog
    ModelCatalog.register_custom_action_dist(
        "masked_multi_categorical", TorchMaskedMultiCategorical
    )
    print("[MaskedMultiCategorical] Registered 'masked_multi_categorical' action distribution")
