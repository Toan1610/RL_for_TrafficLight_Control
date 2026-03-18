"""GESA (General Environment Standardization Architecture) Wrappers.

This module provides composable gymnasium wrappers for standardizing
observations, actions, and rewards in multi-agent traffic signal control.

Architecture:
    SumoMultiAgentEnv
      └── GESAObservationWrapper  (normalize obs to zero-mean, unit-variance)
            └── GESARewardWrapper (normalize rewards via RunningMeanStd)

These wrappers decouple normalization logic from the core SUMO environment,
making them independently testable, configurable, and reusable.

Author: Bui Chi Toan
Date: 2025
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import json

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from preprocessing.observation_normalizer import RunningMeanStd


class GESAObservationWrapper(MultiAgentEnv):
    """Normalizes per-agent observations to zero-mean, unit-variance.

    For each feature dimension in the observation vector (density, queue,
    occupancy, average_speed), maintains running statistics and normalizes
    online using Welford's algorithm.

    This wrapper operates on the dict-based multi-agent observation space:
        obs = {agent_id: {"obs": np.ndarray, "action_mask": np.ndarray}, ...}

    Only the "obs" key is normalized; "action_mask" is passed through unchanged.
    """

    def __init__(self, env: MultiAgentEnv, clip_obs: float = 10.0, min_samples: int = 10):
        """
        Args:
            env: The wrapped multi-agent environment.
            clip_obs: Clip normalized observations to [-clip_obs, clip_obs].
            min_samples: Minimum samples before applying normalization.
        """
        self.env = env
        self.clip_obs = clip_obs
        self.min_samples = min_samples

        # Per-agent running statistics for observations
        # Initialized lazily on first observation
        self._obs_rms: Dict[str, RunningMeanStd] = {}

        # Expose underlying environment attributes
        self.ts_ids = env.ts_ids
        self.single_agent = getattr(env, "single_agent", False)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)

    def observation_spaces(self, ts_id: str):
        return self.env.observation_spaces(ts_id)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def _normalize_obs(self, obs_dict: dict) -> dict:
        """Normalize observations for all agents."""
        normalized = {}
        for agent_id, obs in obs_dict.items():
            if isinstance(obs, dict) and "obs" in obs:
                raw_obs = obs["obs"]
            elif isinstance(obs, np.ndarray):
                raw_obs = obs
            else:
                normalized[agent_id] = obs
                continue

            # Initialize RMS on first encounter
            if agent_id not in self._obs_rms:
                self._obs_rms[agent_id] = RunningMeanStd(shape=raw_obs.shape)

            rms = self._obs_rms[agent_id]
            rms.update(raw_obs)

            if rms.count > self.min_samples:
                norm = rms.normalize(raw_obs, clip=self.clip_obs)
            else:
                norm = raw_obs  # Pass through until enough samples

            # Preserve dict structure
            if isinstance(obs, dict):
                normalized[agent_id] = {**obs, "obs": norm}
            else:
                normalized[agent_id] = norm

        return normalized

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and not self.single_agent:
            obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)
        if isinstance(obs, dict):
            obs = self._normalize_obs(obs)
        return obs, rewards, terminateds, truncateds, infos

    def close(self):
        return self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_obs_normalizer_state(self) -> dict:
        """Serialize observation normalizer state for checkpointing."""
        return {
            agent_id: rms.get_state()
            for agent_id, rms in self._obs_rms.items()
        }

    def set_obs_normalizer_state(self, state: dict):
        """Restore observation normalizer state from checkpoint."""
        for agent_id, rms_state in state.items():
            if agent_id not in self._obs_rms:
                # Infer shape from state
                mean = np.array(rms_state["mean"])
                self._obs_rms[agent_id] = RunningMeanStd(shape=mean.shape)
            self._obs_rms[agent_id].set_state(rms_state)


class GESARewardWrapper(MultiAgentEnv):
    """Normalizes multi-agent rewards using shared RunningMeanStd.

    Applies (reward - mean) / std normalization with optional clipping.
    Statistics are updated online across all agents and all steps.

    This wrapper extracts the reward normalization logic that was previously
    embedded in SumoEnvironment.step(), making it composable and testable.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        clip_rewards: float = 10.0,
        min_samples: int = 10,
        gamma: float = 0.995,
    ):
        """
        Args:
            env: The wrapped multi-agent environment.
            clip_rewards: Clip normalized rewards to [-clip, clip].
            min_samples: Minimum samples before applying normalization.
            gamma: Discount factor (reserved for return-based normalization).
        """
        self.env = env
        self.clip_rewards = clip_rewards
        self.min_samples = min_samples
        self.gamma = gamma
        self.reward_rms = RunningMeanStd(shape=())
        self._state_file: Optional[Path] = None

        # Expose underlying environment attributes
        self.ts_ids = env.ts_ids
        self.single_agent = getattr(env, "single_agent", False)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)

    def observation_spaces(self, ts_id: str):
        return self.env.observation_spaces(ts_id)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def _normalize_rewards(self, rewards: dict) -> Tuple[dict, dict]:
        """Normalize rewards and return both normalized and raw versions."""
        raw_rewards = dict(rewards)

        # Batch update statistics
        reward_values = np.array(list(rewards.values()), dtype=np.float64)
        if len(reward_values) > 0 and not np.all(reward_values == reward_values[0]):
            self.reward_rms.update(reward_values)

        normalized = {}
        for agent_id, reward in rewards.items():
            if self.reward_rms.count > self.min_samples:
                std = np.sqrt(float(self.reward_rms.var) + 1e-8)
                if std > 1e-6:
                    norm = (reward - float(self.reward_rms.mean)) / std
                else:
                    norm = reward
            else:
                norm = reward

            if self.clip_rewards is not None:
                norm = np.clip(norm, -self.clip_rewards, self.clip_rewards)

            if np.isnan(norm) or np.isinf(norm):
                norm = 0.0

            normalized[agent_id] = float(norm)

        return normalized, raw_rewards

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = self.env.step(action_dict)

        # Normalize rewards
        normalized_rewards, raw_rewards = self._normalize_rewards(rewards)

        # Store raw rewards in infos for diagnostics
        for agent_id in raw_rewards:
            if agent_id in infos and isinstance(infos[agent_id], dict):
                infos[agent_id]["raw_reward"] = float(raw_rewards[agent_id])
            else:
                infos[agent_id] = {"raw_reward": float(raw_rewards[agent_id])}

        episode_done = bool(terminateds.get("__all__", False) or truncateds.get("__all__", False))
        if episode_done:
            self._save_state_to_file()

        return obs, normalized_rewards, terminateds, truncateds, infos

    def close(self):
        return self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_reward_normalizer_state(self) -> dict:
        """Serialize reward normalizer state for checkpointing."""
        return self.reward_rms.get_state()

    def set_reward_normalizer_state(self, state: dict):
        """Restore reward normalizer state from checkpoint."""
        self.reward_rms.set_state(state)

    # Compatibility aliases with SumoEnvironment internal normalizer API
    def get_normalizer_state(self) -> dict:
        return self.get_reward_normalizer_state()

    def set_normalizer_state(self, state: dict):
        self.set_reward_normalizer_state(state)

    def set_state_file(self, state_file: Optional[str]) -> None:
        """Attach normalizer state file and restore if present."""
        if not state_file:
            self._state_file = None
            return

        self._state_file = Path(state_file)
        try:
            if self._state_file.exists():
                with open(self._state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.set_reward_normalizer_state(state)
                print(
                    f"[GESA] Restored reward normalizer from {self._state_file}: "
                    f"mean={state.get('mean', 0):.4f}, var={state.get('var', 1):.4f}, "
                    f"count={state.get('count', 0):.0f}"
                )
        except Exception as e:
            print(f"[GESA] Warning: failed to restore reward normalizer state: {e}")

    def _save_state_to_file(self) -> None:
        """Persist reward normalizer state if state file is configured."""
        if self._state_file is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            state = self.get_reward_normalizer_state()
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[GESA] Warning: failed to save reward normalizer state: {e}")


def wrap_with_gesa(
    env: MultiAgentEnv,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_obs: float = 10.0,
    clip_rewards: float = 10.0,
    gamma: float = 0.995,
) -> MultiAgentEnv:
    """Apply GESA wrappers to a multi-agent environment.

    Convenience function that composes the observation and reward wrappers.

    Args:
        env: Base multi-agent environment.
        normalize_obs: Whether to apply observation normalization.
        normalize_reward: Whether to apply reward normalization.
        clip_obs: Observation clip value.
        clip_rewards: Reward clip value.
        gamma: Discount factor for reward normalization.

    Returns:
        Wrapped environment with GESA normalization applied.

    Example:
        >>> base_env = SumoMultiAgentEnv(**config)
        >>> env = wrap_with_gesa(base_env, normalize_obs=True, normalize_reward=True)
    """
    if normalize_obs:
        env = GESAObservationWrapper(env, clip_obs=clip_obs)
    if normalize_reward:
        env = GESARewardWrapper(env, clip_rewards=clip_rewards, gamma=gamma)
    return env
