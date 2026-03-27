"""Observation functions for traffic signals."""

from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal

# Number of standard phases for FRAP action masking
NUM_STANDARD_PHASES = 8
# Fixed detector/lane slots expected by MGMQ model: 12 lanes * 4 features = 48
TARGET_NUM_LANES = 12
# Number of green-time ratio features (one per standard phase, normalized by max_green)
# These encode the *current* green-time allocation so the agent can reason about
# how far each phase is from its limits — essential for countdown-timer control.
NUM_GREEN_TIME_FEATURES = NUM_STANDARD_PHASES  # = 8, always padded to full standard phases


def _pad_or_trim(values, target_len: int = TARGET_NUM_LANES) -> np.ndarray:
    """Pad/truncate 1D values to a fixed length for stable observation shapes."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size >= target_len:
        return arr[:target_len]
    padded = np.zeros(target_len, dtype=np.float32)
    padded[:arr.size] = arr
    return padded


def _get_green_time_features(ts) -> np.ndarray:
    """Return current green-time ratios for all NUM_STANDARD_PHASES phases.

    Maps ``ts.current_green_times`` (length = num_green_phases) to a fixed-size
    vector of length 8 (NUM_STANDARD_PHASES) normalised by ``ts.max_green``.
    Unused phase slots are filled with 0.

    This feature is crucial for countdown-timer control: the agent observes how
    each phase's green time compares to the maximum, enabling it to decide
    whether to increase, keep, or decrease the allocation.
    """
    result = np.zeros(NUM_GREEN_TIME_FEATURES, dtype=np.float32)
    max_g = max(float(ts.max_green), 1.0)
    for i, gt in enumerate(ts.current_green_times[:NUM_GREEN_TIME_FEATURES]):
        result[i] = float(gt) / max_g
    return result


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals.
    
    Returns a Dict observation with:
    - features: Lane features [48] = 12 lanes * 4 features
    - action_mask: Binary mask [8] for valid phases (FRAP)
    """

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> dict:
        """Return the observation dict with features and action_mask."""
        density = _pad_or_trim(self.ts.get_lanes_density_by_detectors())
        queue = _pad_or_trim(self.ts.get_lanes_queue_by_detectors())
        occupancy = _pad_or_trim(self.ts.get_lanes_occupancy_by_detectors())
        average_speed = _pad_or_trim(self.ts.get_lanes_average_speed_by_detectors())
        # CRITICAL FIX: Reorder to Lane-major [Lane0_Feats, Lane1_Feats, ...]
        # This matches the model's expectation: .view(-1, 12, 4)
        obs_data = []
        for i in range(TARGET_NUM_LANES):
            obs_data.extend([density[i], queue[i], occupancy[i], average_speed[i]])

        # Lane features [48]: density/queue/occupancy/speed per lane
        features = np.array(obs_data, dtype=np.float32)
        # CRITICAL: Clip to [0, 1] AND ensure dtype is float32 to match observation_space
        features = np.clip(features, 0.0, 1.0).astype(np.float32)

        # Green-time ratio features [8]: current green allocation / max_green per phase
        # Encodes the countdown-timer state so the agent can decide ±Δ adjustments
        green_time_features = _get_green_time_features(self.ts)  # [0, 1]^8

        # Concatenate: [lane_features(48), green_time_features(8)] = [56]
        features = np.concatenate([features, green_time_features])

        # Get action mask from TrafficSignal (FRAP PhaseStandardizer)
        action_mask = self.ts.get_action_mask()
        action_mask = np.array(action_mask, dtype=np.float32)

        return {
            "features": features,
            "action_mask": action_mask,
        }

    def observation_space(self) -> spaces.Dict:
        """Return the observation space as Dict."""
        # 48 lane features + 8 green-time ratio features = 56 total
        feature_dim = 4 * TARGET_NUM_LANES + NUM_GREEN_TIME_FEATURES
        return spaces.Dict({
            "features": spaces.Box(
                low=np.zeros(feature_dim, dtype=np.float32),
                high=np.ones(feature_dim, dtype=np.float32),
            ),
            "action_mask": spaces.Box(
                low=np.zeros(NUM_STANDARD_PHASES, dtype=np.float32),
                high=np.ones(NUM_STANDARD_PHASES, dtype=np.float32),
            ),
        })


class SpatioTemporalObservationFunction(ObservationFunction):
    """Observation function that returns a history of observations.
    
    Returns a Dict observation with:
    - features: Flattened history [window_size * feature_dim]
    - action_mask: Binary mask [8] for valid phases (FRAP)
    """

    def __init__(self, ts: TrafficSignal, window_size: int = 5):
        """Initialize spatio-temporal observation function."""
        super().__init__(ts)
        self.window_size = window_size
        # Try to get window_size from TrafficSignal if available (will be added later)
        if hasattr(ts, "window_size"):
            self.window_size = ts.window_size

    def compute_current_observation(self) -> np.ndarray:
        """Return the current single-step observation (48 lane features + 8 green-time features = 56)."""
        density = _pad_or_trim(self.ts.get_lanes_density_by_detectors())
        queue = _pad_or_trim(self.ts.get_lanes_queue_by_detectors())
        occupancy = _pad_or_trim(self.ts.get_lanes_occupancy_by_detectors())
        average_speed = _pad_or_trim(self.ts.get_lanes_average_speed_by_detectors())
        # CRITICAL FIX: Reorder to Lane-major [Lane0_Feats, Lane1_Feats, ...]
        # This matches the model's expectation: .view(-1, 12, 4)
        obs_data = []
        for i in range(TARGET_NUM_LANES):
            obs_data.extend([density[i], queue[i], occupancy[i], average_speed[i]])

        # Lane features [48]
        observation = np.array(obs_data, dtype=np.float32)
        observation = np.clip(observation, 0.0, 1.0).astype(np.float32)

        # Append green-time ratio features [8] for countdown-timer context
        green_time_features = _get_green_time_features(self.ts)
        observation = np.concatenate([observation, green_time_features])  # [56]
        return observation

    def __call__(self) -> dict:
        """Return the stacked observation history with action_mask."""
        # Get history from TrafficSignal
        history = self.ts.get_observation_history(self.window_size)
        
        # Stack into a single array [window_size, features]
        # Flattening might be needed depending on how the model expects it, 
        # but usually we want to keep the time dimension separate or flatten it.
        # Here we return flattened array [window_size * features] to be compatible with Gym spaces
        # The model will reshape it back to [window_size, features]
        stacked_obs = np.array(history, dtype=np.float32).flatten()
        # Lane features are already clipped in compute_current_observation
        # We don't clip here to allow green-time features to exceed 1.0 if needed
        features = stacked_obs
        
        # Get action mask from TrafficSignal (FRAP PhaseStandardizer)
        action_mask = self.ts.get_action_mask()
        action_mask = np.array(action_mask, dtype=np.float32)
        
        return {
            "features": features,
            "action_mask": action_mask,
        }

    def observation_space(self) -> spaces.Dict:
        """Return the observation space as Dict."""
        # 48 lane features + 8 green-time ratio features = 56 per timestep
        feature_size = 4 * TARGET_NUM_LANES + NUM_GREEN_TIME_FEATURES

        return spaces.Dict({
            "features": spaces.Box(
                low=np.zeros(self.window_size * feature_size, dtype=np.float32),
                high=np.ones(self.window_size * feature_size, dtype=np.float32),
            ),
            "action_mask": spaces.Box(
                low=np.zeros(NUM_STANDARD_PHASES, dtype=np.float32),
                high=np.ones(NUM_STANDARD_PHASES, dtype=np.float32),
            ),
        })


class NeighborTemporalObservationFunction(ObservationFunction):
    """Spatio-Temporal observation with pre-packaged neighbor features.
    
    This observation function is designed for Local GNN processing with RLlib.
    Instead of requiring global graph structure, each observation contains:
    - Self features history: [T, feature_dim]
    - Neighbor features history: [K, T, feature_dim]
    - Neighbor mask: [K] indicating which neighbors are valid
    
    This allows the model to perform local GNN operations without
    needing to reconstruct the global graph from shuffled batches.
    """

    def __init__(
        self, 
        ts: TrafficSignal, 
        neighbor_provider=None,
        max_neighbors: int = 4,
        window_size: int = 5
    ):
        """Initialize neighbor temporal observation function.
        
        Args:
            ts: TrafficSignal instance for this agent
            neighbor_provider: Object that provides neighbor info and observations
            max_neighbors: Maximum number of neighbors (pad if less)
            window_size: Number of historical timesteps (T)
        """
        super().__init__(ts)
        self.neighbor_provider = neighbor_provider
        self.max_neighbors = max_neighbors
        self.window_size = window_size
        
        # Override from TrafficSignal if available
        if hasattr(ts, "window_size"):
            self.window_size = ts.window_size
            
    def compute_current_observation(self) -> np.ndarray:
        """Return the current single-step observation (for history tracking)."""
        density = _pad_or_trim(self.ts.get_lanes_density_by_detectors())
        queue = _pad_or_trim(self.ts.get_lanes_queue_by_detectors())
        occupancy = _pad_or_trim(self.ts.get_lanes_occupancy_by_detectors())
        average_speed = _pad_or_trim(self.ts.get_lanes_average_speed_by_detectors())
        
        # CRITICAL FIX: Reorder to Lane-major [Lane0_Feats, Lane1_Feats, ...]
        obs_data = []
        for i in range(TARGET_NUM_LANES):
            obs_data.extend([density[i], queue[i], occupancy[i], average_speed[i]])
            
        # Add green time features (8 values)
        green_time_features = _get_green_time_features(self.ts)
        obs_data.extend(green_time_features)
            
        observation = np.array(obs_data, dtype=np.float32)
        # Fix: only clip lane features (if needed). Assuming safe to clip all.
        # Actually observation space might be different for green time,
        # but green time is a ratio [0, 1] anyway.
        # We will not apply clip here to match DefaultObservationFunction.
        return observation

    def __call__(self) -> dict:
        """Return Dict observation with self, neighbor features and mask.
        
        Returns:
            Dict with keys:
                - self_features: np.ndarray of shape [T, feature_dim]
                - neighbor_features: np.ndarray of shape [K, T, feature_dim]  
                - neighbor_mask: np.ndarray of shape [K]
        """
        # 48 lane features + 8 green-time features = 56
        feature_dim = 4 * TARGET_NUM_LANES + 8
        
        # Get T-step history for self
        # Note: self.ts.get_observation_history now returns validated/clipped observations
        # which might be Dicts if using nested observations.
        # However, for Local GNN, we expect the BASE observation to be a vector (Box)
        # We need to extract the raw features vector if the history contains Dicts.
        
        self_history = self.ts.get_observation_history(self.window_size)
        
        # Helper to extract feature vector from observation history element
        def extract_features(obs):
            if isinstance(obs, dict):
                # If observation is already a dict (e.g. from previous step's NeighborObs),
                # we need to extract the "self_features" part or flatten it.
                # BUT: The base observation function usually returns a vector.
                # If we get a dict here, it means we are using a complex observation class
                # that wraps another one.
                # For NeighborTemporalObservationFunction, the base internal observation
                # should be the DEFAULT vector observation.
                if "self_features" in obs:
                    return obs["self_features"][-1] # Take most recent if it's a sequence
                # Fallback: flatten values
                return np.concatenate([v.flatten() for v in obs.values()])
            return np.asarray(obs, dtype=np.float32)

        processed_history = [extract_features(obs) for obs in self_history]
        self_features = np.array(processed_history, dtype=np.float32)  # [T, feature_dim]
        # Ensure shape [T, feature_dim] (handle if extract_features returns [1, feature_dim])
        if self_features.ndim > 2:
             self_features = self_features.reshape(self.window_size, -1)
             
        self_features = np.clip(self_features, 0.0, 1.0)
        
        # Get neighbor features
        neighbor_features, neighbor_mask = self._get_neighbor_features(feature_dim)
        
        # Get neighbor directions (0=N, 1=E, 2=S, 3=W, -1=padding)
        neighbor_directions = self._get_neighbor_directions()
        
        # Get action mask from TrafficSignal (FRAP PhaseStandardizer)
        action_mask = self.ts.get_action_mask()
        action_mask = np.array(action_mask, dtype=np.float32)
        
        return {
            "self_features": self_features,
            "neighbor_features": neighbor_features,
            "neighbor_mask": neighbor_mask,
            "neighbor_directions": neighbor_directions,
            "action_mask": action_mask,
        }
        
    def _get_neighbor_directions(self) -> np.ndarray:
        """Get direction indices for neighbors.
        
        Returns:
            np.ndarray of shape [K] with direction indices:
            0=North, 1=East, 2=South, 3=West, -1=padding.
            Encoded as float: N=0.0, E=0.25, S=0.5, W=0.75, padding=-1.0
        """
        K = self.max_neighbors
        # Default: all padding (-1)
        directions = np.full(K, -1.0, dtype=np.float32)
        
        if self.neighbor_provider is not None and hasattr(self.neighbor_provider, 'get_neighbor_directions'):
            dir_indices = self.neighbor_provider.get_neighbor_directions(self.ts.id)
            for i, d in enumerate(dir_indices[:K]):
                if d >= 0:
                    # Encode as normalized float: 0=0.0, 1=0.25, 2=0.5, 3=0.75
                    directions[i] = d / 4.0
        
        return directions
    
    def _get_neighbor_features(self, feature_dim: int):
        """Get T-step history for all neighbors with padding.
        
        Args:
            feature_dim: Single observation feature dimension
            
        Returns:
            Tuple of (neighbor_features, neighbor_mask)
            - neighbor_features: [K, T, feature_dim]
            - neighbor_mask: [K] with 1.0 for valid neighbors, 0.0 for padding
        """
        K = self.max_neighbors
        T = self.window_size
        
        # Initialize with zeros (padding)
        neighbor_features = np.zeros((K, T, feature_dim), dtype=np.float32)
        neighbor_mask = np.zeros(K, dtype=np.float32)
        
        if self.neighbor_provider is None:
            return neighbor_features, neighbor_mask
            
        # Get list of neighbor IDs
        neighbor_ids = self.neighbor_provider.get_neighbor_ids(self.ts.id)

        # === FIX: SORT NEIGHBOR IDS TO ENSURE CONSISTENT ORDER FOR BiGRU ===
        # BiGRU is sequence-sensitive. Random order from Set causes fluctuation.
        if neighbor_ids:
            neighbor_ids = sorted(neighbor_ids)
        # ===================================================================
        
        for i, neighbor_id in enumerate(neighbor_ids[:K]):
            if neighbor_id is None:
                continue
                
            # Get neighbor's observation history
            neighbor_history = self.neighbor_provider.get_observation_history(
                neighbor_id, T
            )
            
            if neighbor_history is not None and len(neighbor_history) > 0:
                # Helper to extract feature vector from observation history element
                # Same logic as above: handle if neighbor history contains Dicts
                def extract_features(obs):
                    if isinstance(obs, dict):
                        if "self_features" in obs:
                            feat = obs["self_features"]
                            # If shape is [T, F], take last? Or is it [F]?
                            # Usually history items are individual time steps.
                            # If obs is a Dict from NeighborObs, it has 'self_features' as [T, F]
                            # BUT neighbor history should store raw per-step observations from BaseObs
                            # If it stores NeighborObs, we have recursion.
                            # ASSUMPTION: Neighbor provider returns the history of the NEIGHBOR'S
                            # observation function. If neighbor also uses NeighborObs, it returns Dicts.
                            # We need the RAW feature vector (density, queue, etc.)
                            
                            # If the neighbor uses NeighborTemporalObservationFunction, 
                            # its compute_observation() returns a Dict.
                            # We want the 'self_features' part of it, which represents its own state.
                            # 'self_features' in NeighborObs is [T, F] (history).
                            # We just want the most recent one? Or the whole history?
                            
                            # Wait, get_observation_history returns list of observations.
                            # If Neighbor uses NeighborObs, each item in history is a Dict.
                            # Inside that Dict, 'self_features' is ALREADY a history [T, F].
                            # This is redundant.
                            
                            # CORRECT APPROACH:
                            # The neighbor_provider should provide access to the neighbor's BASE features.
                            # But currently it likely calls get_observation_history() on the neighbor TS.
                            
                            # If neighbor supports 'get_lanes_density...' etc, we can reconstruct? No.
                            
                            # Simplification: extracting 'self_features' from the Dict
                            # If obs is {self: [T,F], ...}, we likely want the last step of self.
                            # But wait, history contains T steps.
                            # If each step is a Dict, it means we stored Dicts in history.
                            
                            # Let's look at TrafficSignal.compute_observation():
                            # It appends current_obs to history.
                            # If obs_fn returns Dict, history has Dicts.
                            
                            if "self_features" in obs: # It's a NeighborObs output
                                # It contains a history window [T, F]. 
                                # We probably just want the most recent frame from it?
                                # Or is this a single step obs that just happens to be shaped [1, F]?
                                val = obs["self_features"]
                                if val.ndim > 1: return val[-1] # Take last time step
                                return val
                        return np.concatenate([v.flatten() for v in obs.values()])
                    return np.asarray(obs, dtype=np.float32)

                try:
                    processed_history = [extract_features(obs) for obs in neighbor_history]
                    hist_array = np.array(processed_history, dtype=np.float32)
                    
                    # Ensure shape [T, feature_dim]
                    if hist_array.ndim != 2:
                        # Try to reshape if total elements match
                        if hist_array.size == T * feature_dim:
                            hist_array = hist_array.reshape(T, feature_dim)
                        else:
                            # Log warning or skip?
                            # Fallback: slice or pad
                            if hist_array.shape[0] > T: hist_array = hist_array[-T:]
                            # Dimension mismatch is hard to fix blindly
                            pass

                    hist_array = np.clip(hist_array, 0.0, 1.0)
                    
                    # Only assign if shapes match
                    if hist_array.shape == neighbor_features[i].shape:
                        neighbor_features[i] = hist_array
                        neighbor_mask[i] = 1.0
                except Exception:
                    pass # Skip this neighbor if parsing fails
                
        return neighbor_features, neighbor_mask

    def observation_space(self) -> spaces.Dict:
        """Return the Dict observation space.
        
        Returns:
            spaces.Dict with:
                - self_features: Box [T, feature_dim]
                - neighbor_features: Box [K, T, feature_dim]
                - neighbor_mask: Box [K]
                - action_mask: Box [8] for valid phases (FRAP)
        """
        # Keep Local-GNN feature dimension fixed across all intersections.
        # Observation values are constructed with TARGET_NUM_LANES (12) lane groups
        # + 8 green-time features, so the declared space must match exactly.
        # Using detector-count-dependent dims causes RLlib preprocessor shape errors
        # on heterogeneous networks (e.g., Zurich).
        feature_dim = 4 * TARGET_NUM_LANES + NUM_GREEN_TIME_FEATURES
        T = self.window_size
        K = self.max_neighbors
        
        return spaces.Dict({
            "self_features": spaces.Box(
                low=0.0, high=1.0, 
                shape=(T, feature_dim), 
                dtype=np.float32
            ),
            "neighbor_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(K, T, feature_dim),
                dtype=np.float32
            ),
            "neighbor_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(K,),
                dtype=np.float32
            ),
            "neighbor_directions": spaces.Box(
                low=-1.0, high=1.0,
                shape=(K,),
                dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0.0, high=1.0,
                shape=(NUM_STANDARD_PHASES,),
                dtype=np.float32
            ),
        })