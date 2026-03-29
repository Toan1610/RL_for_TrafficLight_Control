"""
MGMQ-PPO Evaluation Script.

This script evaluates trained MGMQ-PPO models on traffic signal control.
It loads a checkpoint and runs evaluation episodes, collecting metrics.

IMPORTANT: This script uses the same preprocessing configuration as training
to ensure the policy is applied correctly to the actual network.

Author: Bui Chi Toan
Date: 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
    register_sumo_env,
)
from src.models.mgmq_model import MGMQTorchModel, LocalMGMQTorchModel
from src.models.dirichlet_distribution import register_dirichlet_distribution
from src.models.masked_softmax_distribution import register_masked_softmax_distribution
from src.models.masked_multi_categorical import register_masked_multi_categorical
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_env_config,
    get_reward_config,
    get_action_config,
    get_network_config,
    is_local_gnn_enabled,
    load_training_config,
)


# Register custom models (same as training)
ModelCatalog.register_custom_model("mgmq_model", MGMQTorchModel)
ModelCatalog.register_custom_model("local_mgmq_model", LocalMGMQTorchModel)
# Backward compatibility: alias for old checkpoint trained with temporal naming
ModelCatalog.register_custom_model("local_temporal_mgmq_model", LocalMGMQTorchModel)

# Register Dirichlet distribution for action space (legacy)
register_dirichlet_distribution()

# Register Masked Softmax distribution (NEW - RECOMMENDED)
register_masked_softmax_distribution()

# Register Masked MultiCategorical distribution for discrete cycle adjustment
register_masked_multi_categorical()











def evaluate_mgmq(
    checkpoint_path: str,
    network_name: Optional[str] = None,
    num_episodes: int = 10,
    use_gui: bool = False,
    render: bool = False,
    output_file: str = None,
    seeds: List[int] = None,
    use_training_config: bool = True,
    config_path: Optional[str] = None,
    cycle_time_override: Optional[int] = None,
):
    """
    Evaluate a trained MGMQ-PPO model.
    
    Args:
        checkpoint_path: Path to checkpoint
        network_name: Network name
        num_episodes: Ignored if seeds is provided
        use_gui: Use SUMO GUI
        render: Render environment
        output_file: Output file for results
        seeds: List of eval seeds, one episode per seed. If None, auto-generated from num_episodes.
        use_training_config: Whether to load and use training configuration
        cycle_time_override: Optional cycle_time override for evaluation environment
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if seeds:
        seeds = list(seeds)
    else:
        seeds = [42 + i for i in range(num_episodes)]
    num_episodes = len(seeds)

    print("\n" + "="*80)
    print("MGMQ-PPO EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Network: {network_name}")
    print(f"Episodes: {num_episodes}  (seeds: {seeds})")
    print(f"Use Training Config: {use_training_config}")
    print("="*80 + "\n")
    
    # Initialize Ray with memory-efficient settings (same as training)
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(500e6),  # 500MB object store
        _memory=int(500e6),  # 500MB for tasks/actors
        include_dashboard=False,  # Disable dashboard to save memory
        _temp_dir=None,
        log_to_driver=True,  # Forward worker stdout/stderr to driver terminal
        logging_level="warning",  # Reduce Ray internal logs
    )
    
    try:
        # Convert relative path to absolute path for PyArrow compatibility
        checkpoint_path = str(Path(checkpoint_path).resolve())
        
        # Load training config if available (moved up to infer network name)
        training_config = None
        if use_training_config:
            training_config = load_training_config(checkpoint_path)
            
        # Infer network name from training config only when CLI did not provide --network
        if training_config and "network_name" in training_config and network_name is None:
            print(f"✓ Inferred network name from checkpoint: {training_config['network_name']}")
            network_name = training_config["network_name"]

        # Load YAML config for defaults
        yaml_config = load_model_config(config_path)
        yaml_env_cfg = get_env_config(yaml_config)
        yaml_reward_cfg = get_reward_config(yaml_config)
        yaml_action_cfg = get_action_config(yaml_config)
        yaml_mgmq_cfg = get_mgmq_config(yaml_config)

        # Fall back to YAML/default network only if still unspecified
        if network_name is None:
            network_name = yaml_config.get("network", {}).get("name", "grid4x4")
        
        # Get network configuration from YAML
        project_root = Path(__file__).parent.parent
        network_cfg = get_network_config(yaml_config, project_root)
        
        # Override with CLI network name (or inferred name)
        yaml_net_name = yaml_config.get("network", {}).get("name", "grid4x4")
        if network_name != yaml_net_name:
            print(f"⚠ Overriding network paths for {network_name}...")
            override_config = {"network": {"name": network_name}}
            network_cfg = get_network_config(override_config, project_root)
        
        # Use network config from YAML (or CLI override)
        net_file = network_cfg["net_file"]
        route_file = network_cfg["route_file"]
        preprocessing_config = network_cfg["intersection_config"]
        detector_file = network_cfg["detector_file"]
        network_name = network_cfg["network_name"]  # Update network_name from config

        # Validate network files
        if not Path(net_file).exists():
            raise FileNotFoundError(f"Network file not found: {net_file}")
        
        print(f"✓ Network: {network_name}")
        print(f"✓ Network file: {net_file}")
        print(f"✓ Route file: {route_file}")
        
        if preprocessing_config and Path(preprocessing_config).exists():
            print(f"✓ Preprocessing config: {preprocessing_config}")
            print("  (This ensures proper phase/intersection normalization)")
        else:
            preprocessing_config = None
            print("⚠ Warning: No preprocessing config found")
        
        # Get configured traffic signal IDs (used as reference only)
        ts_ids = get_network_ts_ids(network_name)
        
        # Build environment config - use training config if available
        if training_config and "env_config" in training_config:
            # Use stored env_config from training
            stored_env_config = training_config["env_config"]
            
            # FORCE OVERRIDE PATHS from local environment to fix Cloud vs Local path issues
            # We use network paths from YAML config (already resolved above)
            print(f"⚠ Overriding network paths in config to match local environment...")
            
            # Build additional SUMO command with detector file
            # Match training config (0.5s) and apply strict network settings
            additional_sumo_cmd = (
                "--step-length 1 "
                "--lateral-resolution 0.5 "
                "--ignore-route-errors "
                "--tls.actuated.jam-threshold 30 "
                "--no-internal-links true "
                "--device.rerouting.adaptation-steps 18 "
                "--device.rerouting.adaptation-interval 10"
            )
            if detector_file and Path(detector_file).exists():
                additional_sumo_cmd = f"-a {detector_file} {additional_sumo_cmd}"
            
            env_config = {
                "net_file": net_file,  # FROM YAML CONFIG
                "route_file": route_file,  # FROM YAML CONFIG
                "use_gui": use_gui,  # Override with current setting
                "virtual_display": None,  # Disable virtual display for local evaluation
                "render_mode": "human" if render else None,
                "num_seconds": int(stored_env_config.get("num_seconds", 8000)),
                "max_green": int(stored_env_config.get("max_green", 90)),
                "min_green": int(stored_env_config.get("min_green", 5)),
                "cycle_time": int(stored_env_config.get("cycle_time", stored_env_config.get("delta_time", 90))),
                "yellow_time": int(stored_env_config.get("yellow_time", 3)),
                # Keep teleport setting consistent with training config for fair comparison
                "time_to_teleport": int(
                    stored_env_config.get("time_to_teleport", yaml_env_cfg.get("time_to_teleport", 500))
                ),
                "single_agent": False,
                "window_size": int(stored_env_config.get("window_size", 1)),
                "preprocessing_config": preprocessing_config, # FROM YAML CONFIG
                "additional_sumo_cmd": additional_sumo_cmd, # FROM YAML CONFIG
                "reward_fn": stored_env_config.get("reward_fn", yaml_reward_cfg["reward_fn"]),
                "reward_weights": stored_env_config.get("reward_weights", yaml_reward_cfg["reward_weights"]),
                "use_phase_standardizer": stored_env_config.get("use_phase_standardizer", yaml_env_cfg["use_phase_standardizer"]),
                "green_time_step": int(stored_env_config.get("green_time_step", yaml_action_cfg.get("green_time_step", 5))),
                "use_neighbor_obs": stored_env_config.get("use_neighbor_obs", is_local_gnn_enabled(yaml_config)),
                "max_neighbors": stored_env_config.get("max_neighbors", yaml_mgmq_cfg["max_neighbors"]),
                # EVALUATION: Disable normalization so we get RAW rewards
                # This ensures fair comparison with baseline (which also uses raw rewards)
                # The normalized reward column is no longer meaningful for eval
                "normalize_reward": False,
                "clip_rewards": None,
                # Action mode from training config or YAML
                "action_mode": stored_env_config.get("action_mode", yaml_action_cfg["action_mode"]),
            }
            print("\n✓ Using environment config from training:")
            print(f"  num_seconds: {env_config['num_seconds']}")
            print(f"  cycle_time: {env_config['cycle_time']}")
            print(f"  reward_fn: {env_config['reward_fn']}")
            print(f"  reward_weights: {env_config['reward_weights']}")
            print(f"  window_size: {env_config['window_size']}")
            print(f"  use_phase_standardizer: {env_config['use_phase_standardizer']}")
            print(f"  use_neighbor_obs: {env_config['use_neighbor_obs']}")

        else:
            # Build additional SUMO command with detector file
            # Match training config (0.5s) and apply strict network settings
            additional_sumo_cmd = (
                "--step-length 1 "
                "--lateral-resolution 0.5 "
                "--ignore-route-errors "
                "--tls.actuated.jam-threshold 30 "
                "--device.rerouting.adaptation-steps 18 "
                "--device.rerouting.adaptation-interval 10"
            )
            if detector_file and Path(detector_file).exists():
                additional_sumo_cmd = f"-a {detector_file} {additional_sumo_cmd}"
            
            # Use default config from YAML
            env_config = {
                "net_file": net_file,  # FROM YAML CONFIG
                "route_file": route_file,  # FROM YAML CONFIG
                "use_gui": use_gui,
                "virtual_display": None,
                "render_mode": "human" if render else None,
                "num_seconds": yaml_env_cfg["num_seconds"],
                "max_green": yaml_env_cfg["max_green"],
                "min_green": yaml_env_cfg["min_green"],
                "cycle_time": yaml_env_cfg["cycle_time"],
                "yellow_time": yaml_env_cfg["yellow_time"],
                "time_to_teleport": yaml_env_cfg.get("time_to_teleport", 500),
                "single_agent": False,
                "window_size": yaml_mgmq_cfg.get("window_size", 1),
                "preprocessing_config": preprocessing_config,
                "additional_sumo_cmd": additional_sumo_cmd,
                "reward_fn": yaml_reward_cfg["reward_fn"],
                "reward_weights": yaml_reward_cfg["reward_weights"],
                "use_phase_standardizer": yaml_env_cfg.get("use_phase_standardizer", True),
                "green_time_step": int(yaml_action_cfg.get("green_time_step", 5)),
                "use_neighbor_obs": is_local_gnn_enabled(yaml_config),
                "max_neighbors": yaml_mgmq_cfg.get("max_neighbors", 4),
                # EVALUATION: Disable normalization so we get RAW rewards
                # This ensures fair comparison with baseline (which also uses raw rewards)
                "normalize_reward": False,
                "clip_rewards": None,
                # Action mode from YAML
                "action_mode": yaml_action_cfg["action_mode"],
            }
            print("\n✓ Using environment config from YAML defaults")

        # Allow explicit eval-only cycle override while keeping other config intact.
        if cycle_time_override is not None:
            env_config["cycle_time"] = int(cycle_time_override)
            print(f"\n⚠ Overriding cycle_time for evaluation: {env_config['cycle_time']}")
        
        print("")
        
        # Register environment with MultiAgentEnv wrapper
        register_sumo_env(env_config)
        
        # Load algorithm from checkpoint
        print("Loading trained model from checkpoint...")
        try:
            # Try efficient Policy loading first (faster, ignores worker config)
            # This works best if we don't need the full EnvRunner/Worker setup
            # However, compute_single_action expects Policy or Algo. 
            # Let's try full Algo load first, but catch the specific GPU error.
            if torch.cuda.is_available():
                algo = PPO.from_checkpoint(checkpoint_path)
            else:
                # If no GPU, force CPU load by modifying config
                raise RuntimeError("Force CPU fallback")
                
        except Exception as e:
            print(f"⚠ Standard load failed or CPU forced: {e}")
            print("↺ Attempting to reconstruct Algorithm with CPU configuration...")
            
            # Find config file
            checkpoint_dir = Path(checkpoint_path)
            # Checkpoint structure: /path/to/experiment/PPO_.../checkpoint_000000
            params_path = checkpoint_dir.parent / "params.json"
            
            if not params_path.exists():
                print(f"❌ Could not find params.json at {params_path}")
                raise e
            
            with open(params_path, "r") as f:
                config = json.load(f)
            
            # Extract only the essential config we need
            model_config = config.get("model", {})
            
            print("⚠ Building fresh PPOConfig with OLD API stack (ModelV2)...")
            
            from ray.rllib.algorithms.ppo import PPOConfig
            
            # Create a FRESH config - don't use from_dict which brings in problematic values
            algo_config = PPOConfig()
            
            # CRITICAL: Disable new API stack FIRST before anything else
            algo_config.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False
            )
            
            # Set environment
            algo_config.environment(env="sumo_mgmq_v0", env_config=env_config)
            
            # Set resources for CPU
            algo_config.resources(num_gpus=0)
            algo_config.env_runners(num_env_runners=0)
            
            # Set framework
            algo_config.framework(config.get("framework", "torch"))
            
            # Set model config (custom_model is in here)
            algo_config.training(model=model_config)
            
            # For multi-agent: Let RLlib infer policies from environment
            # Just set the policy_mapping_fn. RLlib will create policies based on env.
            algo_config.multi_agent(
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy",
            )
            
            # CRITICAL FIX: Disable normalize_actions
            # Both MaskedSoftmax and MaskedMultiCategorical output raw actions
            algo_config.normalize_actions = False
            
            # Build the algorithm
            algo = algo_config.build()
            
            # Restore weights from checkpoint
            print(f"  Restoring weights from {checkpoint_path}...")
            algo.restore(checkpoint_path)
            
        print("✓ Model loaded successfully\n")
        
        # Create evaluation environment
        env = SumoMultiAgentEnv(**env_config)

        # Use ACTUAL active agents from env (some configured IDs may be skipped, e.g. no E2)
        active_ts_ids = list(env.ts_ids)
        if len(active_ts_ids) != len(ts_ids):
            print(
                f"⚠ Active agents in env: {len(active_ts_ids)} (configured: {len(ts_ids)}). "
                "Using active agents for eval metrics."
            )
        
        # Evaluation metrics
        episode_rewards = []  # normalized (what training sees)
        episode_raw_rewards = []  # raw (actual traffic performance)
        episode_lengths = []
        episode_waiting_times = []
        episode_avg_speeds = []
        episode_total_halts = []
        episode_throughputs = []
        episode_mean_pressures = []
        per_agent_rewards = {ts_id: [] for ts_id in active_ts_ids}
        per_agent_raw_rewards = {ts_id: [] for ts_id in active_ts_ids}
        
        for ep, eval_seed in enumerate(seeds):
            obs, info = env.reset(seed=eval_seed)
            done = {"__all__": False}
            total_reward = 0
            total_raw_reward = 0
            agent_rewards = {ts_id: 0 for ts_id in active_ts_ids}
            agent_raw_rewards = {ts_id: 0 for ts_id in active_ts_ids}
            step_count = 0
            
            while not done.get("__all__", False):
                # Get actions from policy for all agents
                actions = {}
                if not obs:  # Safety check for empty observation
                    break
                
                for agent_id in obs.keys():
                    action = algo.compute_single_action(
                        obs[agent_id],
                        policy_id="default_policy"
                    )
                    actions[agent_id] = action
                    
                    # LOGGING: Monitor action dominance
                    # Only log A0 every 10 steps to reduce clutter
                    if active_ts_ids and agent_id == active_ts_ids[0] and step_count % 10 == 0:
                        obs_dict = obs[agent_id]
                        is_discrete = env_config.get("action_mode", "ratio") == "discrete_adjustment"
                        if is_discrete:
                            # Discrete actions: {0: -step, 1: keep, 2: +step}
                            action_labels = {0: "-", 1: "=", 2: "+"}
                            action_str = " ".join(action_labels.get(int(a), "?") for a in action)
                            print(f"[{agent_id} Step {step_count}] Discrete: [{action_str}]")
                        elif isinstance(obs_dict, dict) and "action_mask" in obs_dict:
                            mask = obs_dict["action_mask"]
                            valid_phases = action[mask > 0.5]
                            masked_phases = action[mask < 0.5]
                            print(f"[{agent_id} Step {step_count}] Valid phases: {np.round(valid_phases, 3)} "
                                  f"(sum={valid_phases.sum():.3f}), Masked: {np.round(masked_phases, 3)}")
                            valid_action = action[action > 0.01] if len(action[action > 0.01]) > 0 else action
                            if np.std(valid_action) < 0.03:
                                print(f"    ⚠ Uniform Policy (Std={np.std(valid_action):.4f})")
                            else:
                                print(f"    ✓ Differentiated (Std={np.std(valid_action):.4f}, Max/Min={valid_action.max():.3f}/{valid_action.min():.3f})")
                        else:
                            print(f"[{agent_id} Step {step_count}] Action: {np.round(action, 3)}")

                # Step environment
                obs, rewards, terminateds, truncateds, info = env.step(actions)
                
                # Accumulate normalized rewards (what training sees)
                for agent_id, reward in rewards.items():
                    total_reward += reward
                    if agent_id in agent_rewards:
                        agent_rewards[agent_id] += reward
                
                # Accumulate raw rewards from info dict
                for agent_id in rewards.keys():
                    agent_info = info.get(agent_id, {})
                    if isinstance(agent_info, dict) and "raw_reward" in agent_info:
                        raw_r = agent_info["raw_reward"]
                    else:
                        raw_r = rewards[agent_id]  # fallback if no raw_reward in info
                    total_raw_reward += raw_r
                    if agent_id in agent_raw_rewards:
                        agent_raw_rewards[agent_id] += raw_r
                
                step_count += 1
                
                # Check if episode is done
                done = truncateds
            
            episode_rewards.append(total_reward)
            episode_raw_rewards.append(total_raw_reward)
            episode_lengths.append(step_count)
            
            # Store per-agent rewards
            for ts_id in active_ts_ids:
                if ts_id in agent_rewards:
                    per_agent_rewards[ts_id].append(agent_rewards[ts_id])
                if ts_id in agent_raw_rewards:
                    per_agent_raw_rewards[ts_id].append(agent_raw_rewards[ts_id])
            
            # Get system metrics from top-level info dict
            if "system_total_waiting_time" in info:
                episode_waiting_times.append(info["system_total_waiting_time"])
            if "system_mean_speed" in info:
                episode_avg_speeds.append(info["system_mean_speed"])
            if "system_total_stopped" in info:
                episode_total_halts.append(info["system_total_stopped"])
            if "system_throughput" in info:
                episode_throughputs.append(info["system_throughput"])
            if "system_mean_pressure" in info:
                episode_mean_pressures.append(info["system_mean_pressure"])
            
            print(f"Episode {ep+1}/{num_episodes} (seed={eval_seed}): Raw Reward={total_raw_reward:.2f}, Normalized={total_reward:.2f}, Steps={step_count}")
        
        env.close()
        
        # Calculate statistics
        results = {
            "checkpoint": checkpoint_path,
            "network": network_name,
            "num_episodes": num_episodes,
            "eval_seeds": seeds,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_raw_reward": float(np.mean(episode_raw_rewards)),
            "std_raw_reward": float(np.std(episode_raw_rewards)),
            "min_raw_reward": float(np.min(episode_raw_rewards)),
            "max_raw_reward": float(np.max(episode_raw_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "episode_rewards": [float(r) for r in episode_rewards],
            "episode_raw_rewards": [float(r) for r in episode_raw_rewards],
            "episode_lengths": [int(l) for l in episode_lengths],
        }
        
        # Per-agent statistics
        per_agent_stats = {}
        for ts_id in active_ts_ids:
            stats = {}
            if per_agent_rewards[ts_id]:
                stats["mean_reward"] = float(np.mean(per_agent_rewards[ts_id]))
                stats["std_reward"] = float(np.std(per_agent_rewards[ts_id]))
            if per_agent_raw_rewards[ts_id]:
                stats["mean_raw_reward"] = float(np.mean(per_agent_raw_rewards[ts_id]))
                stats["std_raw_reward"] = float(np.std(per_agent_raw_rewards[ts_id]))
            if stats:
                per_agent_stats[ts_id] = stats
        results["per_agent_stats"] = per_agent_stats
        
        if episode_waiting_times:
            results["mean_waiting_time"] = float(np.mean(episode_waiting_times))
            results["std_waiting_time"] = float(np.std(episode_waiting_times))
        
        if episode_avg_speeds:
            results["mean_avg_speed"] = float(np.mean(episode_avg_speeds))
            results["std_avg_speed"] = float(np.std(episode_avg_speeds))
        
        if episode_total_halts:
            results["mean_total_halts"] = float(np.mean(episode_total_halts))
            results["std_total_halts"] = float(np.std(episode_total_halts))
        
        if episode_throughputs:
            results["mean_throughput"] = float(np.mean(episode_throughputs))
            results["std_throughput"] = float(np.std(episode_throughputs))
        
        if episode_mean_pressures:
            results["mean_pressure"] = float(np.mean(episode_mean_pressures))
            results["std_pressure"] = float(np.std(episode_mean_pressures))
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"\n  🎯 RAW Reward (actual traffic performance):")
        print(f"     Mean: {results['mean_raw_reward']:.2f} ± {results['std_raw_reward']:.2f}")
        print(f"     Min/Max: {results['min_raw_reward']:.2f} / {results['max_raw_reward']:.2f}")
        print(f"\n  📊 Normalized Reward (what training sees):")
        print(f"     Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"     Min/Max: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
        print(f"\n  📏 Episode Length: {results['mean_length']:.1f}")
        
        if "mean_waiting_time" in results:
            print(f"  ⏱ Mean Waiting Time: {results['mean_waiting_time']:.2f} ± {results.get('std_waiting_time', 0):.2f}")
        if "mean_avg_speed" in results:
            print(f"  🚗 Mean Average Speed: {results['mean_avg_speed']:.2f} ± {results.get('std_avg_speed', 0):.2f}")
        if "mean_total_halts" in results:
            print(f"  🛑 Mean Total Halts: {results['mean_total_halts']:.2f} ± {results.get('std_total_halts', 0):.2f}")
        if "mean_throughput" in results:
            print(f"  🚀 Mean Throughput: {results['mean_throughput']:.0f} ± {results.get('std_throughput', 0):.0f}")
        if "mean_pressure" in results:
            print(f"  ⚖ Mean Pressure: {results['mean_pressure']:.4f} ± {results.get('std_pressure', 0):.4f}")
        
        print("\n  Per-Agent Rewards (raw / normalized):")
        for ts_id, stats in per_agent_stats.items():
            raw_str = f"{stats.get('mean_raw_reward', 0):.2f}" if 'mean_raw_reward' in stats else "N/A"
            norm_str = f"{stats.get('mean_reward', 0):.2f}" if 'mean_reward' in stats else "N/A"
            print(f"    {ts_id}: raw={raw_str}, norm={norm_str}")
        
        print("="*80 + "\n")
        
        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results saved to: {output_file}")
        
        return results
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MGMQ-PPO model on traffic signal control"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--network", type=str, default=None,
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc", "test"],
                        help="Network name (if omitted, infer from checkpoint training config)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    parser.add_argument("--seeds", type=int, nargs='+', default=None,
                        help="Evaluation seeds, one episode per seed. If omitted, auto-generate from --episodes.")
    parser.add_argument("--no-training-config", action="store_true",
                        help="Do not load training config, use defaults")
    parser.add_argument("--no-tranning-config", dest="no_training_config", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml (default: src/config/model_config.yml)")
    parser.add_argument("--cycle-time", type=int, default=None,
                        help="Override environment cycle_time only for evaluation")
    
    args = parser.parse_args()
    
    evaluate_mgmq(
        checkpoint_path=args.checkpoint,
        network_name=args.network,
        num_episodes=args.episodes,
        use_gui=args.gui,
        render=args.render,
        output_file=args.output,
        seeds=args.seeds,
        use_training_config=not args.no_training_config,
        config_path=args.config,
        cycle_time_override=args.cycle_time,
    )
