"""Baseline reward evaluation for fixed-time or MaxPressure controllers.

This script evaluates non-RL baselines using the SAME environment/reward setup
as training and RL evaluation. It supports baseline modes:

- ``fixed``: SUMO default traffic light program (fixed-time)
- ``max_pressure_native``: MP original controller imported from ``mp/src``
- ``max_pressure_legacy``: legacy in-script controller (for A/B checks)
- ``max_pressure``: backward-compat alias of ``max_pressure_native``
"""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.rllib_utils import (
    SumoMultiAgentEnv,
    get_network_ts_ids,
)
from src.config import (
    load_model_config,
    get_mgmq_config,
    get_env_config,
    get_action_config,
    get_reward_config,
    get_network_config,
)


def _resolve_eval_seeds(num_episodes: int, seeds: Optional[List[int]]) -> List[int]:
    """Resolve evaluation seeds from CLI args."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if seeds:
        return list(seeds)
    return [42 + i for i in range(num_episodes)]


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_mp_native_maxpressure_class(project_root: Path):
    """Load MP original MaxPressure class from mp/src as a submodule of root src package."""
    try:
        import src as root_src
    except Exception as exc:
        raise ImportError("Could not import root 'src' package before loading MP native controller") from exc

    controller_pkg_name = "src.controller"
    controller_init = project_root / "mp" / "src" / "controller" / "__init__.py"
    if not controller_init.exists():
        raise FileNotFoundError(f"MP controller package not found: {controller_init}")

    if controller_pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            controller_pkg_name,
            controller_init,
            submodule_search_locations=[str(controller_init.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build import spec for {controller_pkg_name}")
        controller_module = importlib.util.module_from_spec(spec)
        sys.modules[controller_pkg_name] = controller_module
        setattr(root_src, "controller", controller_module)
        spec.loader.exec_module(controller_module)

    # Imported after src.controller exists.
    from src.controller.maxpressure.max_pressure import MaxPressure  # type: ignore

    return MaxPressure


class _MPNativeIfaceAdapter:
    """Adapter that exposes MP-native traci-like methods on top of RL simulator."""

    def __init__(self, simulator: Any):
        self.simulator = simulator
        if getattr(simulator, "sumo", None) is None:
            raise RuntimeError("Simulator SUMO handle is not available")

    def _sumo(self):
        # SUMO connection object is replaced on env reset, so always resolve lazily.
        sumo = getattr(self.simulator, "sumo", None)
        if sumo is None:
            raise RuntimeError("Simulator SUMO handle is not available")
        return sumo

    def get_lanearea_occupancy(self, detector_id: str) -> float:
        return float(self.simulator.get_detector_occupancy(detector_id))

    def get_tls_splits(self, tls_id: str):
        return self._sumo().trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]

    def set_tls_splits(self, tls_id: str, splits):
        self._sumo().trafficlight.setCompleteRedYellowGreenDefinition(tls_id, splits)


class MaxPressureNativeBaselineController:
    """Run MP original MaxPressure controller inside RL evaluation loop."""

    def __init__(
        self,
        simulator: Any,
        net_info: dict,
        active_ts_ids: List[str],
        project_root: Path,
        cycling: str = "exponential",
        sample_interval: float = 10.0,
    ):
        MaxPressure = _load_mp_native_maxpressure_class(project_root)

        self.simulator = simulator
        self.net_info = net_info
        self.iface = _MPNativeIfaceAdapter(simulator)
        self.sample_interval = max(1, int(round(float(sample_interval))))
        self.next_sample_time = float(self.simulator.get_sim_time()) + float(self.sample_interval)

        self.controllers: Dict[str, Any] = {}
        self.cycle_times: Dict[str, float] = {}
        self.triggers: Dict[str, float] = {}
        self.cache_edges_occupancy: Dict[str, List[float]] = {}
        self.controlled_edges: Dict[str, dict] = {}

        tls_dict = (net_info or {}).get("tls", {})
        for ts_id in active_ts_ids:
            tls_info = tls_dict.get(ts_id)
            if not tls_info:
                continue
            ctrl = MaxPressure(
                tls_id=ts_id,
                iface=self.iface,
                tls_info=tls_info,
                sample_interval=float(sample_interval),
                cycling=cycling,
            )
            ctrl.start()
            self.controllers[ts_id] = ctrl

            cycle = float(tls_info.get("cycle", 0.0))
            if cycle <= 0:
                cycle = float(getattr(simulator.traffic_signals.get(ts_id), "delta_time", 90.0))
            self.cycle_times[ts_id] = cycle

            t_now = float(self.simulator.get_sim_time())
            self.triggers[ts_id] = t_now + cycle

            for edge_id, edge_info in (tls_info.get("edges") or {}).items():
                if edge_id not in self.controlled_edges:
                    self.controlled_edges[edge_id] = edge_info

    def _collect_data(self) -> None:
        for edge_id, edge_info in self.controlled_edges.items():
            occ_values = []
            detectors = (edge_info or {}).get("detector") or []
            if not detectors:
                occ_values.append(0.0)
            for det_id in detectors:
                try:
                    occ_values.append(float(self.iface.get_lanearea_occupancy(det_id)))
                except Exception:
                    continue
            mean_occ = float(np.mean(occ_values)) if occ_values else 0.0
            series = self.cache_edges_occupancy.setdefault(edge_id, [])
            series.append(mean_occ)
            if len(series) > 20:
                self.cache_edges_occupancy[edge_id] = series[-20:]

    def reset_runtime(self) -> None:
        """Reset per-episode runtime state after env.reset() recreates SUMO connection."""
        t_now = float(self.simulator.get_sim_time())
        self.next_sample_time = t_now + float(self.sample_interval)
        self.cache_edges_occupancy = {}
        self.triggers = {ts_id: t_now + cycle for ts_id, cycle in self.cycle_times.items()}

        for ctrl in self.controllers.values():
            ctrl.start()
            if hasattr(ctrl, "_calculate_lost_time"):
                try:
                    ctrl.lost_time = ctrl._calculate_lost_time()
                except Exception:
                    pass

    def apply_for_ready_signals(self, traffic_signals: Dict[str, Any]) -> int:
        applied = 0
        t_now = float(self.simulator.get_sim_time())

        # Reconstruct true historical occupancy from TrafficSignal objects
        # instead of error-prone point-in-time sampling catch-up
        for edge_id, edge_info in self.controlled_edges.items():
            detectors = (edge_info or {}).get("detector") or []
            
            edge_series = None
            if detectors:
                # Find the traffic signal that owns this edge's detectors
                owner_ts = None
                for ts in traffic_signals.values():
                    if detectors[0] in getattr(ts, 'detectors_e2', []):
                        owner_ts = ts
                        break
                        
                if owner_ts and hasattr(owner_ts, 'detector_history'):
                    # Extract history for all detectors on this edge and average them
                    hist_len = len(owner_ts.detector_history.get("occupancy", {}).get(detectors[0], []))
                    if hist_len > 0:
                        edge_series = []
                        for i in range(hist_len):
                            step_vals = []
                            for d in detectors:
                                d_hist = owner_ts.detector_history.get("occupancy", {}).get(d, [])
                                if i < len(d_hist):
                                    step_vals.append(d_hist[i] * 100.0)  # scale back to [0..100] for original MP
                            edge_series.append(float(np.mean(step_vals)) if step_vals else 0.0)
            
            if edge_series is None:
                edge_series = [0.0]
                
            self.cache_edges_occupancy[edge_id] = edge_series

        while t_now >= self.next_sample_time:
            self.next_sample_time += float(self.sample_interval)

        for ts_id, ts in traffic_signals.items():
            if ts_id not in self.controllers or not ts.time_to_act:
                continue
            due_time = self.triggers.get(ts_id, t_now)
            if t_now < due_time:
                continue

            cycle = self.cycle_times.get(ts_id, 90.0)
            n_samples = max(1, int(round(cycle / float(self.sample_interval))))
            next_cycle = self.controllers[ts_id].action(t_now, self.cache_edges_occupancy, n_samples)
            self.triggers[ts_id] = float(t_now) + float(next_cycle)
            applied += 1

        return applied


def _discover_mp_net_info(
    project_root: Path,
    network_name: str,
    active_ts_ids: List[str],
    explicit_path: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[dict]]:
    """Find the best matching net-info.json for MaxPressure controller."""
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            raise FileNotFoundError(f"MP net-info not found: {candidate}")
        return candidate, _load_json(candidate)

    mp_data_root = project_root / "mp" / "data"
    if not mp_data_root.exists():
        return None, None

    active_set = set(active_ts_ids)
    candidates = list(mp_data_root.rglob("net-info.json"))
    if not candidates:
        return None, None

    best_score = None
    best_path = None
    best_data = None
    name_key = (network_name or "").lower()

    for path in candidates:
        try:
            data = _load_json(path)
        except Exception:
            continue
        tls_ids = set((data.get("tls") or {}).keys())
        overlap = len(active_set.intersection(tls_ids))
        if overlap <= 0:
            continue

        name_bonus = 1 if name_key and name_key in str(path).lower() else 0
        score = (overlap, name_bonus, -len(str(path)))
        if best_score is None or score > best_score:
            best_score = score
            best_path = path
            best_data = data

    return best_path, best_data


class MaxPressureLegacyBaselineController:
    """Legacy MaxPressure baseline executor (in-script implementation)."""

    def __init__(self, simulator: Any, net_info: dict, active_ts_ids: List[str], cycling: str = "exponential"):
        self.simulator = simulator
        self.cycling = cycling if cycling in {"linear", "exponential"} else "exponential"
        self.tls_cfg: Dict[str, Dict[str, Any]] = {}

        tls_info = net_info.get("tls", {})
        for ts_id in active_ts_ids:
            cfg = tls_info.get(ts_id)
            if not cfg:
                continue
            phase_defs = cfg.get("phases", {})
            if not phase_defs:
                continue

            phase_ids = sorted(int(k) for k in phase_defs.keys())
            min_green = []
            max_green = []
            for pid in phase_ids:
                phase_cfg = phase_defs.get(str(pid), {})
                min_green.append(float(phase_cfg.get("min-green", 1.0)))
                max_green.append(float(phase_cfg.get("max-green", 120.0)))

            self.tls_cfg[ts_id] = {
                "cfg": cfg,
                "phase_ids": phase_ids,
                "min_green": min_green,
                "max_green": max_green,
            }

    def _edge_occupancy(self, edge_info: dict) -> float:
        detectors = edge_info.get("detector") or []
        if not detectors:
            return 0.0
        occ_values = []
        for det_id in detectors:
            try:
                occ_values.append(float(self.simulator.get_detector_occupancy(det_id)))
            except Exception:
                continue
        if not occ_values:
            return 0.0
        return float(np.clip(np.mean(occ_values) / 100.0, 0.0, 1.0))

    def _compute_phase_pressures(self, tls_cfg: dict) -> Dict[int, float]:
        cfg = tls_cfg["cfg"]
        phase_ids = tls_cfg["phase_ids"]

        edges_cfg = cfg.get("edges", {})
        movements_cfg = cfg.get("movements", {})

        edge_occ = {edge_id: self._edge_occupancy(edge_info) for edge_id, edge_info in edges_cfg.items()}

        movements_pressure: Dict[str, float] = {}
        for from_edge, movements_data in movements_cfg.items():
            out_pressure = 0.0
            for out_edge, ratio in (movements_data or {}).items():
                out_pressure += edge_occ.get(out_edge, 0.0) * float(ratio)
            sat_flow = float(edges_cfg.get(from_edge, {}).get("sat_flow", 1.0))
            in_pressure = edge_occ.get(from_edge, 0.0)
            movements_pressure[from_edge] = (in_pressure - out_pressure) * sat_flow

        phase_pressures: Dict[int, float] = {}
        for phase_id in phase_ids:
            phase_cfg = cfg.get("phases", {}).get(str(phase_id), {})
            pressure = 0.0
            for movement in phase_cfg.get("movements", []):
                if not isinstance(movement, list) or len(movement) < 2:
                    continue
                from_edge, to_edge = movement[0], movement[1]
                if from_edge in movements_pressure:
                    ratio = float(movements_cfg.get(from_edge, {}).get(to_edge, 0.0))
                    pressure += movements_pressure[from_edge] * ratio
            phase_pressures[phase_id] = max(pressure, 0.0)
        return phase_pressures

    @staticmethod
    def _constrain_times(raw_times: List[float], target_sum: int, min_times: List[float], max_times: List[float]) -> List[int]:
        n = len(raw_times)
        if n == 0:
            return []
        if target_sum <= 0:
            return [1] * n

        mins = [max(0, int(round(v))) for v in min_times]
        maxs = [max(mins[i], int(round(max_times[i]))) for i in range(n)]

        min_sum = sum(mins)
        max_sum = sum(maxs)
        if min_sum > target_sum:
            # Relax minima proportionally to keep feasibility.
            scaled = [max(1, int(round(m / min_sum * target_sum))) if min_sum > 0 else 1 for m in mins]
            mins = scaled
            min_sum = sum(mins)
        if max_sum < target_sum:
            target_sum = max_sum

        total_raw = float(sum(raw_times))
        if total_raw <= 1e-8:
            raw_times = [1.0] * n
            total_raw = float(n)

        times = []
        for i in range(n):
            value = int(round(raw_times[i] / total_raw * target_sum))
            value = max(mins[i], min(maxs[i], value))
            times.append(value)

        diff = target_sum - sum(times)
        attempts = 0
        while diff != 0 and attempts < max(1, target_sum * 2):
            if diff > 0:
                for i in range(n):
                    if diff <= 0:
                        break
                    if times[i] < maxs[i]:
                        times[i] += 1
                        diff -= 1
            else:
                for i in range(n):
                    if diff >= 0:
                        break
                    if times[i] > mins[i]:
                        times[i] -= 1
                        diff += 1
            attempts += 1
        return times

    def _compute_green_times(self, ts_id: str, ts_obj: Any) -> Optional[List[int]]:
        tls_cfg = self.tls_cfg.get(ts_id)
        if tls_cfg is None:
            return None

        phase_ids = tls_cfg["phase_ids"]
        phase_pressures = self._compute_phase_pressures(tls_cfg)
        pressures = [phase_pressures.get(pid, 0.0) for pid in phase_ids]
        pressure_sum = float(sum(pressures))

        target_sum = int(round(float(getattr(ts_obj, "total_green_time", 0))))
        if target_sum <= 0:
            target_sum = max(1, len(phase_ids))

        if pressure_sum <= 1e-8:
            raw = [1.0 for _ in phase_ids]
        elif self.cycling == "linear":
            raw = pressures
        else:
            mean_pressure = pressure_sum / max(1, len(pressures))
            if mean_pressure <= 1e-8:
                raw = pressures
            else:
                raw = [float(np.exp(p / mean_pressure)) for p in pressures]

        green_times = self._constrain_times(raw, target_sum, tls_cfg["min_green"], tls_cfg["max_green"])

        # Ensure shape compatibility with TrafficSignal in this environment.
        if len(green_times) != int(getattr(ts_obj, "num_green_phases", len(green_times))):
            n = int(getattr(ts_obj, "num_green_phases", len(green_times)))
            if len(green_times) > n:
                green_times = green_times[:n]
            else:
                green_times = green_times + [int(getattr(ts_obj, "min_green", 5))] * (n - len(green_times))
            green_times = self._constrain_times(
                green_times,
                target_sum=int(round(float(getattr(ts_obj, "total_green_time", target_sum)))),
                min_times=[float(getattr(ts_obj, "min_green", 5))] * n,
                max_times=[float(getattr(ts_obj, "max_green", 120))] * n,
            )

        return [int(v) for v in green_times]

    def apply_for_ready_signals(self, traffic_signals: Dict[str, Any]) -> int:
        """Apply MaxPressure split update for traffic signals that are due to act."""
        applied = 0
        for ts_id, ts in traffic_signals.items():
            if ts_id not in self.tls_cfg or not ts.time_to_act:
                continue
            green_times = self._compute_green_times(ts_id, ts)
            if not green_times:
                continue
            self.simulator.set_traffic_light_phase(ts_id, green_times)
            ts.green_times = green_times
            applied += 1
        return applied


def evaluate_baseline(
    network_name: Optional[str] = None,
    num_episodes: int = 5,
    use_gui: bool = False,
    render: bool = False,
    output_file: str = None,
    seeds: list = None,
    config_path: Optional[str] = None,
    controller: str = "max_pressure_native",
    mp_net_info: Optional[str] = None,
):
    """
    Evaluate baseline (no AI) traffic signal control.
    
    Uses the SAME environment as training/evaluation, but without RL policy.
    Controller can be fixed-time, MP native, or legacy in-script MaxPressure.
    
    Args:
        network_name: Network name (grid4x4, zurich, etc.)
        num_episodes: Ignored if seeds is provided
        use_gui: Use SUMO GUI for visualization
        render: Render environment
        output_file: Output file for results
        seeds: List of eval seeds, one episode per seed. If None, generated from num_episodes.
        config_path: Path to model_config.yml
        controller: Baseline controller ("fixed", "max_pressure_native", "max_pressure_legacy")
        mp_net_info: Optional explicit path to MP net-info.json
    """
    seeds = _resolve_eval_seeds(num_episodes, seeds)
    num_episodes = len(seeds)

    print("\n" + "="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    print(f"Network: {network_name}")
    print(f"Controller: {controller}")
    print(f"Episodes: {num_episodes}  (seeds: {seeds})")
    print(f"Config: {config_path or 'default (src/config/model_config.yml)'}")
    print("="*80 + "\n")
    
    # Load YAML config (same as training/eval_mgmq_ppo.py)
    yaml_config = load_model_config(config_path)
    yaml_env_cfg = get_env_config(yaml_config)
    yaml_reward_cfg = get_reward_config(yaml_config)
    yaml_mgmq_cfg = get_mgmq_config(yaml_config)
    yaml_action_cfg = get_action_config(yaml_config)

    # Use YAML default network only when CLI did not provide --network
    if network_name is None:
        network_name = yaml_config.get("network", {}).get("name", "grid4x4")
    
    # Get network configuration from YAML
    project_root = Path(__file__).parent.parent
    network_cfg = get_network_config(yaml_config, project_root)

    # Override with CLI network name if different.
    # IMPORTANT: Do not reuse net_file/route_files from YAML when switching network,
    # otherwise we may end up with mismatched paths like:
    #   network/zurich/PhuQuoc.net.xml
    yaml_net_name = yaml_config.get("network", {}).get("name", "grid4x4")
    if network_name != yaml_net_name:
        print(f"Warning: Overriding network paths for {network_name}...")
        override_cfg = {"network": {"name": network_name}}
        network_cfg = get_network_config(override_cfg, project_root)
    
    # Get network files
    net_file = network_cfg["net_file"]
    route_file = network_cfg["route_file"]
    preprocessing_config = network_cfg["intersection_config"]
    detector_file = network_cfg["detector_file"]
    network_name = network_cfg["network_name"]
    
    # Validate network files
    if not Path(net_file).exists():
        raise FileNotFoundError(f"Network file not found: {net_file}")
    
    print(f"[OK] Network: {network_name}")
    print(f"[OK] Network file: {net_file}")
    print(f"[OK] Route file: {route_file}")
    
    if preprocessing_config and Path(preprocessing_config).exists():
        print(f"[OK] Preprocessing config: {preprocessing_config}")
    else:
        preprocessing_config = None
        print("Warning: No preprocessing config found")
    
    # Get configured traffic signal IDs (used as reference only)
    ts_ids = get_network_ts_ids(network_name)
    print(f"[OK] Traffic signals (configured): {len(ts_ids)} agents")
    
    # Build additional SUMO command (SAME as training/eval_mgmq_ppo.py)
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
        print(f"[OK] Detector file: {detector_file}")
    
    # Print reward config
    print(f"[OK] Reward Function: {yaml_reward_cfg['reward_fn']}")
    print(f"[OK] Reward Weights: {yaml_reward_cfg['reward_weights']}")
    
    # Build environment config (SAME as training; fixed_ts keeps cycle stepping deterministic)
    # IMPORTANT: Disable reward normalization for baseline evaluation
    # We want RAW reward values for fair comparison with AI eval's raw_reward
    env_config = {
        "net_file": net_file,
        "route_file": route_file,
        "use_gui": use_gui,
        "virtual_display": None,
        "render_mode": "human" if render else None,
        "num_seconds": yaml_env_cfg["num_seconds"],
        "max_green": yaml_env_cfg["max_green"],
        "min_green": yaml_env_cfg["min_green"],
        "cycle_time": yaml_env_cfg["cycle_time"],
        "yellow_time": yaml_env_cfg["yellow_time"],
        # Match training: time_to_teleport from config
        "time_to_teleport": yaml_env_cfg.get("time_to_teleport", 500),
        "single_agent": False,
        "window_size": yaml_mgmq_cfg.get("history_length", yaml_mgmq_cfg.get("window_size", 1)),
        "preprocessing_config": preprocessing_config,
        "additional_sumo_cmd": additional_sumo_cmd,
        "reward_fn": yaml_reward_cfg["reward_fn"],
        "reward_weights": yaml_reward_cfg["reward_weights"],
        "use_phase_standardizer": yaml_env_cfg.get("use_phase_standardizer", True),
        "green_time_step": int(yaml_action_cfg.get("green_time_step", 5)),
        "use_neighbor_obs": False,  # Not needed for baseline
        "max_neighbors": yaml_mgmq_cfg.get("max_neighbors", 4),
        # BASELINE: Always disable normalization so rewards are raw
        # This gives the true "reference frame" for comparison
        # AI eval will also report raw_reward from info dict for fair comparison
        "normalize_reward": False,
        "clip_rewards": None,
        # We use fixed_ts stepping and optionally update phase durations via baseline controller.
        "fixed_ts": True,
    }
    
    expected_steps = yaml_env_cfg["num_seconds"] // yaml_env_cfg["cycle_time"]
    print(f"\n[OK] Cycle time: {yaml_env_cfg['cycle_time']}s")
    print(f"[OK] Simulation time: {yaml_env_cfg['num_seconds']}s")
    print(f"[OK] Expected steps per episode: ~{expected_steps}")
    print("")
    
    # Create environment (SAME as eval_mgmq_ppo.py)
    env = SumoMultiAgentEnv(**env_config)

    # Use ACTUAL active agents from env (some configured IDs may be skipped, e.g. no E2)
    active_ts_ids = list(env.ts_ids)
    if len(active_ts_ids) != len(ts_ids):
        print(
            f"Warning: Active agents in env: {len(active_ts_ids)} (configured: {len(ts_ids)}). "
            "Using active agents for eval metrics."
        )

    controller_name = (controller or "max_pressure_native").strip().lower()
    if controller_name not in {"fixed", "max_pressure_native", "max_pressure_legacy", "max_pressure"}:
        raise ValueError("controller must be one of: fixed, max_pressure_native, max_pressure_legacy")

    # Backward-compat alias
    if controller_name == "max_pressure":
        controller_name = "max_pressure_native"

    mp_controller = None
    mp_net_info_path = None
    if controller_name in {"max_pressure_native", "max_pressure_legacy"}:
        mp_net_info_path, mp_net_info_data = _discover_mp_net_info(
            project_root=project_root,
            network_name=network_name,
            active_ts_ids=active_ts_ids,
            explicit_path=mp_net_info,
        )
        if mp_net_info_data is None:
            raise FileNotFoundError(
                "Could not locate net-info.json for MaxPressure baseline. "
                "Provide --mp-net-info explicitly."
            )
        if controller_name == "max_pressure_native":
            mp_controller = MaxPressureNativeBaselineController(
                simulator=env.simulator,
                net_info=mp_net_info_data,
                active_ts_ids=active_ts_ids,
                project_root=project_root,
            )
        else:
            mp_controller = MaxPressureLegacyBaselineController(
                simulator=env.simulator,
                net_info=mp_net_info_data,
                active_ts_ids=active_ts_ids,
            )
        print(f"[OK] MP net-info: {mp_net_info_path}")
        if hasattr(mp_controller, "tls_cfg"):
            controlled_n = len(mp_controller.tls_cfg)
        elif hasattr(mp_controller, "controllers"):
            controlled_n = len(mp_controller.controllers)
        else:
            controlled_n = 0
        print(f"[OK] MP-controlled intersections: {controlled_n} / {len(active_ts_ids)}")
    
    # Evaluation metrics (SAME structure as eval_mgmq_ppo.py)
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
        if hasattr(mp_controller, "reset_runtime"):
            mp_controller.reset_runtime()
        # Keep baseline evaluation output clean by default.
        # If needed, enable TS debug logging manually when diagnosing rewards/actions.

        done = {"__all__": False}
        total_reward = 0
        total_raw_reward = 0
        agent_rewards = {ts_id: 0 for ts_id in active_ts_ids}
        agent_raw_rewards = {ts_id: 0 for ts_id in active_ts_ids}
        step_count = 0
        
        while not done.get("__all__", False):
            # For MaxPressure mode, update splits for signals that are due to act.
            if mp_controller is not None:
                mp_controller.apply_for_ready_signals(env.simulator.traffic_signals)

            # Baseline runner keeps actions empty; fixed_ts stepping handles cycle timing.
            actions = {}
            
            # Step environment (SAME as eval_mgmq_ppo.py)
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
            
            # Check if episode is done (SAME as eval_mgmq_ppo.py)
            done = truncateds
        
        episode_rewards.append(total_reward)
        episode_raw_rewards.append(total_raw_reward)
        episode_lengths.append(step_count)
        
        # Store per-agent rewards (SAME as eval_mgmq_ppo.py)
        for ts_id in active_ts_ids:
            if ts_id in agent_rewards:
                per_agent_rewards[ts_id].append(agent_rewards[ts_id])
            if ts_id in agent_raw_rewards:
                per_agent_raw_rewards[ts_id].append(agent_raw_rewards[ts_id])
        
        # Get system metrics if available (SAME as eval_mgmq_ppo.py)
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
    
    # Calculate statistics (SAME as eval_mgmq_ppo.py)
    results = {
        "network": network_name,
        "controller": controller_name,
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
    if mp_net_info_path is not None:
        results["mp_net_info"] = str(mp_net_info_path)
    
    # Per-agent statistics (SAME as eval_mgmq_ppo.py)
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
    
    # Print results (SAME format as eval_mgmq_ppo.py)
    print("\n" + "="*80)
    print("BASELINE RESULTS")
    print("="*80)
    print(f"\n  RAW Reward (actual traffic performance):")
    print(f"     Mean: {results['mean_raw_reward']:.2f} +/- {results['std_raw_reward']:.2f}")
    print(f"     Min/Max: {results['min_raw_reward']:.2f} / {results['max_raw_reward']:.2f}")
    print(f"\n  Normalized Reward (what training sees):")
    print(f"     Mean: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"     Min/Max: {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f"\n  Episode Length: {results['mean_length']:.1f}")
    
    if "mean_waiting_time" in results:
        print(f"  Mean Waiting Time: {results['mean_waiting_time']:.2f} +/- {results.get('std_waiting_time', 0):.2f}")
    if "mean_avg_speed" in results:
        print(f"  Mean Average Speed: {results['mean_avg_speed']:.2f} +/- {results.get('std_avg_speed', 0):.2f}")
    if "mean_total_halts" in results:
        print(f"  Mean Total Halts: {results['mean_total_halts']:.2f} +/- {results.get('std_total_halts', 0):.2f}")
    if "mean_throughput" in results:
        print(f"  Mean Throughput: {results['mean_throughput']:.0f} +/- {results.get('std_throughput', 0):.0f}")
    if "mean_pressure" in results:
        print(f"  Mean Pressure: {results['mean_pressure']:.4f} +/- {results.get('std_pressure', 0):.4f}")
    
    print("\n  Per-Agent Rewards (raw / normalized):")
    for ts_id, stats in per_agent_stats.items():
        raw_str = f"{stats.get('mean_raw_reward', 0):.2f}" if 'mean_raw_reward' in stats else "N/A"
        norm_str = f"{stats.get('mean_reward', 0):.2f}" if 'mean_reward' in stats else "N/A"
        print(f"    {ts_id}: raw={raw_str}, norm={norm_str}")
    
    print("\n" + "-"*40)
    print("COMPARISON GUIDELINE:")
    print("-"*40)
    print(f"Baseline RAW Mean Reward: {results['mean_raw_reward']:.2f}")
    print(f"If AI's raw_reward > {results['mean_raw_reward']:.2f} -> AI is BETTER")
    print(f"If AI's raw_reward < {results['mean_raw_reward']:.2f} -> AI is WORSE")
    print("="*80 + "\n")
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline traffic signal control (fixed-time or max_pressure)"
    )
    parser.add_argument("--network", type=str, default=None,
                        choices=["grid4x4", "4x4loop", "network_test", "zurich", "PhuQuoc", "test"],
                        help="Network name (if omitted, use network from config)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Use SUMO GUI for visualization")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--output", type=str, default="baseline_results.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--seeds", type=int, nargs='+', default=None,
                        help="Evaluation seeds, one episode per seed. If omitted, auto-generate from --episodes.")
    parser.add_argument("--controller", type=str, default="max_pressure_native",
                        choices=["max_pressure_native", "max_pressure_legacy", "max_pressure", "fixed"],
                        help="Baseline controller type")
    parser.add_argument("--mp-net-info", type=str, default=None,
                        help="Optional explicit path to MP net-info.json")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model_config.yml (default: src/config/model_config.yml)")
    
    args = parser.parse_args()
    
    evaluate_baseline(
        network_name=args.network,
        num_episodes=args.episodes,
        use_gui=args.gui,
        render=args.render,
        output_file=args.output,
        seeds=args.seeds,
        config_path=args.config,
        controller=args.controller,
        mp_net_info=args.mp_net_info,
    )
    
