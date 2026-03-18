# src/controllers/max_pressure.py
from src.controller.base_controller import BaseController
from src.controller import register
from .dto import MaxPressureConfig
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

@register("max_pressure")
class MaxPressure(BaseController):
    def __init__(self, tls_id: str, iface, **params):
        """
        Supported inputs:
        - config: MaxPressureConfig
        - tls_info: TLSInfo | dict, sample_interval: float, cycling: str
        """
        # If caller provided a DTO, honor it; else build from params
        cfg_obj = params.pop("config", None)
        if isinstance(cfg_obj, MaxPressureConfig):
            config = cfg_obj
            # also backfill basic fields to BaseController.cfg for consistency
            params.setdefault("tls_info", config.tls_info)
            params.setdefault("sample_interval", config.sample_interval)
            params.setdefault("cycling", config.cycling)
        elif isinstance(cfg_obj, dict):
            # Build DTO from inner dict
            config = MaxPressureConfig.from_params(**cfg_obj)
            params.setdefault("tls_info", config.tls_info)
            params.setdefault("sample_interval", config.sample_interval)
            params.setdefault("cycling", config.cycling)
        else:
            # Keep params intact for BaseController while building DTO internally
            config = MaxPressureConfig.from_params(**params)

        super().__init__(tls_id, iface, **params)
        self.config = config
        self.lost_time = self._calculate_lost_time()

        self.cache_edges_occupancy = {}
    
    # Note: Only DTO-based construction is supported to ensure encapsulation.

    def start(self):
        # Initialize any necessary data structures or states

        # init cache edge occupancy
        
        for edge_id in self.config.edges.keys():
            self.cache_edges_occupancy[edge_id] = []
        logger.info(
            "MaxPressure start: tls=%s cycle=%s edges=%d phases=%d",
            self.tls_id, self.config.cycle_time, len(self.config.edges), len(self.config.phases)
        )

    def _sample_action(self):
        for edge_id, edge_info in self.config.tls_info.edges.items():
            edge_occupancy = []
            for detector in edge_info.detector:
                edge_occupancy.append(self.iface.get_lanearea_occupancy(detector))

            # if edge have detector
            if edge_occupancy:
                mean_occ = np.mean(edge_occupancy)
                self.cache_edges_occupancy[edge_id].append(mean_occ)
                logger.warning(
                    "[%s] sample edge=%s detectors=%s mean_occ=%.4f",
                    self.tls_id, edge_id, len(edge_occupancy), float(mean_occ)
                )
            # if edge has no detector
            else:
                self.cache_edges_occupancy[edge_id].append(np.float64(0))
                logger.warning("[%s] sample edge=%s no-detector mean_occ=0", self.tls_id, edge_id)
                # Print Warning
                # print(f"Warning: No detector found for edge {edge_id}")

    def _set_split(self, final_greentimes, splits):
        splits = self.iface.get_tls_splits(self.tls_id)
        for i, duration in final_greentimes.items():
            splits.phases[int(i)].duration = duration
        logger.warning("[%s] apply splits: %s", self.tls_id, final_greentimes)
        self.iface.set_tls_splits(self.tls_id, splits)

    def _decide_action(self):
        # Implement your decision-making logic here
        phases_pressure = self._calculate_phases_pressure()
        greentimes = self._initialize_greentime(phases_pressure)
        constrained_greentimes = self._constrain_greentimes(greentimes)

        final_greentimes = constrained_greentimes
        # High-level I/O logs
        try:
            edges_in = {k: float(v) if v == v else 0.0 for k, v in getattr(self, "_last_edges_occupancy", {}).items()}
            logger.warning("[%s] INPUT edges_occupancy=%s", self.tls_id, edges_in)
        except Exception:
            pass
        logger.warning("[%s] INPUT phases_pressure=%s", self.tls_id, {k: float(v) for k, v in phases_pressure.items()})
        logger.warning("[%s] OUTPUT greentimes=%s", self.tls_id, {k: float(v) for k, v in final_greentimes.items()})

        # Update the plan with the optimized green times
        self._set_split(final_greentimes, self.iface.get_tls_splits(self.tls_id))
        # print(f"Time: {self.iface.get_time()} - ID: {self.tls_id} -> MAX PRESSURE: SET CYCLE {final_greentimes}")


    def _calculate_lost_time(self):
        lost_time = 0
        splits = self.iface.get_tls_splits(self.tls_id)
        # Calculate lost time based on non-green phases
        for phase in splits.phases:
            state = phase.state.lower()
            # Count lost time for phases that are not green or are red without green
            # Check if duration < 15 is yellow phase and all red phase
            if phase.duration < 15:
                lost_time += phase.duration
        logger.warning("[%s] lost_time=%.2f", self.tls_id, float(lost_time))
        return lost_time

    def _calculate_phases_pressure(self):
        # calculate edges occupancy
        edges_occupancy = {}
        for edge_id in self.cache_edges_occupancy.keys():
            edges_occupancy[edge_id] = np.mean(self.cache_edges_occupancy[edge_id])/100
        # Ensure we also have occupancy for all outgoing edges referenced in movements

        logger.warning("[%s] edges_occupancy=%s", self.tls_id, {k: float(v) for k, v in edges_occupancy.items()})

        # calculate movements pressure
        movements_pressure = {}
        for from_edge, movements_data in self.config.movements.items():
            # Get sum out_edge by ratio
            logger.warning("[%s] movements_data=%s", self.tls_id, movements_data)
            out_pressure = 0.0
            for out_edge, ratio in movements_data.items():
                # use safe lookup for logging and accumulation
                occ_out = float(edges_occupancy.get(out_edge, 0.0))
                logger.warning("[%s] out_edge=%s occupancy=%.4f ratio=%.4f", self.tls_id, out_edge, occ_out, float(ratio))
                out_pressure += occ_out * float(ratio)
            edge_info = self.config.tls_info.get_edge(from_edge)
            logger.warning("[%s] edge_info=%s", self.tls_id, edge_info)
            logger.warning("[%s] from_edge=%s out_pressure=%.4f", self.tls_id, from_edge, float(out_pressure))
            in_pressure = edges_occupancy[from_edge]
            movements_pressure[from_edge] = (in_pressure - out_pressure) * edge_info.sat_flow
        logger.warning("[%s] movements_pressure=%s", self.tls_id, {k: float(v) for k, v in movements_pressure.items()})
        
        # calculate phases pressure
        phases_pressure = {}
        for phase_id, phase_info in self.config.tls_info.phases.items():
            phase_pressure = 0
            for movement in phase_info.movements:
                from_edge = movement.from_edge
                to_edge = movement.to_edge
                if from_edge in movements_pressure and from_edge in self.config.movements:
                    phase_pressure += movements_pressure.get(from_edge, 0) * self.config.movements[from_edge].get(to_edge, 0)
            phases_pressure[phase_id] = max(phase_pressure, 0)

        # cache last inputs for high-level logs
        self._last_edges_occupancy = edges_occupancy
        self._last_movements_pressure = movements_pressure
        self._last_phases_pressure = phases_pressure

        logger.warning("[%s] phases_pressure=%s", self.tls_id, {k: float(v) for k, v in phases_pressure.items()})
        return phases_pressure

    def _initialize_greentime(self, phases_pressure):
        """Initialize green time for each phase based on pressure - simplified."""
        total_greentime = self.config.cycle_time - self.lost_time
        total_phase_pressures = sum(phases_pressure.values())
        greentimes = {}
        # Handle zero or negative pressure
        if total_phase_pressures <= 0:
            # Equal distribution when no pressure
            equal_greentime = total_greentime / len(phases_pressure)
            for phase in phases_pressure:
                greentimes[phase] = equal_greentime
        else:
            # Proportional distribution based on pressure
            
            if self.config.cycling == "linear":
                for phase in phases_pressure:
                    greentimes[phase] = (phases_pressure[phase] / total_phase_pressures) * total_greentime
            elif self.config.cycling == "exponential":
                mean_pressure = total_phase_pressures / len(phases_pressure)
                exp_pressures = [math.exp(phases_pressure[phase] / mean_pressure) for phase in phases_pressure]
                total_exp_pressures = sum(exp_pressures)
                for i, phase in enumerate(phases_pressure):
                    greentimes[phase] = (exp_pressures[i] / total_exp_pressures) * total_greentime

            else:
                raise ValueError("Cycling method not recognized. Use 'linear' or 'exponential'.")
        logger.warning(
            "[%s] init_greentimes (pre-constraint): total=%.2f values=%s",
            self.tls_id, float(total_greentime), {k: float(v) for k, v in greentimes.items()}
        )
        return greentimes
        
    
    def _constrain_greentimes(self, greentimes):
        """Constrain green times using simple and efficient algorithm."""
        phases = self.config.phases
        greentimes_arr = list(greentimes.values())
        # Get constraints
        min_greentimes = [data["min-green"] for phase, data in phases.items()]
        max_greentimes = [data["max-green"] for phase, data in phases.items()]
        target_sum = int(round(self.config.cycle_time - self.lost_time))

        if target_sum <= 0:
            logger.warning("[%s] non-positive target green time=%s, forcing target_sum=1", self.tls_id, target_sum)
            target_sum = 1
        
        # Validate feasibility
        min_sum = sum(min_greentimes)
        max_sum = sum(max_greentimes)
        
        if min_sum > target_sum:
            logger.warning(
                "[%s] infeasible min constraints: min_sum=%s > target=%s; relaxing minima",
                self.tls_id,
                min_sum,
                target_sum,
            )
            if target_sum <= 0:
                min_greentimes = [0 for _ in min_greentimes]
            else:
                # Scale minimum greens to remain feasible while preserving ratios.
                scaled_min = [max(1, int((m / min_sum) * target_sum)) for m in min_greentimes]
                diff_min = target_sum - sum(scaled_min)
                idx = 0
                while diff_min > 0 and scaled_min:
                    scaled_min[idx % len(scaled_min)] += 1
                    diff_min -= 1
                    idx += 1
                while diff_min < 0 and scaled_min:
                    j = idx % len(scaled_min)
                    if scaled_min[j] > 1:
                        scaled_min[j] -= 1
                        diff_min += 1
                    idx += 1
                min_greentimes = scaled_min
            min_sum = sum(min_greentimes)
            logger.warning("[%s] relaxed min constraints=%s (sum=%s)", self.tls_id, min_greentimes, min_sum)
        if max_sum < target_sum:
            logger.warning(
                "[%s] infeasible max constraints: max_sum=%s < target=%s; clipping target to max_sum",
                self.tls_id,
                max_sum,
                target_sum,
            )
            target_sum = max_sum
        
        # Simple proportional scaling with constraint enforcement
        result = []
        sum_initial = sum(greentimes_arr)
        total_initial = sum_initial if sum_initial > 0 else len(greentimes_arr)
        logger.warning(
            "[%s] constraint: target=%s min=%s max=%s total_initial=%.2f",
            self.tls_id, target_sum, min_greentimes, max_greentimes, float(total_initial)
        )

        # Scale proportionally and apply constraints
        for i, greentime in enumerate(greentimes_arr):
            scaled = (greentime / total_initial) * target_sum if total_initial > 0 else target_sum / len(greentimes_arr)
            constrained = max(min_greentimes[i], min(max_greentimes[i], int(round(scaled))))
            result.append(constrained)
        
        # Adjust to meet exact target sum
        current_sum = sum(result)
        diff = target_sum - current_sum
        
        # Distribute difference
        attempts = 0
        while diff != 0 and attempts < max(1, target_sum):  # Prevent infinite loop
            if diff > 0:  # Need to add time
                for i in range(len(result)):
                    if result[i] < max_greentimes[i] and diff > 0:
                        result[i] += 1
                        diff -= 1
            else:  # Need to remove time
                for i in range(len(result)):
                    if result[i] > min_greentimes[i] and diff < 0:
                        result[i] -= 1
                        diff += 1
            attempts += 1
        
        # convert greentimes_arr to greentimes
        for k, v in zip(greentimes.keys(), result):
            greentimes[k] = v
        logger.warning("[%s] constrained_greentimes=%s", self.tls_id, greentimes)
        return greentimes

    def action(self, t, edges_occupancy, n_samples: int):
        # Perform action every sample interval
        # if int(t) % int(self.config.sample_interval) == 0:
        #     self._sample_action()
        # Take only n_samples elements from each edge occupancy array
        for edge_id in edges_occupancy:
            if len(edges_occupancy[edge_id]) > n_samples:
                edges_occupancy[edge_id] = edges_occupancy[edge_id][-n_samples:]
        self.cache_edges_occupancy = edges_occupancy
        # Perform action every cycle time
        if int(t) % int(self.config.cycle_time) == 0:
            logger.info("[%s] cycle decision at t=%.2f", self.tls_id, t)
            self._decide_action()

        # Next absolute time to act: next cycle boundary
        cycle = float(self.config.cycle_time)
        return cycle


