from typing import Dict, Any
import logging

from src.sim.traci_interface import TraciIF
from src.service.domain.collected_data import CollectedData
from src.controller import build as build_controller

logger = logging.getLogger(__name__)

class SumoRunner():
    def __init__(self, sumo_cfg: dict, controller_plan: dict, net_info: dict):
        self.sumo_cfg = sumo_cfg
        self.iface = TraciIF(sumo_cfg)
        self.net_info = net_info
        self.controller_plan = controller_plan

        self.cache_edges_occupancy = {}
        self.controllers = {}
        
    def _collect_data(self, tls_ids: list[str]):
        # get all edge_ids and edge_info controlled by tls_ids 
        # nếu một edge_id được nhiều tls kiểm soát thì nó sẽ được tính 1 lần
        controlled_edges = {}
        for tls_id in tls_ids:
            if tls_id in self.net_info["tls"]:
                for edge_id, edge_info in self.net_info["tls"][tls_id]["edges"].items():
                    if edge_id not in controlled_edges.keys():
                        controlled_edges[edge_id] = edge_info
        # collect data
        for edge_id, edge_info in controlled_edges.items():
            # cache edge occupancy for MaxPressure
            occupancy = []
            mean_occupancy = 0.0
            if edge_info.get("detector") == [] or edge_info.get("detector") is None:
                occupancy.append(0.0)
            for detector in edge_info["detector"]:
                occupancy.append(self.iface.get_lanearea_occupancy(detector))
            mean_occupancy = sum(occupancy) / len(occupancy) if occupancy else 0.0
            if edge_id not in self.cache_edges_occupancy:
                self.cache_edges_occupancy[edge_id] = [mean_occupancy]
            else:
                self.cache_edges_occupancy[edge_id].append(mean_occupancy)

        # giữ lại 20 phần tử mới nhất trong mỗi mảng cache_edges_occupancy[edge_id]
            if len(self.cache_edges_occupancy[edge_id]) > 20:
                self.cache_edges_occupancy[edge_id] = self.cache_edges_occupancy[edge_id][-20:]
    
    def _controller_for(self, tls_id: str, adaptive_mask: dict):
        # For Adaptive
        # if adaptive_mask.get(tls_id):
        # spec = self.controller_plan[adaptive_mask.get(tls_id)]
        if tls_id in adaptive_mask:
            spec = self.controller_plan[adaptive_mask[tls_id]]
            params = dict(spec.get("params", {}))

            # Bơm tls_info từ net vào cho Adaptive
            tls_info = self.net_info["tls"][tls_id]
            params.setdefault("tls_info", tls_info)
            return build_controller(spec["name"], tls_id, self.iface, **params)
        
        # For default
        spec = self.controller_plan["fixed_time"]
        params = dict(spec.get("params", {}))
        return build_controller(spec["name"], tls_id, self.iface, **params)

    def run(self, adaptive_mask: Dict[str, str]):
        """Run simulation and drive controllers based on schedule.

        - Time is handled in absolute seconds (SUMO time).
        - Each controller returns its next action time via action(t_now).
        - Sampling is performed at a global sample interval from SUMO config.
        """
        logger.info(
            "Starting SUMO run: begin=%s end=%s step=%.3f sample_interval=%s",
            self.iface.begin_time(), self.iface.end_time(), self.iface.step_length(), self.sumo_cfg.get("sample_interval")
        )
        self.iface.start()
        
        # Timeout protection
        import time
        start_wall_time = time.time()
        max_wall_time = 3600  # 1 hour timeout
        last_progress_time = t_now = self.iface.begin_time()
        stuck_threshold = 120  # If no progress for 120s, consider stuck

        try:
            tls_ids = self.iface.list_tls_ids()

            # Validate tls ids referenced in adaptive mask
            for tls_id in adaptive_mask.keys():
                if tls_id not in tls_ids:
                    logger.warning("TLS ID %s not found in SUMO.", tls_id)

            # Initialize controllers
            for tls_id in tls_ids:
                ctrl = self._controller_for(tls_id, adaptive_mask)
                self.controllers[tls_id] = ctrl
                logger.info("Init controller for TLS %s -> %s", tls_id, type(ctrl).__name__)
                ctrl.start()

            # Scheduling variables
            sample_interval = int(self.sumo_cfg.get("sample_interval", 10.0))
            t_now = self.iface.begin_time()
            next_sample_time = t_now + sample_interval 

            # Initialize controller triggers by calling action once at t_now
            # so controllers can compute their first next time
            triggers: Dict[str, float] = {}
            for tls_id, ctrl in self.controllers.items():
                try:
                    cycle = self.iface.get_tls_cycle_time(tls_id)
                    triggers[tls_id] = (t_now + cycle)
                    logger.debug("TLS %s initial next action at t=%d", tls_id, triggers[tls_id])
                except Exception as e:
                    logger.exception("Controller %s initial action failed: %s", tls_id, e)
                    # Fallback: schedule by cycle time if available
                    try:
                        cycle = float(self.iface.get_tls_cycle_time(tls_id))
                        triggers[tls_id] = t_now + (cycle)
                        logger.debug("TLS %s fallback next action at t=%d (cycle)", tls_id, triggers[tls_id])
                    except Exception:
                        # If we can't get a cycle, skip scheduling
                        continue

            # Main loop: advance to the next event time until simulation completes
            # Guard against empty triggers; still keep sampling until SUMO completes
            def _next_event_time() -> float:
                candidates = [next_sample_time]
                if triggers:
                    candidates.append(min(triggers.values()))
                return min(candidates)

            # Step until SUMO has no expected vehicles
            while not self.iface.is_completed():
                # Check for timeout
                wall_time_elapsed = time.time() - start_wall_time
                if wall_time_elapsed > max_wall_time:
                    logger.error("Simulation timeout after %.0f seconds (wall time)", wall_time_elapsed)
                    break
                
                # Check if simulation is stuck
                if t_now > last_progress_time and (t_now - last_progress_time) > stuck_threshold:
                    logger.warning("No progress for %ds, checking status...", stuck_threshold)
                    min_exp = self.iface.traci.simulation.getMinExpectedNumber()
                    logger.warning("Current time=%d, MinExpected=%d", t_now, min_exp)
                    last_progress_time = t_now
                
                t_next = _next_event_time()
                
                # Safety check: prevent infinite loop on same time
                if t_next <= t_now and t_next < self.iface.end_time():
                    logger.error("Time not advancing: t_now=%d, t_next=%d. Forcing step.", t_now, t_next)
                    t_next = t_now + 1

                # Advance simulation to next scheduled event
                logger.debug("Advance sim: t_now=%d -> t_next=%d", t_now, t_next)
                try:
                    self.iface.step_to(t_next)
                except Exception as e:
                    # Handle SUMO closing in between steps gracefully
                    msg = str(e)
                    if "Connection closed by SUMO" in msg:
                        logger.info("SUMO connection closed during step; ending run at t=%d", t_now)
                        break
                    # Unknown error: re-raise
                    raise
                t_now = t_next

                # Sampling event
                if t_now >= next_sample_time:
                    min_expected = self.iface.traci.simulation.getMinExpectedNumber()
                    departed = self.iface.traci.simulation.getDepartedNumber()
                    arrived = self.iface.traci.simulation.getArrivedNumber()
                    wall_elapsed = time.time() - start_wall_time
                    logger.info("Sample at t=%.2f (interval=%ss) | MinExpected=%d | Departed=%d | Arrived=%d | Wall=%.0fs", 
                               t_now, sample_interval, min_expected, departed, arrived, wall_elapsed)
                    
                    # Update progress tracker
                    last_progress_time = t_now
                
                    try:
                        self._collect_data(tls_ids)
                    except Exception:
                        # Keep sampling robust; data collection is auxiliary
                        logger.exception("Data collection failed for TLS %s", tls_id)
                        logger.debug("  TLS %s sample failed", tls_id)
                    next_sample_time += int(sample_interval) 

                # Controller events (may be multiple at the same time)
                for tls_id, due_time in list(triggers.items()):
                    if t_now >= due_time:
                        try:
                            logger.info("Controller action: TLS %s at t=%d", tls_id, t_now)
                            if(type(self.controllers[tls_id]).__name__ != "FixedTime"):
                                cycle = self.iface.get_tls_cycle_time(tls_id)
                                cycle = self.controllers[tls_id].action(t_now, self.cache_edges_occupancy, int(cycle/sample_interval))
                                next_t = int(t_now + cycle)
                                logger.debug("  TLS %s rescheduled next action at t=%d", tls_id, next_t)
                                triggers[tls_id] = next_t
                            else:
                                triggers[tls_id] = 10000000
                        except Exception as e:
                            logger.exception("Controller %s action failed: %s", tls_id, e)
                            # Try to reschedule by its cycle time to avoid stall
                            try:
                                cycle = self.iface.get_tls_cycle_time(tls_id)
                                triggers[tls_id] = int(t_now + cycle)
                                logger.debug(" TLS %s rescheduled by cycle: next=%d", tls_id, triggers[tls_id])
                            except Exception:
                                # Remove if cannot reschedule
                                triggers.pop(tls_id, None)

            logger.info("Simulation completed at t=%d", t_now)
        finally:
            self.iface.close()
            logger.info("SUMO closed")

            