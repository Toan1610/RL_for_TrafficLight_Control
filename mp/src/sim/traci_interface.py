import os
from typing import List, Dict, Optional
from dataclasses import asdict


class TraciIF:
    def __init__(self, sumo_cfg: dict):

        self.cfg = sumo_cfg
        self._net = None
        self._running = False
        self._begin = float(sumo_cfg.get("begin", 0))
        self._end = float(sumo_cfg.get("end", 3600))
        self._step = float(sumo_cfg.get("step_length", 0.1))
        self._step_count = 0

        # Import traci or libsumo
        import importlib
        self.traci = None
        runner = self.cfg.get("runner")
        if runner == "libsumo":
            try:
                self.traci = importlib.import_module("libsumo")
            except ModuleNotFoundError:
                try:
                    self.traci = importlib.import_module("traci")
                except ModuleNotFoundError as e:
                    raise ImportError("Neither 'libsumo' nor 'traci' could be imported. Ensure SUMO is installed and PYTHONPATH is set.") from e
        else:
            try:
                self.traci = importlib.import_module("traci")
            except ModuleNotFoundError as e:
                raise ImportError("'traci' could not be imported. Ensure SUMO is installed and PYTHONPATH is set.") from e

    def _ensure_import(self):
        if self.traci is None:
            raise ImportError("Could not import traci/sumolib. Ensure SUMO is installed and PYTHONPATH is set.")

    def start(self):
        self._ensure_import()

        # Determine gui mode and use full path to SUMO executable
        import sys
        import os
        
        
        # Construct path to SUMO executable in the same environment
        if self.cfg["gui"]:
            sumo_gui = "sumo-gui"
        else:
            sumo_gui = "sumo"

        # Configure SUMO command
        sumoCmd = [
            sumo_gui,
            "--no-warnings",
            "--start",  # start simulation immediately
            "-c", self.cfg["sumocfg"],
            "--step-length", str(self._step),
            "--begin", str(int(self._begin)),
            "--end", str(int(self._end)),
            "--no-step-log", "true"
        ]

        lateral_resolution = self.cfg.get("lateral_resolution")
        if lateral_resolution is not None:
            sumoCmd += ["--lateral-resolution", str(lateral_resolution)]

        time_to_teleport = self.cfg.get("time_to_teleport", 500)
        sumoCmd += ["--time-to-teleport", str(int(time_to_teleport))]

        add = self.cfg.get("add_file")
        if add:
            sumoCmd += ["-a", add]
        self.traci.start(sumoCmd, port=self.cfg.get("port", 8813))
        self._running = True

    def start_evaluation(self, evaluations, output_dir: str):
        self._ensure_import()
        assert self.traci is not None, "TraCI module is not initialized"

        # Determine gui mode and use full path to SUMO executable
        import sys
        import os
        
        # Get the directory of the current Python executable
        python_dir = os.path.dirname(sys.executable)
        
        # Construct path to SUMO executable in the same environment
        if self.cfg["gui"]:
            sumo_gui = os.path.join(python_dir, "sumo-gui.exe")
        else:
            sumo_gui = os.path.join(python_dir, "sumo.exe")

        # Configure SUMO command
        sumoCmd = [
            sumo_gui,
            "--no-warnings",
            "--start",  # start simulation immediately
            "-c", self.cfg["sumocfg"],
            "--step-length", str(self._step),
            "--no-step-log", "true"
        ]

        for eval_item in evaluations:
            sumoCmd += [f"--{eval_item}", f"{output_dir}_{eval_item}.xml"]

        add = self.cfg.get("add_file")
        if add:
            sumoCmd += ["-a", add]
        self.traci.start(sumoCmd)
        self._running = True

    def close(self):
        if self._running:
            self.traci.close()
            self._running = False

    def step(self):
        self.traci.simulationStep()
        # if self.traci.simulation.getTime() >= 800:
        #     self.traci.close()

    def step_to(self, t_abs: float):
        # SUMO cho phép simulationStep(time) nhảy tới absolute time
        self.traci.simulationStep(t_abs)

    def begin_time(self)->float:
        return self._begin

    def end_time(self)->float:
        return self._end

    def list_tls_ids(self)->List[str]:
        return list(self.traci.trafficlight.getIDList())
    
    def get_current_phase(self, tls_id: str) -> int:
        """Lấy pha hiện tại của đèn giao thông."""
        return self.traci.trafficlight.getPhase(tls_id)

    def get_time(self) -> float:
        """Lấy thời gian mô phỏng hiện tại."""
        return self.traci.simulation.getTime()
    
    def get_list_edge(self):
        return self.traci.edge.getIDList()

    def get_lanearea_occupancy(self, detector_id: str) -> float:
        return self.traci.lanearea.getLastIntervalOccupancy(detector_id)

    def get_edge_occupancy(self, edge_id: str) -> float:
        """Lấy thông tin lưu lượng của một đoạn đường."""
        return self.traci.edge.getLastStepOccupancy(edge_id)

    def get_total_vehicle(self):
        """Lấy tổng số phương tiện trên một đoạn đường."""
        return self.traci.vehicle.getIDCount()

    def set_phase(self, tls_id: str, phase_index: int):
        """Set the current phase of the traffic light."""
        self.traci.trafficlight.setPhase(tls_id, phase_index)

    def set_duration(self, tls_id: str, duration: float):
        """Set the duration of the current phase."""
        # Note: This is a simplified version, real implementation should handle yellow/all-red phases
        self.traci.trafficlight.setPhaseDuration(tls_id, duration)

    def _phase_elapsed(self, tls_id: str) -> float:
        return self.traci.trafficlight.getPhaseDuration(tls_id) - self.traci.trafficlight.getNextSwitch(tls_id) + self.traci.simulation.getTime()

    def observe_tls(self, tls_id: str, ctrl_name: str):

        # handle for max_pressure
        if ctrl_name == "max_pressure":
            pi = self.traci.trafficlight.getPhase(tls_id) # Get current phase index. Ex: 0
            phases = [p.state for p in self.traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases] # Get all phase states. Ex: 
            elapsed = max(0.0, float(self.traci.trafficlight.getPhaseDuration(tls_id) - self.traci.trafficlight.getNextSwitch(tls_id) + self.traci.simulation.getTime()))
            lanes_in = self.traci.trafficlight.getControlledLanes(tls_id)
            # crude outgoing lanes via links
            links = self.traci.trafficlight.getControlledLinks(tls_id)
            lanes_out = list({toLane for group in links for (_from, toLane, _via) in group if toLane})

            # simple queues/densities
            q_in = {ln: float(self.traci.lane.getLastStepVehicleNumber(ln)) for ln in lanes_in}
            q_out = {ln: float(self.traci.lane.getLastStepVehicleNumber(ln)) for ln in lanes_out}
            d_in = {ln: float(self.traci.lane.getLastStepOccupancy(ln)) for ln in lanes_in}
            d_out = {ln: float(self.traci.lane.getLastStepOccupancy(ln)) for ln in lanes_out}
            return {
                "phase_index": pi,
                "phase_elapsed": elapsed,
                "phases": phases,
                "min_green": float(self.cfg.get("min_green", 5.0)),
                "yellow": float(self.cfg.get("yellow", 3.0)),
                "all_red": float(self.cfg.get("all_red", 1.0)),
                "incoming_lanes": lanes_in,
                "outgoing_lanes": lanes_out,
                "queues_in": q_in,
                "queues_out": q_out,
                "densities_in": d_in,
                "densities_out": d_out,
            }
        elif ctrl_name == "webster":
            # Implement observation extraction for Webster controller
            return {}
        else:
            return {}

    def safe_switch(self, tls_id: str, next_phase: int, min_green: float, yellow: float, all_red: float):
        # Very simplified: directly set phase (real implementation should insert yellow/all-red safety)
        # self.traci.trafficlight.setPhase(tls_id, int(next_phase))
        pass

    def set_tls_splits(self, tls_id: str, splits):
        # Simplified: set phase duration for the current cycle; full implementation would rebuild a program
        # Here we no-op to keep it safe for a scaffold; SUMO program editing is non-trivial.
        self.traci.trafficlight.setCompleteRedYellowGreenDefinition(tls_id, splits)

    def get_tls_splits(self, tls_id):
        return self.traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]

    def snapshot_network_density(self) -> float:
        lanes = self.traci.lane.getIDList()
        if not lanes:
            return 0.0
        occ = [float(self.traci.lane.getLastStepOccupancy(l)) for l in lanes]
        return float(sum(occ) / len(occ))
    
    def get_step_count(self) -> int:
        return self._step_count
    
    def step_with_delay(self, delay: float = 0.0):
        import time
        self.traci.simulationStep()
        self._step_count += 1
        if delay > 0:
            time.sleep(delay)

    def get_tls_cycle_time(self, tls_id: str) -> float:
        """Get the cycle time of a traffic light system."""
        phases = self.traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0].phases
        return sum(phase.duration for phase in phases if phase.duration > 0)
    
    def step_length(self) -> float:
        return self._step
    
    def is_completed(self) -> bool:
        current_time = self.traci.simulation.getTime()
        # Kết thúc khi đạt thời gian kết thúc được cấu hình
        if current_time >= self._end:
            return True
        # Hoặc khi không còn xe nào trên mạng lưới
        min_expected = self.traci.simulation.getMinExpectedNumber()
        return min_expected <= 0
