from dataclasses import dataclass
from typing import Dict, Any
from .edge_info import EdgeInfo
from .phase_info import PhaseInfo
from .movement_info import MovementInfo


@dataclass
class TLSInfo:
    """Data Transfer Object for Traffic Light System information"""
    cycle: float
    edges: Dict[str, EdgeInfo]
    phases: Dict[str, PhaseInfo]
    movements: Dict[str, Dict[str, float]]  # from_edge -> {to_edge: ratio}
    
    def __post_init__(self):
        if self.cycle <= 0:
            raise ValueError("Cycle time must be positive")
        if not self.edges:
            raise ValueError("Edges cannot be empty")
        if not self.phases:
            raise ValueError("Phases cannot be empty")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TLSInfo':
        """Create TLSInfo from dictionary data"""
        # Process edges
        edges = {}
        for edge_id, edge_data in data.get("edges", {}).items():
            edges[edge_id] = EdgeInfo.from_dict(edge_id, edge_data)
        
        # Process phases
        phases = {}
        for phase_id, phase_data in data.get("phases", {}).items():
            phases[phase_id] = PhaseInfo.from_dict(phase_id, phase_data)
        
        return cls(
            cycle=data.get("cycle", 90.0),
            edges=edges,
            phases=phases,
            movements=data.get("movements", {})
        )
    
    def get_edge(self, edge_id: str) -> EdgeInfo:
        """Get edge information by ID"""
        if edge_id not in self.edges:
            raise KeyError(f"Edge {edge_id} not found")
        return self.edges[edge_id]
    
    def get_phase(self, phase_id: str) -> PhaseInfo:
        """Get phase information by ID"""
        if phase_id not in self.phases:
            raise KeyError(f"Phase {phase_id} not found")
        return self.phases[phase_id]
    
    def get_movements_for_edge(self, from_edge: str) -> Dict[str, float]:
        """Get all movements from a specific edge"""
        return self.movements.get(from_edge, {})
