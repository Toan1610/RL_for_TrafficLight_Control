from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from .movement_info import MovementInfo


@dataclass
class PhaseInfo:
    """Data Transfer Object for phase information"""
    phase_id: str
    movements: List[MovementInfo]
    min_green: float
    max_green: float
    
    def __post_init__(self):
        if self.min_green < 0:
            raise ValueError("Minimum green time cannot be negative")
        if self.max_green <= self.min_green:
            raise ValueError("Maximum green time must be greater than minimum green time")
        if not isinstance(self.movements, list):
            raise TypeError("Movements must be a list")
    
    @classmethod
    def from_dict(cls, phase_id: str, data: Dict[str, Any]) -> 'PhaseInfo':
        """Create PhaseInfo from dictionary data"""
        movements: List[MovementInfo] = []
        for movement in data.get("movements", []):
            # Accept [from_edge, to_edge] or {"from":..., "to":..., "ratio":...}
            if isinstance(movement, (list, tuple)) and len(movement) >= 2:
                movements.append(MovementInfo(from_edge=movement[0], to_edge=movement[1], ratio=0.0))
            elif isinstance(movement, dict):
                from_edge = movement.get("from") or movement.get("from_edge")
                to_edge = movement.get("to") or movement.get("to_edge")
                ratio = movement.get("ratio", 0.0)
                if from_edge and to_edge:
                    movements.append(MovementInfo(from_edge=from_edge, to_edge=to_edge, ratio=ratio))
        
        return cls(
            phase_id=phase_id,
            movements=movements,
            min_green=data.get("min-green", 10.0),
            max_green=data.get("max-green", 60.0)
        )
