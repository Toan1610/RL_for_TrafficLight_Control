from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class MovementInfo:
    """Data Transfer Object for movement information"""
    from_edge: str
    to_edge: str
    ratio: float
    
    def __post_init__(self):
        if not (0 <= self.ratio <= 1):
            raise ValueError("Movement ratio must be between 0 and 1")
        if not self.from_edge or not self.to_edge:
            raise ValueError("From edge and to edge cannot be empty")
    
    @classmethod
    def from_dict(cls, from_edge: str, movements_data: Dict[str, float]) -> list['MovementInfo']:
        """Create list of MovementInfo from dictionary data"""
        movements = []
        for to_edge, ratio in movements_data.items():
            movements.append(cls(
                from_edge=from_edge,
                to_edge=to_edge,
                ratio=ratio
            ))
        return movements
