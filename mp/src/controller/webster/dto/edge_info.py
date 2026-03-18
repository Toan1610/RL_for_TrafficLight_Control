from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class EdgeInfo:
    """Data Transfer Object for edge information"""
    edge_id: str
    detector: List[str]
    sat_flow: float
    
    def __post_init__(self):
        if self.sat_flow < 0:
            raise ValueError("Saturation flow must be positive")
        if not isinstance(self.detector, list):
            raise TypeError("Detector must be a list")
    
    @classmethod
    def from_dict(cls, edge_id: str, data: Dict[str, Any]) -> 'EdgeInfo':
        """Create EdgeInfo from dictionary data"""
        return cls(
            edge_id=edge_id,
            detector=data.get("detector", []),
            sat_flow=data.get("sat_flow", 1800.0)  # default saturation flow
        )
