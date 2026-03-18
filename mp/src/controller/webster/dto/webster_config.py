from dataclasses import dataclass
from typing import Any, Dict
from .tls_info import TLSInfo


@dataclass
class WebsterConfig:
    """Data Transfer Object for MaxPressure controller configuration"""
    tls_info: TLSInfo
    sample_interval: float = 10.0
    cycling: str = "linear"
    
    def __post_init__(self):
        if self.sample_interval <= 0:
            raise ValueError("Sample interval must be positive")
        if self.cycling not in ["linear", "exponential"]:
            raise ValueError("Cycling method must be 'linear' or 'exponential'")
        if not isinstance(self.tls_info, TLSInfo):
            raise TypeError("tls_info must be an instance of TLSInfo")
    
    @classmethod
    def from_params(cls, **params: Any) -> 'WebsterConfig':
        """Create WebsterConfig from parameter dictionary"""
        tls_info_data = params.get("tls_info", {})
        if isinstance(tls_info_data, dict):
            tls_info = TLSInfo.from_dict(tls_info_data)
        elif isinstance(tls_info_data, TLSInfo):
            tls_info = tls_info_data
        else:
            raise TypeError("tls_info must be a dictionary or TLSInfo instance")
        
        sample_interval = params.get("sample_interval", 10.0)
        cycling = params.get("cycling", "linear")
        
        # Type validation
        if not isinstance(sample_interval, (int, float)):
            raise TypeError("sample_interval must be a number")
        if not isinstance(cycling, str):
            raise TypeError("cycling must be a string")
        
        return cls(
            tls_info=tls_info,
            sample_interval=float(sample_interval),
            cycling=cycling
        )
    
    @property
    def cycle_time(self) -> float:
        """Get cycle time from TLS info"""
        return self.tls_info.cycle
    
    @property
    def phases(self) -> Dict[str, Any]:
        """Get phases from TLS info"""
        return {phase_id: {
            "movements": phase.movements,
            "min-green": phase.min_green,
            "max-green": phase.max_green
        } for phase_id, phase in self.tls_info.phases.items()}
    
    @property
    def edges(self) -> Dict[str, Any]:
        """Get edges from TLS info"""
        return {edge_id: {
            "detector": edge.detector,
            "sat_flow": edge.sat_flow
        } for edge_id, edge in self.tls_info.edges.items()}
    
    @property
    def movements(self) -> Dict[str, Dict[str, float]]:
        """Get movements from TLS info"""
        return self.tls_info.movements
