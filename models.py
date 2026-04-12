"""
Data models for the Ampere EV Routing Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional, Dict, Any


class RouteOption(Observation):
    """One road the agent can take from the current city."""
    destination_node: str = Field(default="", description="Name of the next city.")
    distance_km: int = Field(default=0, description="Distance to that node in km.")
    has_fast_charger: bool = Field(default=False, description="True if that node has a fast DC charger.")
    has_slow_charger: bool = Field(default=False, description="True if that node has a slow AC charger.")
    has_rest_facility: bool = Field(default=False, description="True if that node has a dhaba or hotel.")
    terrain: str = Field(default="flat", description="Road terrain: flat, mountain, or urban.")


class ChargerInfo(Observation):
    """
    A charger node on the optimal path ahead.
    Critical for planning charging stops in deserts.
    """
    node: str = Field(default="", description="Name of the charger node.")
    distance_km: int = Field(default=0, description="Distance from current position.")
    charger_kw: int = Field(default=0, description="Charger power in kW.")
    reliability: float = Field(default=1.0, description="Probability charger is working.")
    terrain_after: str = Field(default="flat", description="Terrain after this charger.")


class GPSDashboard(Observation):
    """GPS summary for navigation."""
    distance_to_final_destination_km: int = Field(default=0)
    distance_to_nearest_charger_km: int = Field(default=0)
    nearest_charger_node: str = Field(default="unknown")
    charger_reliability_estimate: float = Field(default=1.0)
    optimal_heading: str = Field(default="")
    time_remaining_minutes: float = Field(default=0.0, description="Minutes remaining before deadline.")


class EVObservation(Observation):
    """Everything the AI agent sees at each decision step."""
    current_location: str = Field(default="")
    battery_percentage: float = Field(default=100.0)
    fatigue_points: float = Field(default=0.0)
    time_elapsed_minutes: float = Field(default=0.0)
    available_routes: List[RouteOption] = Field(default_factory=list)
    navigation_system: GPSDashboard = Field(default_factory=GPSDashboard)
    battery_warning: str = Field(default="OK")
    can_reach_next_charger: bool = Field(default=True)
    estimated_range_km: int = Field(default=0)
    
    # MANDATORY: This field was missing in your old file
    charger_lookahead: List[ChargerInfo] = Field(
        default_factory=list,
        description="Ordered list of upcoming charger nodes on the optimal path."
    )

    # Terminal state fields
    reached_destination: bool = Field(default=False)
    crashed: bool = Field(default=False)
    stranded: bool = Field(default=False)
    final_grader_score: float = Field(default=0.0)

    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EVAction(Action):
    """Your decision at each step."""
    next_waypoint: str = Field(default="", description="Name of the next node.")
    speed_mode: str = Field(default="cruise", description="eco, cruise, highway, or sport.")
    charge_minutes: int = Field(default=0, ge=0, le=480)
    rest_minutes: int = Field(default=0, ge=0, le=480)
    metadata: Dict[str, Any] = Field(default_factory=dict)