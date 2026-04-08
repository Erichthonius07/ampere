"""
Data models for the Ampere EV Routing Environment.

Defines the action and observation spaces for an AI agent
routing a Tata Nexon EV Creative MR across Indian highways
under real-world physics, fatigue, and charging constraints.

Vehicle: Tata Nexon EV Creative MR
  - Battery: 30 kWh
  - Base consumption: 136 Wh/km at optimal speed (50 km/h)
  - Real highway range: ~220km at eco, ~86km at cruise
  - Max charge rate: 50 kW (DC fast)
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Optional


# ── Speed Mode Constants (for reference in descriptions) ─────────────────────
# eco     = 50 km/h → ~220 km range  (drag multiplier 1.00)
# cruise  = 70 km/h → ~112 km range  (drag multiplier 1.96)
# highway = 90 km/h → ~67 km range   (drag multiplier 3.24)
# sport   = 110 km/h → ~45 km range  (drag multiplier 4.84)

# ── Terrain Multipliers (for reference) ──────────────────────────────────────
# flat     = 1.0x drain  (normal NH highway)
# mountain = 1.8x drain  (ghats, hill roads, NH10 Teesta Valley)
# urban    = 1.2x drain  (city stop-start traffic)


# ── Observation Sub-models ────────────────────────────────────────────────────

class RouteOption(Observation):
    """
    One road the agent can take from the current city.
    Always choose next_waypoint from this list only.
    """

    destination_node: str = Field(
        default="",
        description="Name of the next city or milestone you can drive to. Use this exact string in next_waypoint."
    )
    distance_km: int = Field(
        default=0,
        description="Distance to that node in km. Combined with your speed_mode this determines battery drain and drive time."
    )
    has_fast_charger: bool = Field(
        default=False,
        description="True if that node has a fast DC charger (50kW). Charges at ~2.45 percent per minute."
    )
    has_slow_charger: bool = Field(
        default=False,
        description="True if that node has a slow AC charger (7.2kW). Charges at only 0.353 percent per minute. Plan long stops."
    )
    has_rest_facility: bool = Field(
        default=False,
        description="True if that node has a dhaba, petrol pump, or hotel where the driver can rest comfortably."
    )
    terrain: str = Field(
        default="flat",
        description="Road terrain to that node. flat=normal highway (1x drain), mountain=ghat/hill road (1.8x drain), urban=city traffic (1.2x drain). Choose eco speed on mountain terrain."
    )


class GPSDashboard(Observation):
    """
    Multi-hop lookahead summary computed by the environment using Dijkstra.
    Use this to plan ahead beyond the immediately visible nodes.
    """

    distance_to_final_destination_km: int = Field(
        default=0,
        description="Remaining km to the final destination via shortest path."
    )
    distance_to_nearest_charger_km: int = Field(
        default=0,
        description="Shortest path distance in km to the nearest node with any charger. Plan your battery accordingly."
    )
    charger_reliability_estimate: float = Field(
        default=1.0,
        description="Probability the nearest charger is actually working (0.0 to 1.0). In Task 3, chargers may be offline on arrival."
    )
    optimal_heading: str = Field(
        default="",
        description="Plain English direction hint e.g. Head towards Gangtok via NH10. For navigation context only."
    )


# ── Main Observation ──────────────────────────────────────────────────────────

class EVObservation(Observation):
    """
    Everything the AI agent sees at each decision step.
    Read all fields carefully before choosing an action.
    """

    current_location: str = Field(
        default="",
        description="The city or milestone the vehicle is currently at. Your next_waypoint must be adjacent to this node."
    )
    battery_percentage: float = Field(
        default=100.0,
        description="Remaining battery 0.0 to 100.0. If this hits 0.0 the vehicle is stranded and the episode ends with score 0.0."
    )
    fatigue_points: float = Field(
        default=0.0,
        description="Driver fatigue 0.0 to 300.0. Increases by 1 point per minute of driving. Decreases by 3 points per minute of resting or charging. If this hits 300.0 the driver crashes and episode ends with score 0.0."
    )
    time_elapsed_minutes: float = Field(
        default=0.0,
        description="Total minutes elapsed since the trip started. Compare against the task deadline to know if you are on track."
    )
    available_routes: List[RouteOption] = Field(
        default_factory=list,
        description="List of adjacent nodes you can drive to next. Your next_waypoint MUST be one of these destination_node values."
    )
    navigation_system: GPSDashboard = Field(
        default_factory=GPSDashboard,
        description="GPS summary for multi-hop planning. Use distance_to_nearest_charger_km to decide whether to charge now or push forward."
    )
    battery_warning: str = Field(
        default="OK",
        description="OK if battery above 30 percent. WARNING if battery between 15 and 30 percent. CRITICAL if battery below 15 percent. On CRITICAL charge immediately or choose eco speed only."
    )
    can_reach_next_charger: bool = Field(
        default=True,
        description="True if your current battery is enough to reach the nearest charger at eco speed (50 km/h). If False you are at serious risk of stranding."
    )
    estimated_range_km: int = Field(
        default=0,
        description="Estimated remaining range in km at your last chosen speed mode. Use this to sanity-check if you can reach the next charger."
    )


# ── Action ────────────────────────────────────────────────────────────────────

class EVAction(Action):
    """
    Your decision at each step. Output all four fields.
    Invalid next_waypoint will be penalized. Repeated invalid actions terminate the episode.
    """

    next_waypoint: str = Field(
        default="",
        description="Name of the next node to drive to. Must exactly match one of the destination_node values in available_routes. Ghost nodes (e.g. NH44_G1) are valid waypoints with no charger."
    )
    speed_mode: str = Field(
        default="cruise",
        description="Driving speed mode. eco=50kmh gives ~220km range best for danger zones. cruise=70kmh gives ~112km range balanced choice. highway=90kmh gives ~67km range use only if charger is close. sport=110kmh gives ~45km range almost never use this. Mountain terrain multiplies drain by 1.8x so always use eco on mountain edges."
    )
    charge_minutes: int = Field(
        default=0,
        ge=0,
        le=480,
        description="Minutes to spend charging at current node before driving. Only has effect if node has a charger. Slow AC charger gives 0.353 percent per minute. Fast DC charger gives 2.45 percent per minute. Charging also reduces fatigue by 3 points per minute."
    )
    rest_minutes: int = Field(
        default=0,
        ge=0,
        le=480,
        description="Minutes to rest at current node before driving. Does not require a charger. Reduces fatigue by 3 points per minute. Use this at dhabas or rest stops when fatigue is above 150."
    )