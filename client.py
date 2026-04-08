"""Ampere Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EVAction, EVObservation


class AmpereEnv(EnvClient[EVAction, EVObservation, State]):
    """
    Client for the Ampere EV Routing Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with AmpereEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_location)
        ...
        ...     action = EVAction(
        ...         next_waypoint="Hosur",
        ...         speed_mode="cruise",
        ...         charge_minutes=0,
        ...         rest_minutes=0,
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.battery_percentage)
    """

    def _step_payload(self, action: EVAction) -> Dict:
        """Convert EVAction to JSON payload for step message."""
        return {
            "next_waypoint":  action.next_waypoint,
            "speed_mode":     action.speed_mode,
            "charge_minutes": action.charge_minutes,
            "rest_minutes":   action.rest_minutes,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EVObservation]:
        """Parse server response into StepResult[EVObservation]."""
        obs_data = payload.get("observation", {})

        from .models import GPSDashboard, RouteOption

        routes = [
            RouteOption(**r)
            for r in obs_data.get("available_routes", [])
        ]

        nav = obs_data.get("navigation_system", {})
        gps = GPSDashboard(
            distance_to_final_destination_km=nav.get("distance_to_final_destination_km", 0),
            distance_to_nearest_charger_km=nav.get("distance_to_nearest_charger_km", 0),
            charger_reliability_estimate=nav.get("charger_reliability_estimate", 1.0),
            optimal_heading=nav.get("optimal_heading", ""),
        )

        observation = EVObservation(
            current_location=obs_data.get("current_location", ""),
            battery_percentage=obs_data.get("battery_percentage", 100.0),
            fatigue_points=obs_data.get("fatigue_points", 0.0),
            time_elapsed_minutes=obs_data.get("time_elapsed_minutes", 0.0),
            available_routes=routes,
            navigation_system=gps,
            battery_warning=obs_data.get("battery_warning", "OK"),
            can_reach_next_charger=obs_data.get("can_reach_next_charger", True),
            estimated_range_km=obs_data.get("estimated_range_km", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )