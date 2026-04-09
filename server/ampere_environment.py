"""
Ampere Environment Implementation.

EV routing simulation for Indian highways using a Tata Nexon EV Creative MR.
The AI agent must route the vehicle from start to end while managing:
  - Battery drain (aerodynamic drag + terrain multiplier)
  - Driver fatigue (MORTH-compliant 300-point scale)
  - Stochastic charger failures (Task 3 only)
  - Tight per-task deadlines and step limits

Physics based on real Nexon EV Creative MR specs:
  - Battery: 30 kWh
  - Base consumption: 136 Wh/km at optimal speed (50 km/h)
  - DC fast charge: 50 kW max → 2.45%/min
  - AC slow charge: 7.2 kW → 0.353%/min

Tasks:
  task_1_blr_cbe — Bangalore → Coimbatore  365km  deadline 420min  easy
  task_2_gwh_gtk — Guwahati  → Gangtok     540km  deadline 660min  medium  starts 80%
  task_3_knp_slg — Kanpur    → Siliguri   1110km  deadline 1680min hard    stochastic
"""

import json
import os
from uuid import uuid4

import networkx as nx
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EVAction, EVObservation, GPSDashboard, RouteOption
except ImportError:
    from models import EVAction, EVObservation, GPSDashboard, RouteOption


# ── Constants ─────────────────────────────────────────────────────────────────

SPEED_MODES = {
    "eco":     50,
    "cruise":  70,
    "highway": 90,
    "sport":   110,
}

TERRAIN_MULTIPLIERS = {
    "flat":     1.0,
    "mountain": 1.8,
    "urban":    1.2,
}

# Nexon EV Creative MR — real world specs
VEHICLE = {
    "battery_capacity_kwh":       30.0,
    "base_consumption_wh_per_km": 136.0,
    "optimal_speed_kmh":          50.0,
    "max_charge_rate_kw":         50.0,
}

DEFAULT_SCENARIO = "task_1_blr_cbe"

# Resolve graph_data.json relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DATA_PATH = os.path.join(_HERE, "..", "graph_data.json")


# ── Environment ───────────────────────────────────────────────────────────────

class AmpereEnvironment(Environment):
    """
    Ampere EV Routing Environment.

    Simulates routing a Tata Nexon EV Creative MR across Indian highways.
    Inherits from openenv Environment base class.

    The evaluator selects tasks by calling reset with a scenario_key.
    Each task has its own deadline, max_steps, and starting battery.

    Time constraints per task (enforced in step()):
      task_1_blr_cbe — deadline 420 min,  max_steps 24
      task_2_gwh_gtk — deadline 660 min,  max_steps 28
      task_3_knp_slg — deadline 1680 min, max_steps 50

    Grader scores:
      1.0      — reached destination on time
      0.3–0.6  — reached but late (interpolated over 120 min window)
      0.3      — reached but 120+ min late
      0.0      — stranded, crashed, or timed out
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Load graph data once at startup
        graph_path = GRAPH_DATA_PATH
        if not os.path.exists(graph_path):
            graph_path = os.path.join(_HERE, "graph_data.json")
        with open(graph_path, "r") as f:
            self._all_graph_data = json.load(f)

        # Episode state — set properly in reset()
        self.map_graph: nx.DiGraph = None
        self.current_node: str = ""
        self.end_node: str = ""
        self.battery: float = 100.0
        self.fatigue: float = 0.0
        self.time_elapsed: float = 0.0
        self.deadline_mins: float = 480.0
        self.max_steps: int = 30
        self.stochastic: bool = False
        self.consecutive_errors: int = 0
        self.current_step_count: int = 0
        self._random: np.random.Generator = np.random.default_rng()
        self._scenario_key: str = DEFAULT_SCENARIO

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, scenario_key: str = DEFAULT_SCENARIO) -> EVObservation:
        """
        Reset the environment for a new episode.

        Args:
            scenario_key: One of:
                task_1_blr_cbe  (easy,   365km,  deadline 420min,  100% battery)
                task_2_gwh_gtk  (medium, 540km,  deadline 660min,   80% battery)
                task_3_knp_slg  (hard,  1110km,  deadline 1680min, 100% battery)

        Returns:
            Initial EVObservation with current state and available routes.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._random = np.random.default_rng()
        self._scenario_key = scenario_key

        scenario = self._all_graph_data.get(scenario_key)
        if scenario is None:
            available = [k for k in self._all_graph_data if k.startswith("task_")]
            raise ValueError(
                f"Unknown scenario_key '{scenario_key}'. "
                f"Available tasks: {available}"
            )

        # Build directed graph from JSON
        self.map_graph = nx.DiGraph()
        for node_name, node_data in scenario["nodes"].items():
            self.map_graph.add_node(node_name, **node_data)
        for edge in scenario["edges"]:
            self.map_graph.add_edge(
                edge["from"],
                edge["to"],
                distance_km=edge["distance_km"],
                terrain=edge.get("terrain", "flat"),
            )

        # Set episode variables
        self.current_node         = scenario["start_node"]
        self.end_node             = scenario["end_node"]
        self.battery              = float(scenario.get("initial_battery", 100.0))
        self.fatigue              = 0.0
        self.time_elapsed         = 0.0
        self.deadline_mins        = float(scenario["deadline_mins"])
        self.max_steps            = int(scenario["max_steps"])
        self.stochastic           = bool(scenario.get("stochastic", False))
        self.consecutive_errors   = 0
        self.current_step_count   = 0

        return self._build_observation()

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: EVAction) -> EVObservation:
        """
        Execute one action in the environment.

        Sequence:
          1. LLM Firewall (waypoint + speed_mode + step limit)
          2. Physics (battery drain, drive time, fatigue)
          3. Charging (BMS-limited, stochastic failure in Task 3)
          4. Rest (fatigue recovery)
          5. Value clamping
          6. Terminal condition checks
          7. 8-component reward calculation
          8. Observation construction

        Returns:
            EVObservation with reward and done flag set.
            On terminal steps, metadata contains final_grader_score.
        """
        self._state.step_count += 1
        self.current_step_count += 1

        # ── LLM Firewall ──────────────────────────────────────────────────────

        if self.map_graph is None:
            raise RuntimeError("Call reset() before step()")

        # Check 1: step limit
        if self.current_step_count > self.max_steps:
            return self._terminal_obs(
                reward=-50.0,
                error="Terminated: max_steps exceeded",
                grader_score=0.01,
            )

        # Check 2: waypoint must be adjacent
        valid_neighbors = list(self.map_graph.successors(self.current_node))
        if action.next_waypoint not in valid_neighbors:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                return self._terminal_obs(
                    reward=-100.0,
                    error=f"Terminated: 3 consecutive hallucinations. "
                          f"Last bad waypoint: '{action.next_waypoint}'",
                    grader_score=0.01,
                )
            obs = self._build_observation()
            obs.reward = -10.0
            obs.done = False
            obs.metadata = {
                "error": f"Invalid waypoint '{action.next_waypoint}'. "
                         f"Valid options: {valid_neighbors}",
                "consecutive_errors": self.consecutive_errors,
            }
            return obs

        # Check 3: speed_mode must be valid
        if action.speed_mode not in SPEED_MODES:
            self.consecutive_errors += 1
            obs = self._build_observation()
            obs.reward = -5.0
            obs.done = False
            obs.metadata = {
                "error": f"Invalid speed_mode '{action.speed_mode}'. "
                         f"Must be one of: {list(SPEED_MODES.keys())}",
                "consecutive_errors": self.consecutive_errors,
            }
            return obs

        # Valid action — reset error counter
        self.consecutive_errors = 0

        # ── Physics ───────────────────────────────────────────────────────────

        # Save previous state for reward calculation
        prev_node    = self.current_node
        prev_battery = self.battery

        edge_data   = self.map_graph[self.current_node][action.next_waypoint]
        distance_km = edge_data["distance_km"]
        terrain     = edge_data.get("terrain", "flat")

        speed_kmh          = SPEED_MODES[action.speed_mode]
        terrain_multiplier = TERRAIN_MULTIPLIERS.get(terrain, 1.0)
        drag_multiplier    = (speed_kmh / VEHICLE["optimal_speed_kmh"]) ** 2

        # Battery drain for this leg
        energy_used_wh = (
            VEHICLE["base_consumption_wh_per_km"]
            * distance_km
            * drag_multiplier
            * terrain_multiplier
        )
        battery_drained_pct = (
            energy_used_wh / (VEHICLE["battery_capacity_kwh"] * 1000)
        ) * 100.0

        self.battery -= battery_drained_pct

        # Drive time and fatigue
        drive_time_mins    = (distance_km / speed_kmh) * 60.0
        self.time_elapsed  += drive_time_mins
        self.fatigue       += drive_time_mins   # +1 fatigue per minute driving

        # Move vehicle
        self.current_node      = action.next_waypoint
        time_spent_this_step   = drive_time_mins

        # ── Charging ──────────────────────────────────────────────────────────

        node_data   = self.map_graph.nodes[self.current_node]
        charger_kw  = node_data.get("charger_kw", 0)
        reliability = node_data.get("reliability", 1.0)
        charger_worked = False

        if action.charge_minutes > 0 and charger_kw > 0:
            # Stochastic failure roll — Task 3 only
            if self.stochastic and reliability < 1.0:
                charger_worked = self._random.random() < reliability
            else:
                charger_worked = True

            if charger_worked:
                effective_kw = min(charger_kw, VEHICLE["max_charge_rate_kw"])
                charge_rate_pct_per_min = (
                    effective_kw / VEHICLE["battery_capacity_kwh"]
                ) / 60.0 * 100.0
                charge_gained = charge_rate_pct_per_min * action.charge_minutes
                self.battery = min(100.0, self.battery + charge_gained)

            # Time passes and fatigue recovers regardless of charger status
            self.time_elapsed  += action.charge_minutes
            self.fatigue       -= action.charge_minutes * 3.0
            time_spent_this_step += action.charge_minutes

        # ── Rest ──────────────────────────────────────────────────────────────

        if action.rest_minutes > 0:
            self.time_elapsed  += action.rest_minutes
            self.fatigue       -= action.rest_minutes * 3.0
            time_spent_this_step += action.rest_minutes

        # ── Clamp values ──────────────────────────────────────────────────────

        self.battery = max(0.0, min(100.0, self.battery))
        self.fatigue = max(0.0, min(300.0, self.fatigue))

        # ── Terminal conditions ───────────────────────────────────────────────

        stranded  = self.battery <= 0.0
        crashed   = self.fatigue >= 300.0
        reached   = self.current_node == self.end_node
        timed_out = self.current_step_count >= self.max_steps

        terminated = stranded or crashed or reached or timed_out

        # ── Reward (8 components) ─────────────────────────────────────────────

        reward = 0.0

        # 1 & 2. Potential-based progress shaping
        # Rewards moving closer, penalises backtracking/looping
        try:
            prev_remaining_km = nx.shortest_path_length(
                self.map_graph, prev_node, self.end_node,
                weight="distance_km",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            prev_remaining_km = 9999

        try:
            current_remaining_km = nx.shortest_path_length(
                self.map_graph, self.current_node, self.end_node,
                weight="distance_km",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            current_remaining_km = 9999

        if prev_remaining_km != 9999 and current_remaining_km != 9999:
            reward += (prev_remaining_km - current_remaining_km) * 0.2

        # 3. Time pressure — discourage unnecessary delay
        reward -= 0.02 * time_spent_this_step

        # 4. Battery efficiency — penalise aggressive driving
        battery_used = prev_battery - self.battery
        reward -= 0.05 * max(0.0, battery_used)

        # 5. Fatigue management — proactive penalty before crash
        if self.fatigue > 200:
            reward -= 5.0
        elif self.fatigue > 150:
            reward -= 2.0

        # 6. Risk awareness — low battery near charger desert
        nearest_km = self._get_nearest_charger_km()
        if self.battery < 20.0 and nearest_km > 100:
            reward -= 5.0

        # 7. Smart charging — reward charging when battery is strategically low
        # Simple threshold: charge when below 40% and charger worked
        if action.charge_minutes > 0 and prev_battery < 40.0 and charger_worked:
            reward += 2.0

        # 8. Destination optimisation — smooth bonus curve
        # Peak +20 at exactly 10% battery, tapers to 0 as deviation increases
        if reached:
            ideal_battery = 10.0
            battery_diff = abs(self.battery - ideal_battery)
            destination_bonus = max(0.0, 20.0 - (0.5 * battery_diff))
            reward += destination_bonus

        # Terminal failure penalty
        if crashed or stranded:
            reward -= 50.0

        # ── Build observation ─────────────────────────────────────────────────

        obs = self._build_observation()
        obs.reward = round(reward, 4)
        obs.done   = terminated

        if terminated:
            grade = self._calculate_final_grade(crashed, stranded, reached, timed_out)
            obs.metadata = {
                "final_grader_score":    grade,
                "scenario":              self._scenario_key,
                "time_elapsed_minutes":  round(self.time_elapsed, 1),
                "deadline_minutes":      self.deadline_mins,
                "minutes_over_deadline": max(0.0, round(self.time_elapsed - self.deadline_mins, 1)),
                "battery_remaining_pct": round(self.battery, 2),
                "fatigue_remaining":     round(self.fatigue, 2),
                "reached_destination":   reached,
                "stranded":              stranded,
                "crashed":               crashed,
                "timed_out":             timed_out,
                "total_steps":           self.current_step_count,
            }

        return obs

    # ── State property ────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_observation(self) -> EVObservation:
        """Build EVObservation from current state using NetworkX Dijkstra."""

        routes = []
        for neighbor in self.map_graph.successors(self.current_node):
            edge = self.map_graph[self.current_node][neighbor]
            nd   = self.map_graph.nodes[neighbor]
            routes.append(RouteOption(
                destination_node  = neighbor,
                distance_km       = edge["distance_km"],
                has_fast_charger  = nd.get("charger_type") == "fast_dc",
                has_slow_charger  = nd.get("charger_type") == "slow_ac",
                has_rest_facility = nd.get("has_rest_facility", False),
                terrain           = edge.get("terrain", "flat"),
            ))

        try:
            dist_to_end = int(nx.shortest_path_length(
                self.map_graph, self.current_node, self.end_node,
                weight="distance_km",
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist_to_end = 9999

        nearest_km, nearest_reliability = self._get_nearest_charger_info()

        gps = GPSDashboard(
            distance_to_final_destination_km = dist_to_end,
            distance_to_nearest_charger_km   = int(nearest_km),
            charger_reliability_estimate      = round(nearest_reliability, 2),
            optimal_heading                   = f"Head towards {self.end_node}",
        )

        # Battery warning
        if self.battery < 15.0:
            warning = "CRITICAL"
        elif self.battery < 30.0:
            warning = "WARNING"
        else:
            warning = "OK"

        # Can reach nearest charger at eco speed?
        eco_drain_pct_per_km = (
            VEHICLE["base_consumption_wh_per_km"]
            / (VEHICLE["battery_capacity_kwh"] * 1000)
            * 100.0
        )
        max_range_eco = (
            self.battery / eco_drain_pct_per_km
            if eco_drain_pct_per_km > 0 else 0
        )
        can_reach = nearest_km <= max_range_eco

        # Estimated range at cruise speed
        cruise_drag      = (SPEED_MODES["cruise"] / VEHICLE["optimal_speed_kmh"]) ** 2
        cruise_wh_per_km = VEHICLE["base_consumption_wh_per_km"] * cruise_drag
        remaining_wh     = (self.battery / 100.0) * (VEHICLE["battery_capacity_kwh"] * 1000)
        estimated_range  = int(remaining_wh / cruise_wh_per_km) if cruise_wh_per_km > 0 else 0

        return EVObservation(
            current_location       = self.current_node,
            battery_percentage     = round(self.battery, 2),
            fatigue_points         = round(self.fatigue, 2),
            time_elapsed_minutes   = round(self.time_elapsed, 2),
            available_routes       = routes,
            navigation_system      = gps,
            battery_warning        = warning,
            can_reach_next_charger = can_reach,
            estimated_range_km     = estimated_range,
        )

    def _terminal_obs(self, reward: float, error: str, grader_score: float) -> EVObservation:
        """Helper to build a terminal observation with grader score."""
        obs = self._build_observation()
        obs.reward = reward
        obs.done   = True
        obs.metadata = {
            "error":              error,
            "final_grader_score": grader_score,
            "scenario":           self._scenario_key,
            "total_steps":        self.current_step_count,
        }
        return obs

    def _get_nearest_charger_info(self) -> tuple:
        """Returns (distance_km, reliability) to nearest charger via Dijkstra."""
        nearest_km          = 9999.0
        nearest_reliability = 1.0

        for node in self.map_graph.nodes:
            if node == self.current_node:
                continue
            nd = self.map_graph.nodes[node]
            if nd.get("charger_kw", 0) <= 0:
                continue
            try:
                d = nx.shortest_path_length(
                    self.map_graph, self.current_node, node,
                    weight="distance_km",
                )
                if d < nearest_km:
                    nearest_km          = d
                    nearest_reliability = nd.get("reliability", 1.0)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        return nearest_km, nearest_reliability

    def _get_nearest_charger_km(self) -> float:
        km, _ = self._get_nearest_charger_info()
        return km

    def _calculate_final_grade(
        self,
        crashed:   bool,
        stranded:  bool,
        reached:   bool,
        timed_out: bool,
    ) -> float:
        assert self.time_elapsed >= 0, "time_elapsed sanity check failed"

        # 1. Terminal Failures (0.0)
        # Explicitly check timed_out so agents driving in circles get 0.0
        if crashed or stranded or not reached or (timed_out and not reached):
            return 0.0

        # 2. Perfect Score (1.0)
        if self.time_elapsed <= self.deadline_mins:
            return 1.0

        # 3. The Late Tax (0.3 to 0.6)
        # Smooth, continuous penalty. 1 minute late = 0.60, 120 minutes late = 0.30
        minutes_late = self.time_elapsed - self.deadline_mins
        
        if minutes_late >= 120:
            return 0.30
            
        grade = 0.60 - (0.30 * (minutes_late / 120.0))
        return round(grade, 2)