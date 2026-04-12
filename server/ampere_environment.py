"""
Ampere Environment Implementation.

EV routing simulation for Indian highways using a Tata Nexon EV Creative MR.
The AI agent must route the vehicle from start to end while managing:
  - Battery drain (aerodynamic drag + terrain multiplier)
  - Driver fatigue (MORTH-compliant 300-point scale)
  - Stochastic charger failures (Task 3 only)
  - Tight per-task deadlines and step limits
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
    "battery_capacity_kwh":       45.0,
    "base_consumption_wh_per_km": 136.0,
    "optimal_speed_kmh":          50.0,
    "max_charge_rate_kw":         60.0,
}

DEFAULT_SCENARIO = "task_1_blr_cbe"

_HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DATA_PATH = os.path.join(_HERE, "..", "graph_data.json")


# ── Environment ───────────────────────────────────────────────────────────────

class AmpereEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        graph_path = GRAPH_DATA_PATH
        if not os.path.exists(graph_path):
            graph_path = os.path.join(_HERE, "graph_data.json")
        with open(graph_path, "r") as f:
            self._all_graph_data = json.load(f)

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

    def reset(self, scenario_key: str = DEFAULT_SCENARIO) -> EVObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._random = np.random.default_rng()
        self._scenario_key = scenario_key

        scenario = self._all_graph_data.get(scenario_key)
        if scenario is None:
            raise ValueError(f"Unknown scenario_key '{scenario_key}'")

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

    def step(self, action: EVAction) -> EVObservation:
        self._state.step_count += 1
        self.current_step_count += 1

        if self.map_graph is None:
            raise RuntimeError("Call reset() before step()")

        if self.current_step_count > self.max_steps:
            return self._terminal_obs(-50.0, "Terminated: max_steps exceeded", 0.01)

        # Allow the agent to stay at its current location to charge
        valid_neighbors = list(self.map_graph.successors(self.current_node))
        valid_waypoints = valid_neighbors + [self.current_node]
        
        if action.next_waypoint not in valid_waypoints:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                return self._terminal_obs(-100.0, "Terminated: 3 consecutive hallucinations.", 0.01)
            obs = self._build_observation()
            obs.reward = -10.0
            obs.done = False
            obs.metadata = {"error": f"Invalid waypoint '{action.next_waypoint}'.", "consecutive_errors": self.consecutive_errors}
            return obs

        if action.speed_mode not in SPEED_MODES:
            self.consecutive_errors += 1
            obs = self._build_observation()
            obs.reward = -5.0
            obs.done = False
            obs.metadata = {"error": f"Invalid speed_mode '{action.speed_mode}'.", "consecutive_errors": self.consecutive_errors}
            return obs

        self.consecutive_errors = 0

        # ── Physics ───────────────────────────────────────────────────────────
        prev_node    = self.current_node
        prev_battery = self.battery

        # If staying in place to charge, distance is 0
        if action.next_waypoint == self.current_node:
            distance_km = 0.0
            terrain = "flat"
        else:
            edge_data   = self.map_graph[self.current_node][action.next_waypoint]
            distance_km = edge_data["distance_km"]
            terrain     = edge_data.get("terrain", "flat")

        speed_kmh          = SPEED_MODES[action.speed_mode]
        terrain_multiplier = TERRAIN_MULTIPLIERS.get(terrain, 1.0)
        drag_multiplier    = (speed_kmh / VEHICLE["optimal_speed_kmh"]) ** 2

        energy_used_wh = (
            VEHICLE["base_consumption_wh_per_km"] * distance_km * drag_multiplier * terrain_multiplier
        )
        battery_drained_pct = (energy_used_wh / (VEHICLE["battery_capacity_kwh"] * 1000)) * 100.0

        self.battery -= battery_drained_pct

        drive_time_mins    = (distance_km / speed_kmh) * 60.0 if speed_kmh > 0 else 0
        self.time_elapsed  += drive_time_mins
        self.fatigue       += drive_time_mins

        self.current_node      = action.next_waypoint
        time_spent_this_step   = drive_time_mins

        # ── Charging ──────────────────────────────────────────────────────────
        node_data   = self.map_graph.nodes[self.current_node]
        charger_kw  = node_data.get("charger_kw", 0)
        reliability = node_data.get("reliability", 1.0)
        charger_worked = False

        if action.charge_minutes > 0 and charger_kw > 0:
            if self.stochastic and reliability < 1.0:
                charger_worked = self._random.random() < reliability
            else:
                charger_worked = True

            if charger_worked:
                effective_kw = min(charger_kw, VEHICLE["max_charge_rate_kw"])
                charge_rate_pct_per_min = (effective_kw / VEHICLE["battery_capacity_kwh"]) / 60.0 * 100.0
                charge_gained = charge_rate_pct_per_min * action.charge_minutes
                self.battery = min(100.0, self.battery + charge_gained)

            self.time_elapsed  += action.charge_minutes
            self.fatigue       -= action.charge_minutes * 3.0
            time_spent_this_step += action.charge_minutes

        if action.rest_minutes > 0:
            self.time_elapsed  += action.rest_minutes
            self.fatigue       -= action.rest_minutes * 3.0
            time_spent_this_step += action.rest_minutes

        self.battery = max(0.0, min(100.0, self.battery))
        self.fatigue = max(0.0, min(300.0, self.fatigue))

        # ── Terminal conditions ───────────────────────────────────────────────
        stranded  = self.battery <= 0.0
        crashed   = self.fatigue >= 300.0
        reached   = self.current_node == self.end_node
        timed_out = self.current_step_count >= self.max_steps
        terminated = stranded or crashed or reached or timed_out

        # ── Reward ─────────────────────────────────────────────
        reward = 0.0

        try:
            prev_remaining_km = nx.shortest_path_length(self.map_graph, prev_node, self.end_node, weight="distance_km")
            current_remaining_km = nx.shortest_path_length(self.map_graph, self.current_node, self.end_node, weight="distance_km")
            if prev_remaining_km != 9999 and current_remaining_km != 9999:
                reward += (prev_remaining_km - current_remaining_km) * 0.2
        except:
            pass

        reward -= 0.02 * time_spent_this_step
        reward -= 0.05 * max(0.0, prev_battery - self.battery)

        if self.battery < 25.0:
            reward -= (25.0 - self.battery) * 0.3
        if self.fatigue > 200:
            reward -= 5.0
        elif self.fatigue > 150:
            reward -= 2.0

        nearest_km = self._get_nearest_charger_km()
        if self.battery < 20.0 and nearest_km > 100:
            reward -= 5.0

        if action.charge_minutes > 0 and prev_battery < 40.0 and charger_worked:
            reward += 2.0

        if reached:
            ideal_battery = 10.0
            destination_bonus = max(0.0, 20.0 - (0.5 * abs(self.battery - ideal_battery)))
            reward += destination_bonus

        if crashed or stranded:
            reward -= 50.0

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

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self) -> EVObservation:
        routes = []
        
        # Add the current node so the agent knows it can stay put
        curr_nd = self.map_graph.nodes[self.current_node]
        routes.append(RouteOption(
            destination_node  = self.current_node,
            distance_km       = 0.0,
            has_fast_charger  = curr_nd.get("charger_type") == "fast_dc",
            has_slow_charger  = curr_nd.get("charger_type") == "slow_ac",
            has_rest_facility = curr_nd.get("has_rest_facility", False),
            terrain           = "flat",
        ))

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
            dist_to_end = int(nx.shortest_path_length(self.map_graph, self.current_node, self.end_node, weight="distance_km"))
        except:
            dist_to_end = 9999

        nearest_km, nearest_reliability = self._get_nearest_charger_info()
        gps = GPSDashboard(
            distance_to_final_destination_km = dist_to_end,
            distance_to_nearest_charger_km   = int(nearest_km),
            charger_reliability_estimate     = round(nearest_reliability, 2),
            optimal_heading                  = f"Head towards {self.end_node}",
        )

        warning = "CRITICAL" if self.battery < 15.0 else ("WARNING" if self.battery < 30.0 else "OK")

        eco_drain_pct_per_km = (VEHICLE["base_consumption_wh_per_km"] / (VEHICLE["battery_capacity_kwh"] * 1000) * 100.0)
        max_range_eco = self.battery / eco_drain_pct_per_km if eco_drain_pct_per_km > 0 else 0
        can_reach = nearest_km <= max_range_eco

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
        obs = self._build_observation()
        obs.reward = reward
        obs.done   = True
        obs.metadata = {"error": error, "final_grader_score": grader_score, "scenario": self._scenario_key, "total_steps": self.current_step_count}
        return obs

    def _get_nearest_charger_info(self) -> tuple:
        nearest_km          = 9999.0
        nearest_reliability = 1.0
        for node in self.map_graph.nodes:
            if node == self.current_node: continue
            nd = self.map_graph.nodes[node]
            if nd.get("charger_kw", 0) <= 0: continue
            try:
                d = nx.shortest_path_length(self.map_graph, self.current_node, node, weight="distance_km")
                if d < nearest_km:
                    nearest_km          = d
                    nearest_reliability = nd.get("reliability", 1.0)
            except:
                continue
        return nearest_km, nearest_reliability

    def _get_nearest_charger_km(self) -> float:
        return self._get_nearest_charger_info()[0]

    def _calculate_final_grade(self, crashed: bool, stranded: bool, reached: bool, timed_out: bool) -> float:
        route_completion = 0.0
        try:
            start_node = self._all_graph_data[self._scenario_key]["start_node"]
            total_dist = nx.shortest_path_length(self.map_graph, start_node, self.end_node, weight="distance_km")
            current_dist = nx.shortest_path_length(self.map_graph, self.current_node, self.end_node, weight="distance_km")
            route_completion = max(0.0, min(1.0, (total_dist - current_dist) / total_dist))
        except:
            pass

        base_score = route_completion * 0.70

        if reached and not (crashed or stranded):
            time_bonus = 0.0
            if self.time_elapsed <= self.deadline_mins:
                time_efficiency = (self.deadline_mins - self.time_elapsed) / self.deadline_mins
                time_bonus = time_efficiency * 0.15

            safety_bonus = 0.14
            total_score = base_score + time_bonus + safety_bonus
            return min(0.99, max(0.01, round(total_score, 3)))

        return min(0.99, max(0.01, round(base_score, 3)))