"""
Ampere Environment Implementation.
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

SPEED_MODES = {"eco": 50, "cruise": 70, "highway": 90, "sport": 110}
TERRAIN_MULTIPLIERS = {"flat": 1.0, "mountain": 1.8, "urban": 1.2}

VEHICLE = {
    "battery_capacity_kwh": 45.0,
    "base_consumption_wh_per_km": 136.0,
    "optimal_speed_kmh": 50.0,
    "max_charge_rate_kw": 60.0,
}

DEFAULT_SCENARIO = "task_1_blr_cbe"
_HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DATA_PATH = os.path.join(_HERE, "..", "graph_data.json")


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
                edge["from"], edge["to"],
                distance_km=edge["distance_km"],
                terrain=edge.get("terrain", "flat"),
            )

        self.current_node = scenario["start_node"]
        self.end_node = scenario["end_node"]
        self.battery = float(scenario.get("initial_battery", 100.0))
        self.fatigue = 0.0
        self.time_elapsed = 0.0
        self.deadline_mins = float(scenario["deadline_mins"])
        self.max_steps = int(scenario["max_steps"])
        self.stochastic = bool(scenario.get("stochastic", False))
        self.consecutive_errors = 0
        self.current_step_count = 0

        return self._build_observation()

    def step(self, action: EVAction) -> EVObservation:
        self._state.step_count += 1
        self.current_step_count += 1

        if self.map_graph is None:
            raise RuntimeError("Call reset() before step()")

        if self.current_step_count > self.max_steps:
            return self._terminal_obs(-50.0, "Max steps exceeded", 0.01)

        valid_neighbors = list(self.map_graph.successors(self.current_node))
        valid_waypoints = valid_neighbors + [self.current_node]
        
        if action.next_waypoint not in valid_waypoints:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                return self._terminal_obs(-100.0, "3 consecutive hallucinations", 0.01)
            obs = self._build_observation()
            obs.reward = -10.0
            obs.metadata = {"error": f"Invalid waypoint '{action.next_waypoint}'"}
            return obs

        if action.speed_mode not in SPEED_MODES:
            self.consecutive_errors += 1
            obs = self._build_observation()
            obs.reward = -5.0
            obs.metadata = {"error": f"Invalid speed_mode '{action.speed_mode}'"}
            return obs

        self.consecutive_errors = 0

        # Physics
        prev_node = self.current_node
        prev_battery = self.battery
        
        if action.next_waypoint == self.current_node:
            distance_km = 0.0
            terrain = "flat"
        else:
            edge_data = self.map_graph[self.current_node][action.next_waypoint]
            distance_km = edge_data["distance_km"]
            terrain = edge_data.get("terrain", "flat")

        speed_kmh = SPEED_MODES[action.speed_mode]
        terrain_mult = TERRAIN_MULTIPLIERS.get(terrain, 1.0)
        drag_mult = (speed_kmh / VEHICLE["optimal_speed_kmh"]) ** 2

        energy_wh = VEHICLE["base_consumption_wh_per_km"] * distance_km * drag_mult * terrain_mult
        self.battery -= (energy_wh / (VEHICLE["battery_capacity_kwh"] * 1000)) * 100.0

        drive_time = (distance_km / speed_kmh) * 60.0 if speed_kmh > 0 else 0
        self.time_elapsed += drive_time
        self.fatigue += drive_time
        
        self.current_node = action.next_waypoint
        time_spent = drive_time

        # Charging
        node_data = self.map_graph.nodes[self.current_node]
        charger_kw = node_data.get("charger_kw", 0)
        reliability = node_data.get("reliability", 1.0)
        charger_worked = False

        if action.charge_minutes > 0 and charger_kw > 0:
            if self.stochastic and reliability < 1.0:
                charger_worked = self._random.random() < reliability
            else:
                charger_worked = True

            if charger_worked:
                kw = min(charger_kw, VEHICLE["max_charge_rate_kw"])
                rate = (kw / VEHICLE["battery_capacity_kwh"]) / 60.0 * 100.0
                self.battery = min(100.0, self.battery + rate * action.charge_minutes)

            self.time_elapsed += action.charge_minutes
            self.fatigue -= action.charge_minutes * 3.0
            time_spent += action.charge_minutes

        # Rest
        if action.rest_minutes > 0:
            self.time_elapsed += action.rest_minutes
            self.fatigue -= action.rest_minutes * 3.0
            time_spent += action.rest_minutes

        self.battery = max(0.0, min(100.0, self.battery))
        self.fatigue = max(0.0, min(300.0, self.fatigue))

        # Terminal Checks
        stranded = self.battery <= 0.0
        crashed = self.fatigue >= 300.0
        reached = self.current_node == self.end_node
        terminated = stranded or crashed or reached

        # Reward
        reward = 0.0
        try:
            prev_dist = nx.shortest_path_length(self.map_graph, prev_node, self.end_node, weight="distance_km")
            curr_dist = nx.shortest_path_length(self.map_graph, self.current_node, self.end_node, weight="distance_km")
            reward += (prev_dist - curr_dist) * 0.2
        except: pass
        
        reward -= 0.02 * time_spent
        if self.battery < 20.0: reward -= 5.0
        if crashed or stranded: reward -= 50.0
        if reached: reward += 20.0

        obs = self._build_observation()
        obs.reward = round(reward, 4)
        obs.done = terminated

        if terminated:
            grade = self._calculate_final_grade(crashed, stranded, reached)
            obs.metadata = {
                "final_grader_score": grade,
                "reached_destination": reached,
                "time_elapsed_minutes": round(self.time_elapsed, 1),
                "deadline_minutes": self.deadline_mins,
                "battery_remaining_pct": round(self.battery, 2),
                "fatigue_remaining": round(self.fatigue, 2),
            }

        return obs

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self) -> EVObservation:
        routes = []
        # Add current node (Stay option)
        curr_nd = self.map_graph.nodes[self.current_node]
        routes.append(RouteOption(
            destination_node=self.current_node, distance_km=0,
            has_fast_charger=curr_nd.get("charger_type") == "fast_dc",
            has_slow_charger=curr_nd.get("charger_type") == "slow_ac",
            has_rest_facility=curr_nd.get("has_rest_facility", False),
            terrain="flat"
        ))
        
        for neighbor in self.map_graph.successors(self.current_node):
            edge = self.map_graph[self.current_node][neighbor]
            nd = self.map_graph.nodes[neighbor]
            routes.append(RouteOption(
                destination_node=neighbor, distance_km=edge["distance_km"],
                has_fast_charger=nd.get("charger_type") == "fast_dc",
                has_slow_charger=nd.get("charger_type") == "slow_ac",
                has_rest_facility=nd.get("has_rest_facility", False),
                terrain=edge.get("terrain", "flat")
            ))

        dist_to_end = 9999
        try:
            dist_to_end = int(nx.shortest_path_length(self.map_graph, self.current_node, self.end_node, weight="distance_km"))
        except: pass

        gps = GPSDashboard(
            distance_to_final_destination_km=dist_to_end,
            distance_to_nearest_charger_km=0,
            optimal_heading=f"Head towards {self.end_node}",
            time_remaining_minutes=max(0.0, self.deadline_mins - self.time_elapsed)
        )

        return EVObservation(
            current_location=self.current_node,
            battery_percentage=round(self.battery, 2),
            fatigue_points=round(self.fatigue, 2),
            time_elapsed_minutes=round(self.time_elapsed, 2),
            available_routes=routes,
            navigation_system=gps,
            battery_warning="OK",
            can_reach_next_charger=True,
            estimated_range_km=int(self.battery / 0.302),
            charger_lookahead=self._get_charger_lookahead()
        )

    def _get_charger_lookahead(self) -> list:
        try:
            path = nx.shortest_path(self.map_graph, self.current_node, self.end_node, weight="distance_km")
        except: return []
        
        lookahead = []
        cum_dist = 0
        for i in range(len(path) - 1):
            n_a, n_b = path[i], path[i+1]
            cum_dist += self.map_graph[n_a][n_b]["distance_km"]
            nd = self.map_graph.nodes[n_b]
            if nd.get("charger_kw", 0) > 0:
                lookahead.append({
                    "node": n_b, "distance_km": cum_dist,
                    "charger_kw": nd["charger_kw"], "reliability": nd.get("reliability", 1.0),
                    "terrain_after": self.map_graph[n_b][path[i+2]].get("terrain", "flat") if i+2 < len(path) else "flat"
                })
        return lookahead

    def _terminal_obs(self, reward: float, error: str, grader_score: float) -> EVObservation:
        obs = self._build_observation()
        obs.reward = reward
        obs.done = True
        obs.metadata = {"error": error, "final_grader_score": grader_score}
        return obs

    def _calculate_final_grade(self, crashed: bool, stranded: bool, reached: bool) -> float:
        """
        Strict Multi-Factor Grading using deadlines from graph_data.json.
        Components:
        1. Completion (Base): Reached destination?
        2. Time (50%): Did you beat the deadline?
        3. Battery (25%): Safe arrival charge?
        4. Fatigue (25%): Driver safety?
        """
        if crashed or stranded or not reached:
            return 0.01
        
        # TIME COMPONENT (50%)
        deadline = self.deadline_mins
        elapsed = self.time_elapsed
        
        if elapsed <= deadline:
            time_score = 0.50
            if elapsed <= deadline * 0.5:
                time_score += 0.10
        else:
            minutes_late = elapsed - deadline
            lateness_ratio = minutes_late / deadline
            
            if lateness_ratio <= 0.25:
                time_score = 0.30
            elif lateness_ratio <= 0.50:
                time_score = 0.15
            elif lateness_ratio <= 1.0:
                time_score = 0.05
            else:
                time_score = 0.01

        # BATTERY COMPONENT (25%)
        battery_score = 0.25 if self.battery >= 20.0 else (0.15 if self.battery >= 10.0 else 0.05)

        # FATIGUE COMPONENT (25%)
        fatigue_score = 0.25 if self.fatigue <= 150 else (0.10 if self.fatigue <= 250 else 0.01)

        total_score = time_score + battery_score + fatigue_score
        return max(0.01, min(0.99, round(total_score, 2)))