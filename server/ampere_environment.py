"""
Ampere Environment Implementation.

EV routing simulation for Indian highways using a Tata Nexon EV Creative MR.
"""
import math
import json
import os
import random
from uuid import uuid4
from typing import List, Dict, Optional

import numpy as np
import networkx as nx

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Absolute imports from the root models.py
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

VEHICLE = {
    "battery_capacity_kwh":       30.0,
    "base_consumption_wh_per_km": 136.0,
    "optimal_speed_kmh":          50.0,
    "max_charge_rate_kw":         50.0,
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
        self._shortest_paths: dict = {}
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
            available = [k for k in self._all_graph_data if k.startswith("task_")]
            raise ValueError(f"Unknown scenario_key '{scenario_key}'. Available tasks: {available}")

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

        try:
            self._shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.map_graph, weight="distance_km"))
        except Exception:
            self._shortest_paths = {}

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
        time_spent_this_step = 0.0

        if self.map_graph is None:
            raise RuntimeError("Call reset() before step()")

        if self.current_step_count > self.max_steps:
            return self._terminal_obs(-50.0, 0.0)

        valid_neighbors = list(self.map_graph.successors(self.current_node))
        if action.next_waypoint not in valid_neighbors:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 3:
                return self._terminal_obs(-100.0, 0.0)
            
            obs = self._build_observation()
            obs.reward = -10.0
            obs.done = False
            return obs

        if action.speed_mode not in SPEED_MODES:
            self.consecutive_errors += 1
            obs = self._build_observation()
            obs.reward = -5.0
            obs.done = False
            return obs

        self.consecutive_errors = 0

        prev_node    = self.current_node
        prev_battery = self.battery

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

        edge_data   = self.map_graph[self.current_node][action.next_waypoint]
        distance_km = edge_data["distance_km"]
        terrain     = edge_data.get("terrain", "flat")

        speed_kmh          = SPEED_MODES[action.speed_mode]
        terrain_multiplier = TERRAIN_MULTIPLIERS.get(terrain, 1.0)
        drag_multiplier    = (speed_kmh / VEHICLE["optimal_speed_kmh"]) ** 2

        energy_used_wh = VEHICLE["base_consumption_wh_per_km"] * distance_km * drag_multiplier * terrain_multiplier
        battery_drained_pct = (energy_used_wh / (VEHICLE["battery_capacity_kwh"] * 1000)) * 100.0
        self.battery -= battery_drained_pct

        drive_time_mins    = (distance_km / speed_kmh) * 60.0
        self.time_elapsed  += drive_time_mins
        self.fatigue       += drive_time_mins
        time_spent_this_step += drive_time_mins

        self.current_node  = action.next_waypoint

        self.battery = max(0.0, min(100.0, self.battery))
        self.fatigue = max(0.0, min(300.0, self.fatigue))

        stranded  = self.battery <= 0.0
        crashed   = self.fatigue >= 300.0
        reached   = self.current_node == self.end_node
        timed_out = self.current_step_count >= self.max_steps
        terminated = stranded or crashed or reached or timed_out

        reward = 0.0
        prev_remaining_km = self._shortest_paths.get(prev_node, {}).get(self.end_node, 9999)
        current_remaining_km = self._shortest_paths.get(self.current_node, {}).get(self.end_node, 9999)

        if prev_remaining_km != 9999 and current_remaining_km != 9999:
            reward += (prev_remaining_km - current_remaining_km) * 0.2

        adjusted_time_penalty = 0.02 if terrain_multiplier <= 1.0 else 0.01 
        reward -= adjusted_time_penalty * time_spent_this_step

        battery_used = prev_battery - self.battery
        reward -= 0.05 * max(0.0, battery_used)

        if self.fatigue > 200:
            reward -= 5.0
        elif self.fatigue > 150:
            reward -= 2.0

        nearest_km = self._get_nearest_charger_km()
        if self.battery < 20.0 and nearest_km > 100:
            reward -= 5.0

        eco_drain_pct_per_km = (VEHICLE["base_consumption_wh_per_km"] / (VEHICLE["battery_capacity_kwh"] * 1000)) * 100.0
        safe_battery_threshold = (nearest_km * eco_drain_pct_per_km) + 15.0 
        
        if action.charge_minutes > 0 and prev_battery < safe_battery_threshold and charger_worked:
            reward += 2.0

        if reached:
            ideal_battery = 10.0
            battery_diff = abs(self.battery - ideal_battery)
            destination_bonus = max(0.0, 20.0 - (0.5 * battery_diff))
            reward += destination_bonus

        if crashed or stranded:
            reward -= 50.0

        score = 0.0
        if terminated:
            score = self._calculate_final_grade(crashed, stranded, reached, timed_out)

        obs = self._build_observation(reached=reached, crashed=crashed, stranded=stranded, score=score)
        obs.reward = round(reward, 4)
        obs.done   = terminated
        return obs

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self, reached=False, crashed=False, stranded=False, score=0.0) -> EVObservation:
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

        dist_to_end = self._shortest_paths.get(self.current_node, {}).get(self.end_node, 9999)
        nearest_km, nearest_reliability = self._get_nearest_charger_info()

        # BYPASS SCHEMA CACHE: Inject the grader report directly into the optimal_heading string!
        heading = f"Head towards {self.end_node}"
        if reached or crashed or stranded or score > 0:
            heading = f"🏆 FINAL SCORE: {score} / 1.0  |  Reached: {reached}  |  Stranded: {stranded}  |  Crashed: {crashed}"

        gps = GPSDashboard(
            distance_to_final_destination_km = int(dist_to_end),
            distance_to_nearest_charger_km   = int(nearest_km),
            charger_reliability_estimate     = round(nearest_reliability, 2),
            optimal_heading                  = heading,
        )

        warning = "OK"
        if self.battery < 15.0:
            warning = "CRITICAL"
        elif self.battery < 30.0:
            warning = "WARNING"

        eco_drain_pct_per_km = (VEHICLE["base_consumption_wh_per_km"] / (VEHICLE["battery_capacity_kwh"] * 1000) * 100.0)
        max_range_eco = (self.battery / eco_drain_pct_per_km if eco_drain_pct_per_km > 0 else 0)
        can_reach = nearest_km <= max_range_eco

        eco_drag         = (SPEED_MODES["eco"] / VEHICLE["optimal_speed_kmh"]) ** 2
        eco_wh_per_km    = VEHICLE["base_consumption_wh_per_km"] * eco_drag
        
        worst_terrain_multiplier = 1.0
        for r in routes:
            if r.terrain == "mountain":
                worst_terrain_multiplier = TERRAIN_MULTIPLIERS["mountain"]
                break
        
        effective_wh_per_km = eco_wh_per_km * worst_terrain_multiplier
        remaining_wh     = (self.battery / 100.0) * (VEHICLE["battery_capacity_kwh"] * 1000)
        estimated_range  = int(remaining_wh / effective_wh_per_km) if effective_wh_per_km > 0 else 0

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
            reached_destination    = reached,
            crashed                = crashed,
            stranded               = stranded,
            final_grader_score     = score
    )
    
    def _terminal_obs(self, reward: float, grader_score: float) -> EVObservation:
        obs = self._build_observation(score=grader_score)
        obs.reward = reward
        obs.done   = True
        return obs

    def _get_nearest_charger_info(self) -> tuple:
        nearest_km          = 9999.0
        nearest_reliability = 1.0

        for node in self.map_graph.nodes:
            if node == self.current_node:
                continue
            
            nd = self.map_graph.nodes[node]
            if nd.get("charger_kw", 0) <= 0:
                continue
            
            d = self._shortest_paths.get(self.current_node, {}).get(node, 9999)
            if d < nearest_km:
                nearest_km          = d
                nearest_reliability = nd.get("reliability", 1.0)

        return nearest_km, nearest_reliability

    def _get_nearest_charger_km(self) -> float:
        km, _ = self._get_nearest_charger_info()
        return km

    def _calculate_final_grade(self, crashed: bool, stranded: bool, reached: bool, timed_out: bool) -> float:
        if not reached or crashed or stranded:
            return 0.0

        if self.time_elapsed <= self.deadline_mins:
            return 1.0

        minutes_late = self.time_elapsed - self.deadline_mins
        allowed_late_window = self.deadline_mins * 0.20

        if minutes_late >= allowed_late_window:
            return 0.3

        grade = 0.6 - (0.3 * (minutes_late / allowed_late_window))
        return round(grade, 2)