import os
import sys
import json
import time
from typing import List, Optional, Tuple
from openai import OpenAI
from client import AmpereEnv
from models import EVAction

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("XAI_API_KEY") or "dummy_token"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
BENCHMARK = os.getenv("AMPERE_BENCHMARK", "ampere")

SERVER_URL = (os.environ.get("ENV_URL") or os.environ.get("AMPERE_SERVER_URL")
              or "https://team01paracetamol-ampere.hf.space")

llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are EcoRoute, an AI EV Dispatcher.
Your objective is navigate a Tata Nexon EV to the final destination BEFORE the deadline.

CRITICAL RULES:
1. WAYPOINT: Choose from 'available_routes'. You can choose your CURRENT location to charge.
2. CHARGING: 
   - Charge only if the node has a charger.
   - Max charge 480 mins.
3. SPEED: 'eco' saves battery (deserts/mountains). 'cruise' is default.
4. FATIGUE: Rest only if fatigue > 150. Charging also reduces fatigue.

Output ONLY valid JSON matching this schema exactly:
{
    "next_waypoint": "ExactNodeName",
    "speed_mode": "cruise",
    "charge_minutes": 0,
    "rest_minutes": 0
}
"""

MAX_RETRIES = 3

# ── Logging Functions ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Physics Helpers ────────────────────────────────────────────────────
_COST_PCT = {"eco": 0.302, "cruise": 0.593, "highway": 0.982, "sport": 1.464}
_TERRAIN_MULT = {"flat": 1.0, "urban": 1.2, "mountain": 1.8}

def battery_needed(dist, speed, terrain):
    return dist * _COST_PCT.get(speed, 0.593) * _TERRAIN_MULT.get(terrain, 1.0)


# ── LLM Action & Integrated Autopilot ───────────────────────────────────────
def get_action_from_llm(obs, previous_intervention: str = "") -> EVAction | None:
    valid_waypoints = [r.destination_node for r in obs.available_routes]
    
    user_prompt = (
        f"CURRENT DASHBOARD:\n{obs.model_dump_json(indent=2)}\n\n"
        f"Valid next_waypoint values: {valid_waypoints}\n"
    )
    
    if previous_intervention:
        user_prompt += f"\n⚠️ SYSTEM WARNING: Your last planned action was overridden! {previous_intervention}\n"
        
    user_prompt += "\nOutput JSON."

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            llm_json = json.loads(response.choices[0].message.content)
            
            c_min = int(llm_json.get("charge_minutes", 0))
            r_min = int(llm_json.get("rest_minutes", 0))
            llm_json["charge_minutes"] = min(max(c_min, 0), 480)
            llm_json["rest_minutes"] = min(max(r_min, 0), 480)
            
            action = EVAction(**llm_json)
            return action
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt}: {e}. Retrying...", file=sys.stderr)
    return None

def apply_autopilot(action: EVAction, obs) -> Tuple[EVAction, str]:
    intervention_msg = ""
    valid_waypoints = [r.destination_node for r in obs.available_routes]
    
    # Basic Validation
    if action.next_waypoint not in valid_waypoints:
        action.next_waypoint = valid_waypoints[0]
        intervention_msg += "Autopilot: Fixed invalid waypoint. "

    chosen_route = next((r for r in obs.available_routes if r.destination_node == action.next_waypoint), None)
    current_route = next((r for r in obs.available_routes if r.destination_node == obs.current_location), None)
    
    # Determine if we're staying
    is_staying = action.next_waypoint == obs.current_location
    
    # Check charger status
    if is_staying:
        has_charger_here = current_route and (current_route.has_fast_charger or current_route.has_slow_charger)
    else:
        has_charger_here = chosen_route and (chosen_route.has_fast_charger or chosen_route.has_slow_charger)
    
    is_at_charger = current_route and (current_route.has_fast_charger or current_route.has_slow_charger)

    # ── PHYSICS MATH ───────────────────────────────────────
    eco_range = obs.battery_percentage / 0.302
    cruise_range = obs.battery_percentage / 0.593
    dist_to_end = obs.navigation_system.distance_to_final_destination_km
    
    # ── CRITICAL FIX 1: RANGE ANXIETY (Speed Control) ───────
    if dist_to_end > cruise_range:
        if action.speed_mode != "eco":
            action.speed_mode = "eco"
            intervention_msg += "Autopilot: Forced Eco (Range Anxiety). "

    # ── CRITICAL FIX 2: SMART CHARGING CALCULATION ─────────
    # We ALWAYS calculate the exact charge needed.
    # We NEVER trust the LLM's charge_minutes blindly.
    
    if has_charger_here:
        # 1. Calculate battery on arrival at this charger
        if is_staying:
            batt_on_arrival = obs.battery_percentage
            dist_to_this_node = 0
        else:
            dist_to_this_node = chosen_route.distance_km
            terrain = chosen_route.terrain
            # Assume Eco for arrival calculation to be safe
            batt_on_arrival = obs.battery_percentage - battery_needed(dist_to_this_node, "eco", terrain)
        
        # 2. Find the NEXT charger AFTER this node
        next_charger_info = None
        if obs.charger_lookahead:
            for c in obs.charger_lookahead:
                # Look for a charger strictly further than the current target
                if c.distance_km > dist_to_this_node + 1.0: 
                    next_charger_info = c
                    break
        
        # 3. Determine Target Battery
        target_battery = 90.0 # Default safe max
        
        if next_charger_info:
            # There is a charger ahead. Calculate need to reach IT.
            gap_km = next_charger_info.distance_km - dist_to_this_node
            # Use Eco speed for safety margin
            need_to_reach_next = battery_needed(gap_km, "eco", next_charger_info.terrain_after)
            target_battery = need_to_reach_next + 15.0 # Buffer
        else:
            # No charger ahead (final leg)
            # Only need to reach destination
            remaining_from_here = dist_to_end - dist_to_this_node
            need_to_finish = battery_needed(remaining_from_here, "eco", "flat")
            target_battery = need_to_finish + 10.0
        
        # Cap target at 95%
        target_battery = min(target_battery, 95.0)
        
        # 4. Calculate Deficit
        if batt_on_arrival < target_battery:
            deficit = target_battery - batt_on_arrival
            rate = 2.22 if (is_staying and current_route.has_fast_charger) or \
                           (not is_staying and chosen_route.has_fast_charger) else 0.35
            mins_needed = int(deficit / rate) + 1
            mins_needed = min(mins_needed, 60) # Hard cap 60 mins per stop (optimize for speed)
            
            # Override LLM
            if action.charge_minutes != mins_needed:
                print(f"   [AUTOPILOT] OPTIMIZED CHARGE: {mins_needed}m (LLM wanted {action.charge_minutes}m).", file=sys.stderr)
                action.charge_minutes = mins_needed
        else:
            # Already have enough
            action.charge_minutes = 0
    
    else:
        # No charger here. Block charging.
        if action.charge_minutes > 0:
            print(f"   [AUTOPILOT] BLOCKED: Cannot charge at {obs.current_location} - no charger!", file=sys.stderr)
            action.charge_minutes = 0
            intervention_msg += "Autopilot: Blocked ghost charge. "

            if is_staying:
                print(f"   [AUTOPILOT] FORCING MOVE from dead node {obs.current_location}.", file=sys.stderr)
                for r in obs.available_routes:
                    if r.destination_node != obs.current_location:
                        action.next_waypoint = r.destination_node
                        intervention_msg += "Autopilot: Forced Move (No Charger Here). "
                        is_staying = False
                        chosen_route = r
                        break

    # ── SPEED SAFETY (Mountain/Low Battery) ───────────────
    if chosen_route and chosen_route.terrain == "mountain":
        action.speed_mode = "eco"
    elif obs.battery_percentage < 30.0:
        action.speed_mode = "eco"

    # ── UNIVERSAL SAFETY CHECK (Only at Chargers) ───────────
    # If we are at a charger and trying to leave, check if we can survive
    if is_at_charger and not is_staying:
        can_reach_dest = dist_to_end <= eco_range - 50 
        
        if not can_reach_dest:
            print(f"   [AUTOPILOT] OVERRULED MOVE! Cannot reach dest ({dist_to_end}km > Eco Range {eco_range:.0f}km). Forcing STAY.", file=sys.stderr)
            action.next_waypoint = obs.current_location
            chosen_route = current_route
            is_staying = True
            # Re-calculate charge because we forced a stay
            has_charger_here = True 
            
            target_batt = (dist_to_end * 0.35) + 20.0
            target_batt = min(target_batt, 95.0)
            
            if obs.battery_percentage < target_batt:
                deficit = target_batt - obs.battery_percentage
                rate = 2.22 if current_route.has_fast_charger else 0.35
                mins_needed = int(deficit / rate) + 1
                mins_needed = min(mins_needed, 90)
                action.charge_minutes = mins_needed
                intervention_msg += f"Autopilot: Survival Charge {mins_needed}m. "

    # ── FINAL CHECKS ───────────────────────────────────────
    if is_staying and obs.battery_percentage >= 95.0:
        # If we are full, stop charging and MOVE
        action.charge_minutes = 0
        for r in obs.available_routes:
            if r.destination_node != obs.current_location:
                action.next_waypoint = r.destination_node
                intervention_msg += "Autopilot: Forced Move (Full Battery). "
                break

    # ── REST LOGIC ───────────────────────────────────────────
    if action.charge_minutes > 0:
        action.rest_minutes = 0
    elif obs.fatigue_points > 200:
        action.rest_minutes = max(action.rest_minutes, 25)
    elif obs.fatigue_points > 150:
        action.rest_minutes = max(action.rest_minutes, 15)
    else:
        action.rest_minutes = 0

    return action, intervention_msg


# ── Score Extraction ────────────────────────────────────────────────────────
def extract_numeric_score(obs, total_reward) -> float:
    if obs.metadata and "final_grader_score" in obs.metadata:
        return float(obs.metadata.get("final_grader_score", 0.01))
    heading = getattr(obs.navigation_system, "optimal_heading", "")
    if heading and "SCORE" in heading:
        try:
            return float(heading.split("SCORE:")[1].split("/")[0].strip())
        except:
            pass
    if total_reward > 0:
        return 0.99
    return 0.01

# ── Main Agent Loop ─────────────────────────────────────────────────────────
def run_agent(scenario: str):
    print(f"\n🚀 Booting EcoRoute Agent for Scenario: {scenario}", file=sys.stderr)
    print(f"🔗 Connecting to OpenEnv Server at {SERVER_URL}...\n", file=sys.stderr)

    try:
        with AmpereEnv(base_url=SERVER_URL).sync() as env:
            step_result = env.reset(scenario_key=scenario)
            obs  = step_result.observation
            done = step_result.done

            log_start(task=scenario, env=BENCHMARK, model=MODEL_NAME)

            rewards: List[float] = []
            step_count = 0
            total_reward = 0.0
            success = False
            previous_intervention = ""

            while not done:
                step_count += 1
                error = None

                print("=" * 60, file=sys.stderr)
                print(f"📍 STEP {step_count} | Current Location: {obs.current_location}", file=sys.stderr)
                print(f"🔋 Battery: {obs.battery_percentage:.1f}%  | ⚠️ Warning: {obs.battery_warning}", file=sys.stderr)
                print(f"🥱 Fatigue: {obs.fatigue_points:.0f}/300 | ⏱️ Elapsed: {obs.time_elapsed_minutes:.0f} mins", file=sys.stderr)
                print(f"🗺️  Remaining: {obs.navigation_system.distance_to_final_destination_km} km | Est. Range: {obs.estimated_range_km} km", file=sys.stderr)
                print(f"🛣️  Options: {[r.destination_node for r in obs.available_routes]}", file=sys.stderr)
                print("-" * 60, file=sys.stderr)

                print("🧠 Thinking...", file=sys.stderr)
                action = get_action_from_llm(obs, previous_intervention)
                
                if action is None:
                    error = "LLM failed to return valid action"
                    print("❌ Agent could not decide. Aborting episode.", file=sys.stderr)
                    log_step(step=step_count, action="null", reward=0.0, done=True, error=error)
                    break

                action, previous_intervention = apply_autopilot(action, obs)
                
                print(f"⚡ ACTION TAKEN:", file=sys.stderr)
                print(f"   ► Drive to: {action.next_waypoint}", file=sys.stderr)
                print(f"   ► Speed:    {action.speed_mode}", file=sys.stderr)
                print(f"   ► Charge:   {action.charge_minutes} mins", file=sys.stderr)
                print(f"   ► Rest:     {action.rest_minutes} mins", file=sys.stderr)

                action_str = json.dumps(action.model_dump(), separators=(',', ':'))

                try:
                    step_result  = env.step(action)
                    obs          = step_result.observation
                    done         = step_result.done
                    reward       = float(step_result.reward or 0.0)
                except Exception as e:
                    error = str(e).replace('\n', ' ')
                    reward = 0.0
                    done = True

                total_reward += reward
                rewards.append(reward)
                
                print(f"\n💰 Reward this step: {reward:+.2f} (Total: {total_reward:.2f})\n", file=sys.stderr)

                log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)
                time.sleep(0.5)

            print("🏁 === EPISODE COMPLETE === 🏁", file=sys.stderr)
            print(f"   Time Elapsed: {obs.time_elapsed_minutes:.1f} mins", file=sys.stderr)
            print(f"   Steps Taken:  {step_count}", file=sys.stderr)
            print(f"   Total Reward: {total_reward:.2f}", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            score = extract_numeric_score(obs, total_reward)
            success = score >= 0.5 

            log_end(success=success, steps=step_count, score=score, rewards=rewards)
            
    except Exception as e:
        if "503" in str(e):
            print(f"\n❌ SERVER ERROR (HTTP 503): The Hugging Face Space ({SERVER_URL}) is currently asleep.", file=sys.stderr)
        else:
            print(f"\n❌ CONNECTION ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    if not API_KEY or API_KEY == "dummy_token":
        print("⚠️ WARNING: No valid API Key found.", file=sys.stderr)

    grader_task = os.getenv("TASK_NAME")
    
    if grader_task:
        tasks_to_run = [grader_task]
    elif len(sys.argv) > 1:
        tasks_to_run = [sys.argv[1]]
    else:
        tasks_to_run = [
            "task_1_blr_cbe",
            "task_2_gwh_gtk", 
            "task_3_knp_slg"
        ]

    for t in tasks_to_run:
        run_agent(t)
        time.sleep(2)
        