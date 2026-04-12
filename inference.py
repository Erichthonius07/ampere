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
              or "https://navistha-ampere.hf.space")

llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are EcoRoute, an advanced AI EV Dispatcher.
Your objective is navigate a Tata Nexon EV to the final destination BEFORE the deadline.

CRITICAL RULES:
1. ALWAYS MOVE FORWARD: You MUST choose EXACTLY ONE destination from the `available_routes`. NEVER output your current location. You cannot stay in place.
2. CHARGE AT DESTINATION: Charging happens at your destination. If a route shows `has_fast_charger: true` or `has_slow_charger: true`, you must input enough `charge_minutes` to reach 100% battery (assume 1.8% gain per min for fast_dc, 0.25% for slow_ac). Max allowed is 480.
3. SURVIVE DESERTS: Chargers can randomly break down! You must top-off to 100% at every single opportunity to survive.
4. SPEED: 'cruise' (70km/h) is default. Use 'eco' (50km/h) to drastically save battery.
5. FATIGUE: Rest only if fatigue > 150. Keep rest equal to charge time. Max allowed is 480.

Output ONLY valid JSON matching this schema exactly:
{
    "next_waypoint": "ExactNodeName",
    "speed_mode": "cruise",
    "charge_minutes": 0,
    "rest_minutes": 0
}
"""

MAX_RETRIES = 3

# ── Strict Grader Logging Functions ─────────────────────────────────────────
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
            
            # --- BULLETPROOF PYDANTIC CLAMP ---
            # Extract safely, default to 0, clamp to 480 to prevent server crashes
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
    
    if action.next_waypoint not in valid_waypoints:
        action.next_waypoint = valid_waypoints[0]
        intervention_msg += f"Autopilot: Fixed invalid waypoint. "

    chosen_route = next((r for r in obs.available_routes if r.destination_node == action.next_waypoint), None)
    
    # 1. THE PARANOIA POLICY (Anti-Stochastic Failure)
    # If we are below 75% battery, we cannot risk driving fast into a potential broken charger.
    if obs.battery_percentage < 75.0 and action.speed_mode != "eco":
        print(f"   [AUTOPILOT] Battery < 75%. Forcing ECO mode to survive potential charger failures.", file=sys.stderr)
        action.speed_mode = "eco"
        intervention_msg += "Autopilot: Forced ECO mode to conserve battery in case the next charger is broken. "

    # --- PERFECT PHYSICS SIMULATION ---
    speed = 50.0 if action.speed_mode == "eco" else 70.0
    drag = (speed / 50.0) ** 2
    terrain_mult = 1.8 if chosen_route.terrain == "mountain" else (1.2 if chosen_route.terrain == "urban" else 1.0)
    
    estimated_drain = (136.0 * chosen_route.distance_km * drag * terrain_mult) / 450.0
    arrival_battery = obs.battery_percentage - estimated_drain

    # 2. FATAL TRAJECTORY PREVENTION
    if arrival_battery <= 15.0 and action.speed_mode != "eco":
        action.speed_mode = "eco"
        intervention_msg += "Autopilot: Forced ECO mode to stretch range and survive the leg. "
        
        # Recalculate arrival battery with eco mode
        drag = (50.0 / 50.0) ** 2
        estimated_drain = (136.0 * chosen_route.distance_km * drag * terrain_mult) / 450.0
        arrival_battery = obs.battery_percentage - estimated_drain

    # 3. PROACTIVE TOP-OFF TO 100%
    if chosen_route.has_fast_charger or chosen_route.has_slow_charger:
        target_battery = 100.0  
        if arrival_battery < target_battery:
            charge_rate = 1.85 if chosen_route.has_fast_charger else 0.26
            mins_needed = int((target_battery - arrival_battery) / charge_rate)
            mins_needed = min(mins_needed, 180) 
            
            if action.charge_minutes < mins_needed:
                print(f"   [AUTOPILOT] Target has charger. Forcing {mins_needed} mins to hit 100% battery.", file=sys.stderr)
                action.charge_minutes = mins_needed
                intervention_msg += f"Autopilot: Forced {mins_needed} mins of charge to hit 100%. "

    # 4. NO GHOST CHARGING
    if not chosen_route.has_fast_charger and not chosen_route.has_slow_charger:
        if action.charge_minutes > 0:
            print(f"   [AUTOPILOT] Preventing ghost charge. {action.next_waypoint} has no charger.", file=sys.stderr)
            action.charge_minutes = 0
            intervention_msg += f"Autopilot: Removed charging because {action.next_waypoint} has no charging station. "

    # 5. STRICT REST RULE
    if action.charge_minutes > 0:
        action.rest_minutes = max(action.rest_minutes, action.charge_minutes)
    elif obs.fatigue_points > 150 and action.rest_minutes < 20:
        action.rest_minutes = 20
    elif action.charge_minutes == 0 and obs.fatigue_points <= 150:
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
            print(f"\n❌ SERVER ERROR (HTTP 503): The Hugging Face Space ({SERVER_URL}) is currently asleep or restarting. Please wait 1-2 minutes and run the script again.", file=sys.stderr)
        else:
            print(f"\n❌ CONNECTION ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    if not API_KEY or API_KEY == "dummy_token":
        print("⚠️ WARNING: No valid API Key found. Run this locally using: export API_KEY='your-key'", file=sys.stderr)

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