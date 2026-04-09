import os
import json
import time
from typing import List, Optional
from openai import OpenAI
from client import AmpereEnv
from models import EVAction

# ── Config ─────────────────────────────────────────────────────────────────
# Strict compliance with Hackathon requirements
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("XAI_API_KEY") or "dummy_token"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
BENCHMARK = os.getenv("AMPERE_BENCHMARK", "ampere")

SERVER_URL = (os.environ.get("ENV_URL") or os.environ.get("AMPERE_SERVER_URL")
              or "https://navistha-ampere.hf.space")

llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are EcoRoute, an advanced AI EV Dispatcher.
Your objective is to navigate a Tata Nexon EV to the final destination BEFORE the deadline.

CRITICAL RULES:
1. SPEED: 'cruise' (70km/h) is your default. Use 'eco' (50km/h) ONLY on mountain terrain.
2. CHARGING: Charge at fast_dc nodes when battery is below 40%. Keep stops short (15-25 min).
3. WAYPOINTS: Choose EXACTLY one value from the 'Valid next_waypoint values' list.
4. FATIGUE: Rest only if fatigue > 150. Keep rest short (10-15 min).

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


# ── LLM Action ──────────────────────────────────────────────────────────────
def get_action_from_llm(obs) -> EVAction | None:
    valid_waypoints = [r.destination_node for r in obs.available_routes]
    user_prompt = (
        f"CURRENT DASHBOARD:\n{obs.model_dump_json(indent=2)}\n\n"
        f"Valid next_waypoint values: {valid_waypoints}\n"
        f"Output JSON."
    )
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
            action = EVAction(**llm_json)
            if action.next_waypoint not in valid_waypoints:
                print(f"[DEBUG] Attempt {attempt}: Invalid waypoint. Retrying...", flush=True)
                continue
            return action
        except Exception as e:
            print(f"[DEBUG] Attempt {attempt}: {e}. Retrying...", flush=True)
    return None

# ── Autopilot Override ──────────────────────────────────────────────────────
def apply_autopilot(action: EVAction, obs) -> EVAction:
    chosen_route = next(
        (r for r in obs.available_routes if r.destination_node == action.next_waypoint),
        None
    )
    if chosen_route and chosen_route.terrain == "mountain":
        action.speed_mode = "eco"
    else:
        action.speed_mode = "cruise"

    action.charge_minutes = 0
    if chosen_route and chosen_route.has_fast_charger:
        dist_remaining = obs.navigation_system.distance_to_final_destination_km
        if dist_remaining > 60:
            if obs.battery_percentage < 40:
                action.charge_minutes = 25
            elif 40 <= obs.battery_percentage < 60:
                action.charge_minutes = 10
        else:
            if obs.battery_percentage < 20:
                action.charge_minutes = 15

    action.rest_minutes = 0
    if obs.fatigue_points > 200:
        action.rest_minutes = 20
    elif obs.fatigue_points > 150 and chosen_route and getattr(chosen_route, 'has_rest_facility', False):
        action.rest_minutes = 10

    return action

# ── Score Extraction ────────────────────────────────────────────────────────
def extract_numeric_score(obs, total_reward) -> float:
    if obs.metadata and "final_grader_score" in obs.metadata:
        return float(obs.metadata.get("final_grader_score", 0.0))
    heading = getattr(obs.navigation_system, "optimal_heading", "")
    if heading and "SCORE" in heading:
        try:
            return float(heading.split("SCORE:")[1].split("/")[0].strip())
        except:
            pass
    return max(0.0, min(1.0, float(total_reward))) # fallback clamp to [0,1]

# ── Main Agent Loop ─────────────────────────────────────────────────────────
def run_agent(scenario: str):
    print(f"[DEBUG] Booting EcoRoute Agent for Scenario: {scenario}", flush=True)

    with AmpereEnv(base_url=SERVER_URL).sync() as env:
        step_result = env.reset(scenario_key=scenario)
        obs  = step_result.observation
        done = step_result.done

        # 1. REQUIRED GRADER OUTPUT: Start tag
        log_start(task=scenario, env=BENCHMARK, model=MODEL_NAME)

        rewards: List[float] = []
        step_count = 0
        total_reward = 0.0
        success = False

        while not done:
            step_count += 1
            error = None

            action = get_action_from_llm(obs)
            if action is None:
                error = "LLM failed to return valid action"
                # If LLM completely fails, log the failure step before breaking so the grader doesn't desync
                log_step(step=step_count, action="null", reward=0.0, done=True, error=error)
                break

            action = apply_autopilot(action, obs)
            
            # Format action as a single-line string to prevent scraper crashes
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
            
            # 2. REQUIRED GRADER OUTPUT: Step tag
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)
            time.sleep(0.5)

        # 3. REQUIRED GRADER OUTPUT: End tag
        score = extract_numeric_score(obs, total_reward)
        success = score >= 0.5  # Assuming a score >= 0.5 means the episode was technically a success

        log_end(success=success, steps=step_count, score=score, rewards=rewards)


if __name__ == "__main__":
    import sys
    
    if not API_KEY or API_KEY == "dummy_token":
        print("[DEBUG] WARNING: No valid API Key found. Run this locally using: export API_KEY='your-key'", flush=True)

    # If the Grader passes a specific TASK_NAME environment variable, run only that.
    grader_task = os.getenv("TASK_NAME")
    
    if grader_task:
        tasks_to_run = [grader_task]
    elif len(sys.argv) > 1:
        tasks_to_run = [sys.argv[1]]
    else:
        # Otherwise, run all 3 of your tasks sequentially! 
        # IMPORTANT: Make sure these string keys EXACTLY match the keys in your graph_data.json
        tasks_to_run = [
            "task_1_blr_cbe",
            "task_2_gwh_gtk", 
            "task_3_knp_slg"
        ]

    for t in tasks_to_run:
        run_agent(t)
        time.sleep(2) # Give the connection a brief rest between runs