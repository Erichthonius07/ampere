import os
import json
import time
from openai import OpenAI
from client import AmpereEnv
from models import EVAction

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY    = (os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
              or os.environ.get("XAI_API_KEY") or "dummy_token")
BASE_URL   = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
SERVER_URL = (os.environ.get("ENV_URL") or os.environ.get("AMPERE_SERVER_URL")
              or "https://navistha-ampere.hf.space")

llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            llm_json = json.loads(response.choices[0].message.content)
            action = EVAction(**llm_json)
            if action.next_waypoint not in valid_waypoints:
                print(f"   ⚠️  Attempt {attempt}: Invalid waypoint. Retrying...")
                continue
            return action
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt}: {e}. Retrying...")
    return None


# ── Autopilot Override ──────────────────────────────────────────────────────
def apply_autopilot(action: EVAction, obs) -> EVAction:
    """
    Deterministic safety layer applied AFTER the LLM chooses a waypoint.
    Only overrides speed and charge decisions, never the waypoint.
    """
    # -- 1. Terrain-aware speed --
    # Find the route option for the chosen waypoint to check its terrain
    chosen_route = next(
        (r for r in obs.available_routes if r.destination_node == action.next_waypoint),
        None
    )
    if chosen_route and chosen_route.terrain == "mountain":
        action.speed_mode = "eco"
    else:
        action.speed_mode = "cruise"

    # -- 2. Charging logic --
    action.charge_minutes = 0  # default: no charge

    # Is there a fast charger at the NEXT node (where we're going)?
    if chosen_route and chosen_route.has_fast_charger:
        dist_remaining = obs.navigation_system.distance_to_final_destination_km

        if dist_remaining > 60:  # Not in the final approach
            # Charge if battery is low
            if obs.battery_percentage < 40:
                action.charge_minutes = 25   # ~60% charge at 50kW
            elif obs.battery_percentage < 60 and obs.battery_percentage > 40:
                action.charge_minutes = 10   # Quick top-up
        else:
            # Final approach — only charge if critically low
            if obs.battery_percentage < 20:
                action.charge_minutes = 15

    # -- 3. Fatigue management --
    action.rest_minutes = 0
    if obs.fatigue_points > 200:
        action.rest_minutes = 20
    elif obs.fatigue_points > 150 and chosen_route and chosen_route.has_rest_facility:
        action.rest_minutes = 10

    return action


# ── Score Extraction ────────────────────────────────────────────────────────
def extract_final_score(obs) -> str:
    """
    Robust extraction: tries metadata dict first, then parses optimal_heading
    string as fallback (for openenv's built-in grader format).
    """
    # Primary: our custom metadata
    if obs.metadata:
        score = obs.metadata.get("final_grader_score")
        if score is not None:
            reached  = obs.metadata.get("reached_destination", "?")
            stranded = obs.metadata.get("stranded", "?")
            crashed  = obs.metadata.get("crashed", "?")
            mins_over = obs.metadata.get("minutes_over_deadline", 0)
            return (f"{score:.2f}  |  Reached: {reached}  |  "
                    f"Stranded: {stranded}  |  Crashed: {crashed}  |  "
                    f"Mins over deadline: {mins_over}")

    # Fallback: openenv embeds score in optimal_heading as
    # "🏆 FINAL SCORE: X.X / 1.0  |  Reached: ..."
    heading = getattr(obs.navigation_system, "optimal_heading", "")
    if heading and "SCORE" in heading:
        return heading

    return "N/A (score not available)"


# ── Main Agent Loop ─────────────────────────────────────────────────────────
def run_agent(scenario: str = "task_1_blr_cbe"):
    print(f"\n🚀 EcoRoute Agent — Scenario: {scenario}")
    print(f"   Server: {SERVER_URL}\n")

    with AmpereEnv(base_url=SERVER_URL).sync() as env:
        step_result = env.reset(scenario_key=scenario)
        obs  = step_result.observation
        done = step_result.done

        step_count   = 0
        total_reward = 0.0

        while not done:
            step_count += 1
            print("=" * 60)
            print(f"📍 STEP {step_count} | {obs.current_location}")
            print(f"🔋 {obs.battery_percentage:.1f}%  "
                  f"| ⚠️  {obs.battery_warning}  "
                  f"| 🥱 Fatigue: {obs.fatigue_points:.0f}/300")
            print(f"⏱️  Elapsed: {obs.time_elapsed_minutes:.0f} min  "
                  f"| 🗺️  Remaining: "
                  f"{obs.navigation_system.distance_to_final_destination_km} km")
            print(f"🛣️  Options: {[r.destination_node for r in obs.available_routes]}")
            print("-" * 60)

            action = get_action_from_llm(obs)
            if action is None:
                print("❌ LLM failed after retries. Aborting.")
                break

            # Apply deterministic autopilot on top of LLM waypoint choice
            action = apply_autopilot(action, obs)

            print(f"⚡ → {action.next_waypoint}  "
                  f"| speed={action.speed_mode}  "
                  f"| charge={action.charge_minutes}m  "
                  f"| rest={action.rest_minutes}m")

            step_result  = env.step(action)
            obs          = step_result.observation
            done         = step_result.done
            total_reward += step_result.reward
            print(f"💰 Reward: {step_result.reward:+.2f}  (Total: {total_reward:.2f})")
            time.sleep(0.5)

        # ── Episode Summary ────────────────────────────────────────────────
        print("\n🏁 === EPISODE COMPLETE ===")
        print(f"   Time elapsed : {obs.time_elapsed_minutes:.1f} min")
        print(f"   Battery left : {obs.battery_percentage:.1f}%")
        print(f"   Total reward : {total_reward:.2f}")
        print(f"   Steps taken  : {step_count}")
        print(f"🏆 GRADER SCORE: {extract_final_score(obs)}")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "task_1_blr_cbe"
    run_agent(scenario)