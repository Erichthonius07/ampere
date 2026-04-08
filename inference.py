import os
import json
import time
from openai import OpenAI
from client import AmpereEnv
from models import EVAction

# Fallback chain: Grader Key -> HF Token -> Your Local Key -> Dummy (prevents crash)
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("XAI_API_KEY") or "dummy_token"
# Fallback chain: Grader URL -> Your Local Groq URL
BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")

# Initialize the LLM Client safely
llm_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

SERVER_URL = os.environ.get("AMPERE_SERVER_URL", "https://navistha-ampere.hf.space/web")

# Read server URL from environment variable — set AMPERE_SERVER_URL for cloud deployment
SYSTEM_PROMPT = """You are EcoRoute, an advanced AI EV Dispatcher. 
Your objective is to safely navigate a Tata Nexon EV to the final destination before the deadline.

CRITICAL RULES:
1. PHYSICS & TIME MANAGEMENT: Aerodynamic drag is exponential. 
   - 'eco' (50km/h): Safest (~220km range). Use this 80% of the time.
   - 'cruise' (70km/h): Use this to save time if the next node is < 80km away AND battery > 60%.
   - 'highway' (90km/h): MASSIVE battery drain (~67km max range). ONLY use this "sprint" if the next node is < 45km away AND battery > 50%.
   - 'sport' (110km/h): THE LAST MILE DASH. Drains battery 4.8x faster (~45km max range). ONLY use this if the final destination is < 25km away, you have plenty of battery, and you are about to run out of time!
2. TERRAIN: 'mountain' terrain drains battery 1.8x faster. MUST use 'eco' speed on mountains.
3. FATIGUE: If fatigue hits 300, you crash. Use 'rest_minutes' to recover (-3 points/min). Charging also counts as resting.
4. CHARGING (SURVIVAL): If battery is below 40%, you MUST set 'charge_minutes' to at least 45 at the current node to survive. 'fast_dc' gives 2.45%/min, 'slow_ac' gives 0.353%/min. 
5. WAYPOINTS: Choose a 'next_waypoint' that exactly matches a 'destination_node' in 'available_routes'.

You must output your decision strictly as a JSON object matching this schema:
{
    "next_waypoint": "ExactNodeName",
    "speed_mode": "eco" | "cruise" | "highway" | "sport",
    "charge_minutes": 0,
    "rest_minutes": 0
}
"""

MAX_RETRIES = 3


def get_action_from_llm(obs) -> EVAction | None:
    """
    Calls the LLM and returns a validated EVAction.
    Retries up to MAX_RETRIES times on parse/validation errors.
    Also validates that the chosen waypoint is actually available.
    Returns None if all retries are exhausted.
    """
    valid_waypoints = [r.destination_node for r in obs.available_routes]
    user_prompt = (
        f"CURRENT DASHBOARD:\n{obs.model_dump_json(indent=2)}\n\n"
        f"Valid next_waypoint values (choose EXACTLY one): {valid_waypoints}\n\n"
        f"What is your next action? Output ONLY valid JSON."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            llm_json = json.loads(response.choices[0].message.content)
            action = EVAction(**llm_json)

            # Validate waypoint is actually reachable from current node
            if action.next_waypoint not in valid_waypoints:
                print(
                    f"   ⚠️  Attempt {attempt}: LLM chose invalid waypoint "
                    f"'{action.next_waypoint}'. Valid: {valid_waypoints}. Retrying..."
                )
                continue

            return action

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"   ⚠️  Attempt {attempt}: Failed to parse LLM response — {e}. Retrying...")

    print("❌ All retries exhausted. Could not get a valid action from the LLM.")
    return None


def run_agent(scenario="task_1_blr_cbe"):
    print(f"\n🚀 Booting EcoRoute Agent for Scenario: {scenario}")
    print(f"Connecting to OpenEnv Server at {SERVER_URL}...\n")

    with AmpereEnv(base_url=SERVER_URL).sync() as env:

        # 1. Start the Episode
        step_result = env.reset(scenario_key=scenario)
        obs = step_result.observation
        done = step_result.done

        step_count = 0
        total_reward = 0.0

        # 2. The Main RL Loop
        while not done:
            step_count += 1
            print("=" * 60)
            print(f"📍 STEP {step_count} | Current Location: {obs.current_location}")
            print(f"🔋 Battery: {obs.battery_percentage}%  | ⚠️ Warning: {obs.battery_warning}")
            print(f"🥱 Fatigue: {obs.fatigue_points}/300 | ⏱️ Elapsed: {obs.time_elapsed_minutes} mins")
            print(f"🛣️  Options: {[r.destination_node for r in obs.available_routes]}")
            print("-" * 60)

            print("🧠 Thinking...")
            action = get_action_from_llm(obs)

            if action is None:
                print("❌ Agent could not decide. Aborting episode.")
                break

            print(f"⚡ ACTION TAKEN:")
            print(f"   ► Drive to: {action.next_waypoint}")
            print(f"   ► Speed:    {action.speed_mode}")
            print(f"   ► Charge:   {action.charge_minutes} mins")
            print(f"   ► Rest:     {action.rest_minutes} mins")

            # 3. Send the action to the Environment
            step_result = env.step(action)
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
            total_reward += reward

            print(f"💰 Reward this step: {reward:.2f} (Total: {total_reward:.2f})\n")
            time.sleep(1)

        # 4. Episode Finished — Print the Grader Report
        # 4. Episode Finished — Print the Grader Report
        print("🏁 === EPISODE COMPLETE === 🏁")
        print(f"Time Elapsed: {obs.time_elapsed_minutes} mins")
        print("-" * 60)
        print(obs.navigation_system.optimal_heading)
        print("=" * 60)


if __name__ == "__main__":
    if not API_KEY or API_KEY == "dummy_token":
        print("⚠️ WARNING: No valid API Key found. The LLM calls will likely fail.")
        print("Run this locally using: export XAI_API_KEY='your-key'")

    # Swap to task_2 or task_3 once task_1 is passing
    run_agent("task_1_blr_cbe")