import os
import json
import time
from openai import OpenAI
from client import AmpereEnv
from models import EVAction

# Initialize the LLM Client
llm_client = OpenAI(
    api_key=os.environ.get("XAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Read server URL from environment variable — set AMPERE_SERVER_URL for cloud deployment
SERVER_URL = os.environ.get("AMPERE_SERVER_URL", "http://localhost:8000")

SYSTEM_PROMPT = """You are EcoRoute, an advanced AI EV Dispatcher. 
Your objective is to safely navigate a Tata Nexon EV to the final destination before the deadline.

CRITICAL RULES:
1. PHYSICS: Aerodynamic drag is exponential. 
   - 'eco' (50km/h): Safest, maximum range.
   - 'cruise' (70km/h): Balanced.
   - 'highway' (90km/h) & 'sport' (110km/h): Massive battery drain. Use rarely.
2. TERRAIN: If a route's terrain is 'mountain', battery drain is multiplied by 1.8x. You MUST use 'eco' speed on mountains or you will die.
3. FATIGUE: Driver fatigue increases by 1 point per minute driving. If it hits 300, you crash. Use 'rest_minutes' to recover (-3 points/min). Dhabas/hotels are great for this.
4. CHARGING: 'slow_ac' only gives 0.23% per minute. 'fast_dc' gives 1.25% per minute. Do the math before choosing 'charge_minutes'.
5. WAYPOINTS: You can ONLY choose a 'next_waypoint' that exactly matches a 'destination_node' in your 'available_routes'.

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
        print("🏁 === EPISODE COMPLETE === 🏁")
        print(f"Destination Reached: {obs.metadata.get('reached_destination')}")
        print(f"Crashed (Fatigue):   {obs.metadata.get('crashed')}")
        print(f"Stranded (Battery):  {obs.metadata.get('stranded')}")
        print(
            f"Time Elapsed:        {obs.metadata.get('time_elapsed_minutes')} "
            f"/ {obs.metadata.get('deadline_minutes')} mins"
        )
        print("-" * 30)
        print(f"🏆 FINAL GRADER SCORE: {obs.metadata.get('final_grader_score')} / 1.0")
        print("=" * 60)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable is missing!")
        print("Run this in your terminal first:")
        print("  Windows:   set OPENAI_API_KEY=sk-your-key")
        print("  Mac/Linux: export OPENAI_API_KEY=sk-your-key")
        exit(1)

    # Swap to task_2_mum_pun or task_3_knp_slg once task_1 is passing
    run_agent("task_1_blr_cbe")