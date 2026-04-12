import os
import sys
import json
import time
from typing import List, Optional, Tuple
from openai import OpenAI
from client import AmpereEnv
from models import EVAction

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("XAI_API_KEY") or "dummy_token"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
BENCHMARK    = os.getenv("AMPERE_BENCHMARK", "ampere")
SERVER_URL   = (os.environ.get("ENV_URL") or os.environ.get("AMPERE_SERVER_URL")
                or "https://team01paracetamol-ampere.hf.space")

llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are EcoRoute, an AI EV Dispatcher routing a Tata Curvv EV across Indian highways.
Goal: reach the destination BEFORE the deadline.

PHYSICS (memorise):
  eco(50kmh)=0.302%/km  cruise(70kmh)=0.593%/km  highway(90kmh)=0.982%/km  sport(110kmh)=1.464%/km
  Terrain multipliers: flat=1.0x  urban=1.2x  mountain=1.8x
  Fast-DC(60kW)=+2.22%/min  Slow-AC(7.2kW)=+0.35%/min
  Fatigue: +1pt/min driving, -3pt/min charging or resting. Crash at 300pt.

RULES:
1. WAYPOINT — pick exactly one from "Valid next_waypoint values".
2. SPEED — eco on mountain or battery<30%. cruise default. highway only if next charger <80km.
3. CHARGING — ONLY set charge_minutes>0 if the destination node HAS a charger (has_fast_charger or has_slow_charger). Charging at a no-charger node wastes time with zero gain.
4. REST — only if fatigue>120 AND not charging (charging already recovers fatigue at 3pt/min).
5. DESERT RULE — if next charger >200km away, charge to 90%+ before entering.

Output ONLY valid JSON:
{"next_waypoint": "ExactNodeName", "speed_mode": "cruise", "charge_minutes": 0, "rest_minutes": 0}
"""

MAX_RETRIES = 3

# ── Grader Logging (stdout) ─────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ── Physics Helpers ─────────────────────────────────────────────────────────
_COST_PCT    = {"eco": 0.302, "cruise": 0.593, "highway": 0.982, "sport": 1.464}
_TERRAIN_MULT = {"flat": 1.0, "urban": 1.2, "mountain": 1.8}

def battery_needed(dist, speed, terrain):
    return dist * _COST_PCT.get(speed, 0.593) * _TERRAIN_MULT.get(terrain, 1.0)

# ── LLM Call ────────────────────────────────────────────────────────────────
def get_action_from_llm(obs, previous_intervention: str = "") -> EVAction | None:
    valid_waypoints = [r.destination_node for r in obs.available_routes]

    # Compact dashboard — all info the LLM needs, no JSON blob
    lookahead_str = "  (none — possible desert ahead!)"
    if obs.charger_lookahead:
        lookahead_str = "\n" + "\n".join(
            f"  {c.node} +{c.distance_km}km | {c.charger_kw}kW | rel={c.reliability:.0%} | after={c.terrain_after}"
            for c in obs.charger_lookahead[:5]
        )

    routes_str = "\n".join(
        f"  {r.destination_node}: {r.distance_km}km terrain={r.terrain} "
        f"fast={r.has_fast_charger} slow={r.has_slow_charger} rest={r.has_rest_facility}"
        for r in obs.available_routes
    )

    user_prompt = (
        f"LOCATION : {obs.current_location}\n"
        f"BATTERY  : {obs.battery_percentage:.1f}% [{obs.battery_warning}]  range≈{obs.estimated_range_km}km\n"
        f"FATIGUE  : {obs.fatigue_points:.0f}/300\n"
        f"TIME     : {obs.time_elapsed_minutes:.0f}min elapsed\n"
        f"TO DEST  : {obs.navigation_system.distance_to_final_destination_km}km\n"
        f"NEXT CHG : {obs.navigation_system.nearest_charger_node} ({obs.navigation_system.distance_to_nearest_charger_km}km, rel={obs.navigation_system.charger_reliability_estimate:.0%})\n"
        f"CAN REACH: {obs.can_reach_next_charger}\n"
        f"\nCHARGERS AHEAD:{lookahead_str}\n"
        f"\nROUTES:\n{routes_str}\n"
        f"\nValid next_waypoint values: {valid_waypoints}\n"
    )
    if previous_intervention:
        user_prompt += f"\n⚠️ LAST ACTION OVERRIDDEN: {previous_intervention}\n"
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
            llm_json["charge_minutes"] = min(max(int(llm_json.get("charge_minutes", 0)), 0), 480)
            llm_json["rest_minutes"]   = min(max(int(llm_json.get("rest_minutes",   0)), 0), 480)
            return EVAction(**llm_json)
        except Exception as e:
            print(f"   ⚠️  LLM attempt {attempt}: {e}", file=sys.stderr)
    return None

# ── Autopilot ───────────────────────────────────────────────────────────────
def apply_autopilot(action: EVAction, obs) -> Tuple[EVAction, str]:
    msg = ""
    valid_waypoints = [r.destination_node for r in obs.available_routes]

    # Fix hallucinated waypoint
    if action.next_waypoint not in valid_waypoints:
        action.next_waypoint = valid_waypoints[0]
        msg += "fixed-waypoint "

    chosen_route  = next((r for r in obs.available_routes if r.destination_node == action.next_waypoint), None)
    current_route = next((r for r in obs.available_routes if r.destination_node == obs.current_location), None)
    is_staying    = action.next_waypoint == obs.current_location

    dest_has_charger = chosen_route and (chosen_route.has_fast_charger or chosen_route.has_slow_charger)
    here_has_charger = current_route and (current_route.has_fast_charger or current_route.has_slow_charger)

    eco_range  = obs.battery_percentage / 0.302
    dist_to_end = obs.navigation_system.distance_to_final_destination_km

    # ── Speed ────────────────────────────────────────────────────────────────
    if chosen_route and chosen_route.terrain == "mountain":
        action.speed_mode = "eco"
    elif obs.battery_percentage < 30.0:
        action.speed_mode = "eco"
    elif dist_to_end > eco_range:                        # range anxiety
        action.speed_mode = "eco"
        msg += "eco-range-anxiety "
    elif action.speed_mode in ("highway", "sport") and obs.battery_percentage < 60.0:
        action.speed_mode = "cruise"
        msg += "downgraded-speed "

    # ── Charging ─────────────────────────────────────────────────────────────
    if not dest_has_charger:
        if action.charge_minutes > 0:
            print(f"   [AP] No charger at {action.next_waypoint} — cleared {action.charge_minutes}m charge.", file=sys.stderr)
            msg += "blocked-ghost-charge "
        action.charge_minutes = 0

        # If staying at a dead node with no charge/rest, force a move
        if is_staying and action.rest_minutes == 0:
            for r in obs.available_routes:
                if r.destination_node != obs.current_location:
                    action.next_waypoint = r.destination_node
                    msg += "forced-move-dead-node "
                    chosen_route = r
                    is_staying   = False
                    break
    else:
        # Destination has a charger — calculate exact charge needed
        dist_to_dest  = chosen_route.distance_km if chosen_route else 0
        batt_on_arrive = obs.battery_percentage - battery_needed(
            dist_to_dest, action.speed_mode,
            chosen_route.terrain if chosen_route else "flat"
        )

        # Find next charger after destination
        next_chg = next(
            (c for c in obs.charger_lookahead if c.distance_km > dist_to_dest + 1),
            None
        )

        if next_chg:
            gap_km         = next_chg.distance_km - dist_to_dest
            need_for_gap   = battery_needed(gap_km, "eco", next_chg.terrain_after)
            target_battery = min(need_for_gap + 15.0, 95.0)
        else:
            # Last charger before destination or desert
            remaining = dist_to_end - dist_to_dest
            if remaining <= dist_to_dest + 5:
                target_battery = 20.0               # final leg, low target
            else:
                target_battery = 92.0               # desert ahead
                msg += "desert-charge "

        if batt_on_arrive < target_battery:
            deficit  = target_battery - batt_on_arrive
            rate     = 2.22 if (chosen_route and chosen_route.has_fast_charger) else 0.35
            mins     = min(int(deficit / rate) + 1, 60)   # hard cap 60min/stop
            if mins != action.charge_minutes:
                print(f"   [AP] Charge {mins}m @ {action.next_waypoint} (need {deficit:.1f}%, {rate}%/m).", file=sys.stderr)
                msg += f"charge-{mins}m "
            action.charge_minutes = mins
        else:
            if action.charge_minutes > 0:
                print(f"   [AP] Battery sufficient — skipped charge.", file=sys.stderr)
            action.charge_minutes = 0

    # ── Universal safety: can we leave this charger? ─────────────────────────
    if here_has_charger and not is_staying:
        if dist_to_end > eco_range - 50:
            print(f"   [AP] Survival stay: dist={dist_to_end}km > range={eco_range:.0f}km.", file=sys.stderr)
            action.next_waypoint  = obs.current_location
            is_staying            = True
            target_b = min(dist_to_end * 0.35 + 20.0, 95.0)
            if obs.battery_percentage < target_b:
                rate  = 2.22 if here_has_charger and current_route.has_fast_charger else 0.35
                mins  = min(int((target_b - obs.battery_percentage) / rate) + 1, 90)
                action.charge_minutes = mins
                msg += f"survival-charge-{mins}m "

    # ── Idle guard: staying + nothing happening → force move ────────────────
    if is_staying and action.charge_minutes == 0 and action.rest_minutes == 0:
        for r in obs.available_routes:
            if r.destination_node != obs.current_location:
                action.next_waypoint = r.destination_node
                msg += "forced-move-idle "
                break

    # ── Full battery → stop charging and move ────────────────────────────────
    if is_staying and obs.battery_percentage >= 95.0:
        action.charge_minutes = 0
        for r in obs.available_routes:
            if r.destination_node != obs.current_location:
                action.next_waypoint = r.destination_node
                msg += "forced-move-full "
                break

    # ── Rest logic ───────────────────────────────────────────────────────────
    if action.charge_minutes > 0:
        action.rest_minutes = 0               # charging already recovers fatigue
    elif obs.fatigue_points > 200:
        action.rest_minutes = max(action.rest_minutes, 25)
    elif obs.fatigue_points > 150:
        action.rest_minutes = max(action.rest_minutes, 15)
    else:
        action.rest_minutes = 0

    return action, msg.strip()

# ── Score Extraction ────────────────────────────────────────────────────────
def extract_numeric_score(obs, total_reward) -> float:
    if obs.metadata and "final_grader_score" in obs.metadata:
        return float(obs.metadata["final_grader_score"])
    return 0.99 if total_reward > 0 else 0.01

# ── Main Agent Loop ─────────────────────────────────────────────────────────
def run_agent(scenario: str):
    print(f"\n{'='*55}", file=sys.stderr)
    print(f"🚀 {scenario}  →  {SERVER_URL}", file=sys.stderr)
    print(f"{'='*55}", file=sys.stderr)

    try:
        with AmpereEnv(base_url=SERVER_URL).sync() as env:
            step_result = env.reset(scenario_key=scenario)
            obs  = step_result.observation
            done = step_result.done

            log_start(task=scenario, env=BENCHMARK, model=MODEL_NAME)

            rewards: List[float] = []
            step_count   = 0
            total_reward = 0.0
            prev_intervention = ""

            while not done:
                step_count += 1
                error = None

                # ── Compact step header ──────────────────────────────────────
                nav = obs.navigation_system
                lookahead_preview = ""
                if obs.charger_lookahead:
                    c = obs.charger_lookahead[0]
                    lookahead_preview = f"  next-chg={c.node}(+{c.distance_km}km/{c.charger_kw}kW)"

                print(
                    f"\n[{step_count:02d}] {obs.current_location}"
                    f"  🔋{obs.battery_percentage:.0f}%[{obs.battery_warning}]"
                    f"  😴{obs.fatigue_points:.0f}"
                    f"  ⏱{obs.time_elapsed_minutes:.0f}m"
                    f"  🏁{nav.distance_to_final_destination_km}km"
                    f"  📡{nav.nearest_charger_node}({nav.distance_to_nearest_charger_km}km)"
                    f"{lookahead_preview}",
                    file=sys.stderr
                )
                print(
                    f"     range≈{obs.estimated_range_km}km"
                    f"  opts={[r.destination_node for r in obs.available_routes]}",
                    file=sys.stderr
                )

                action = get_action_from_llm(obs, prev_intervention)
                if action is None:
                    error = "LLM failed"
                    print("❌ LLM failed. Aborting.", file=sys.stderr)
                    log_step(step_count, "null", 0.0, True, error)
                    break

                action, prev_intervention = apply_autopilot(action, obs)

                ap_note = f"  [{prev_intervention}]" if prev_intervention else ""
                print(
                    f"     ➤ {action.next_waypoint} @ {action.speed_mode}"
                    f"  chg={action.charge_minutes}m  rst={action.rest_minutes}m"
                    f"{ap_note}",
                    file=sys.stderr
                )

                action_str = json.dumps(action.model_dump(), separators=(',', ':'))

                try:
                    step_result  = env.step(action)
                    obs          = step_result.observation
                    done         = step_result.done
                    reward       = float(step_result.reward or 0.0)
                except Exception as e:
                    error  = str(e).replace('\n', ' ')
                    reward = 0.0
                    done   = True

                total_reward += reward
                rewards.append(reward)
                print(f"     💰 {reward:+.2f}  (total {total_reward:.2f})", file=sys.stderr)

                log_step(step_count, action_str, reward, done, error)
                time.sleep(0.5)

            # ── Episode summary ──────────────────────────────────────────────
            meta  = obs.metadata or {}
            score = extract_numeric_score(obs, total_reward)
            success = score >= 0.5

            # Use direct obs fields (always set); fall back to metadata dict
            reached   = obs.reached_destination or meta.get("reached_destination", False)
            stranded  = obs.stranded            or meta.get("stranded",  False)
            crashed   = obs.crashed             or meta.get("crashed",   False)
            timed_out = meta.get("timed_out", False)
            batt_left = meta.get("battery_remaining_pct", obs.battery_percentage)
            over_mins = meta.get("minutes_over_deadline", 0.0)

            if reached:
                status = f"✅ REACHED ({'ON TIME' if not over_mins else f'LATE +{over_mins:.0f}m'})"
            elif stranded:
                status = "🪫 STRANDED"
            elif crashed:
                status = "💥 CRASHED"
            elif timed_out:
                status = "⏰ TIMED OUT"
            else:
                status = "❓ UNKNOWN"

            print(f"\n{'─'*55}", file=sys.stderr)
            print(f"🏁 {status}  score={score:.3f}", file=sys.stderr)
            print(
                f"   time={obs.time_elapsed_minutes:.0f} mins  steps={step_count}  "
                f"battery={batt_left:.1f}%  reward={total_reward:.1f}",
                file=sys.stderr
            )
            print(f"{'─'*55}", file=sys.stderr)

            log_end(success, step_count, score, rewards)

    except Exception as e:
        if "503" in str(e):
            print(f"❌ SERVER ASLEEP (503): {SERVER_URL}", file=sys.stderr)
        else:
            print(f"❌ CONNECTION ERROR: {e}", file=sys.stderr)

# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not API_KEY or API_KEY == "dummy_token":
        print("⚠️  No API key. Set XAI_API_KEY or API_KEY.", file=sys.stderr)

    grader_task = os.getenv("TASK_NAME")
    if grader_task:
        tasks_to_run = [grader_task]
    elif len(sys.argv) > 1:
        tasks_to_run = [sys.argv[1]]
    else:
        tasks_to_run = ["task_1_blr_cbe", "task_2_gwh_gtk", "task_3_knp_slg"]

    for t in tasks_to_run:
        run_agent(t)
        time.sleep(2)