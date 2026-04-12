# ⚡ Ampere: EV Dispatcher AI for Indian Highways

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq](https://img.shields.io/badge/Groq-Fast_LLM-f55036)](https://groq.com/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-brightgreen)](https://github.com/)

**Ampere** is an OpenEnv simulation environment where an LLM agent acts as a trip dispatcher, routing a real Tata Curvv EV Creative 45 across Indian highway corridors under real-world physics, infrastructure, and human constraints.

---

## 🛑 The Problem: India's Charging Infrastructure Gap

India manages over 1,46,000 km of national highways (NHAI), but inter-city EV travel remains a logistical challenge. According to McKinsey's 2024 Mobility Consumer Pulse survey, nearly 30% of EV owners globally are considering switching back to petrol vehicles — with inadequate or unreliable public charging infrastructure cited as the leading cause.

Standard routing engines like Google Maps are fundamentally unsuited for EVs. They calculate shortest distance but ignore the three real constraints of highway EV travel:

- **Aerodynamic physics** — battery drain scales with the square of speed. Driving at 90 km/h drains 3.24× more battery per km than 50 km/h.
- **Charger deserts** — stretches of 150–300 km with no working charger exist on real Indian highways like NH19.
- **Driver fatigue** — MORTH mandates rest after sustained driving. A stranded driver and a crashed driver are equally unacceptable outcomes.

The agent cannot simply "charge when battery is low." It must plan multiple legs ahead, weigh the risk of an unreliable charger against the cost of a detour, and synchronise mandatory rest stops with charging sessions to avoid wasting time. This is a genuine **Multi-Objective Optimisation problem under Partial Observability (POMDP)**.

---

## 💡 The Solution

We built **Ampere** using a hybrid neuro-symbolic architecture — an LLM pathfinder paired with a deterministic physics-engine autopilot:

- **System 1 (The LLM)**: Analyses the route graph, evaluates remaining distance and charger availability, and selects the optimal next waypoint based on battery, fatigue, and deadline context.
- **System 2 (The Autopilot)**: A deterministic safety layer that intercepts LLM decisions. It calculates exact charge times needed to survive the next leg, automatically drops to `eco` speed on mountain terrain or when battery falls below 35%, and overlaps rest with charging to minimise wasted time.

---

## ⚙️ The Physics Engine

Built on real Tata Curvv EV Creative 45 specifications — not approximations:

| Parameter | Real Value | Simulation Value |
|---|---|---|
| Battery capacity | 45 kWh usable | 45.0 kWh |
| Base consumption | 136 Wh/km at 50 km/h | 136.0 Wh/km |
| DC fast charge (60 kW) | 10–80% in 40 min | 2.22%/min |
| AC slow charge (7.2 kW) | 10–100% in 7.25 hrs | 0.353%/min |
| Real-world highway range | 330–350 km (mixed) | ~330 km at eco speed |
| Mountain terrain penalty | Real ghat/hill roads | 1.8× drain multiplier |
| Urban terrain penalty | City stop-start traffic | 1.2× drain multiplier |

**Speed modes and their real range consequences:**

| Mode | Speed | Range (full charge) | When to use |
|---|---|---|---|
| `eco` | 50 km/h | ~330 km | Charger deserts, mountain climbs |
| `cruise` | 70 km/h | ~168 km | Normal highway legs |
| `highway` | 90 km/h | ~102 km | Short legs with charger confirmed ahead |
| `sport` | 110 km/h | ~68 km | Last-mile dash only |

**Terrain multipliers:**

| Terrain | Multiplier | Where applied |
|---|---|---|
| `flat` | 1.0× | Open NH highway stretches |
| `urban` | 1.2× | City exits/approaches — Bangalore, Guwahati, Kanpur-Lucknow corridor, Gorakhpur, Siliguri |
| `mountain` | 1.8× | Thoppur Ghats (Task 1), NH10 Teesta Valley (Task 2) |

**Fatigue model** — MORTH-compliant 300-point scale:
- +1 point per minute driving
- −3 points per minute resting or charging
- Terminal at 300 (crash → episode score 0.01)

**Stochastic infrastructure** — Task 3 applies per-station failure probabilities derived from real Google Maps ratings on the Kanpur–Siliguri corridor. Charger status is unknown until arrival (POMDP). The agent must maintain battery buffers to survive failures discovered on arrival.

---

## 🗺️ The Three Tasks

| Task | Route | Distance | Key Challenge |
|---|---|---|---|
| **Easy** | Bangalore → Coimbatore | 365 km | Terrain-aware speed control, Thoppur ghats |
| **Medium** | Guwahati → Gangtok | 540 km | 80% start battery, 170 km charger desert, NH10 mountain section |
| **Hard** | Kanpur → Siliguri | 1110 km | NH19 charger desert (300 km gap), stochastic failures, fatigue across 28 hours |

All route data — node distances, charger locations, reliability ratings — collected from real Google Maps data on each corridor.

---

## 📊 Evaluation Results

The grader scores strictly between 0.01 and 0.99:

| Score | Meaning |
|---|---|
| **0.99** | Reached destination safely before deadline |
| **0.31–0.59** | Reached destination, late (interpolated penalty) |
| **0.01** | Stranded (0% battery), crashed (fatigue 300), or timed out |

Task 1 scores consistently near 0.99. Task 3 exhibits variance due to stochastic charger failures — this is intended. A naive agent driving at constant cruise speed strands on Task 2 and fails Task 3 on nearly every run.

---

## ⚙️ Local Installation & Usage

### Prerequisites
- Python 3.11+
- `uv` (recommended) or `pip`
- Groq API Key or Hugging Face Token

### Setup

```bash
# Clone the repo
git clone https://github.com/Erichthonius07/ampere.git
cd ampere

# Install dependencies
pip install -r requirements.txt
# OR using uv:
uv sync

# Set your API key (Windows PowerShell)
$env:XAI_API_KEY="your_groq_or_xai_key"

# Set your API key (Mac/Linux)
export XAI_API_KEY="your_groq_or_xai_key"

# Run the agent against all 3 tasks
python inference.py
```

The inference script boots the environment server, connects via WebSocket, and runs the agent sequentially through all three evaluation tasks.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent core | OpenAI Python SDK, Llama 3.3 70B via Groq |
| Environment server | FastAPI, uvicorn, websockets |
| Physics & graph | NetworkX (Dijkstra pathfinding), NumPy |
| Schema validation | Pydantic v2, openenv-core |
| Cloud deployment | Hugging Face Spaces |

---

## 📁 Project Structure

```
ampere/
├── ampere/
│   ├── models.py              # EVAction, EVObservation, GPSDashboard schemas
│   ├── client.py              # WebSocket client
│   ├── openenv.yaml           # Task definitions and metadata
│   └── server/
│       ├── app.py             # FastAPI server via create_app()
│       └── ampere_environment.py  # Physics engine, reward function, grader
├── graph_data.json            # Real route data for all 3 tasks
├── inference.py               # LLM agent with autopilot safety layer
└── requirements.txt
```

---

*Built by Team Paracetamol*
