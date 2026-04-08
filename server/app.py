"""
FastAPI application for the Ampere EV Routing Environment.

Endpoints:
    POST /reset  — Reset the environment for a new episode
    POST /step   — Execute an action
    GET  /state  — Get current environment state
    GET  /schema — Get action/observation schemas

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production (HF Spaces):
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

import sys
import os
import json

# Ensure the package root is on sys.path so `models` and `server` resolve
# correctly whether the app is launched from the repo root or via Dockerfile.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

from models import EVAction, EVObservation
from server.ampere_environment import AmpereEnvironment
from fastapi.responses import JSONResponse

app = create_app(
    AmpereEnvironment,
    EVAction,
    EVObservation,
    env_name="ampere",
    max_concurrent_envs=1,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "..", "graph_data.json")

with open(json_path, "r") as f:
    graph_data = json.load(f)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return JSONResponse({"status": "healthy"})


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Ampere EV environment server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()
    main(host=args.host, port=args.port)