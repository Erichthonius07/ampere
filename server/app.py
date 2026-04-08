"""
FastAPI application for the Ampere EV Routing Environment.

Endpoints:
    POST /reset  — Reset the environment for a new episode
    POST /step   — Execute an action
    GET  /state  — Get current environment state
    GET  /schema — Get action/observation schemas
    WS   /ws     — WebSocket endpoint for persistent sessions

Usage:
    # Development:
    uvicorn ampere.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn ampere.server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from .models import EVAction, EVObservation
    from ampere_environment import AmpereEnvironment
except ModuleNotFoundError:
    from models import EVAction, EVObservation
    from server.ampere_environment import AmpereEnvironment


app = create_app(
    AmpereEnvironment,
    EVAction,
    EVObservation,
    env_name="ampere",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)