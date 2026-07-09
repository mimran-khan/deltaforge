"""DeltaForge Dashboard -- FastAPI server.

Run: python -m dashboard.server
     make dashboard
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from dashboard.routes import router
from dashboard.websocket import start_watcher, stop_watcher, ws_endpoint

STATIC_DIR = Path(__file__).resolve().parent.parent / "dashboard-mock"

DASHBOARD_HOST = getattr(settings, "DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = int(getattr(settings, "DASHBOARD_PORT", 8900))

_cors_env = getattr(settings, "DASHBOARD_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else [f"http://localhost:{DASHBOARD_PORT}", f"http://127.0.0.1:{DASHBOARD_PORT}"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    start_watcher(loop)
    yield
    stop_watcher()


app = FastAPI(
    title="DeltaForge Dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.add_api_websocket_route("/ws/live", ws_endpoint)

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    import uvicorn
    uvicorn.run(
        "dashboard.server:app",
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
