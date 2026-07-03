"""WebSocket manager with file-change notifications.

Uses watchdog to monitor data files and pushes updates to connected clients.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

from config import settings
from dashboard import data_access as da


class ConnectionManager:
    """Track active WebSocket connections and broadcast messages."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        async with self._lock:
            stale = []
            for ws in self._connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._connections.remove(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()

_loop: Optional[asyncio.AbstractEventLoop] = None


def _set_loop(loop: asyncio.AbstractEventLoop):
    global _loop
    _loop = loop


def _push(message: dict):
    """Thread-safe push from watchdog callbacks into the async event loop."""
    if _loop is None or _loop.is_closed():
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(message), _loop)


class _DataFileHandler(FileSystemEventHandler):
    """React to changes in data/ directory."""

    def __init__(self):
        self._debounce: dict[str, float] = {}

    def _should_handle(self, path: str) -> bool:
        import time
        now = time.monotonic()
        last = self._debounce.get(path, 0)
        if now - last < 0.5:
            return False
        self._debounce[path] = now
        return True

    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if not self._should_handle(event.src_path):
            return

        if path.name == "capital.json":
            _push({"type": "capital", "data": da.read_capital()})
        elif path.name == "engine_state.json":
            state = da.read_engine_state()
            if state:
                _push({"type": "engine", "data": state})
        elif path.name == "multi_asset_state.json":
            state = da.read_multi_asset_state()
            if state:
                _push({"type": "multi_asset", "data": state})
        elif path.name == "events.jsonl":
            evts = da.get_events(limit=1)
            if evts:
                _push({"type": "event", "data": evts[0]})
        elif path.suffix == ".jsonl" and "json" in str(path.parent):
            logs = da.get_logs(limit=1)
            if logs:
                _push({"type": "log", "data": logs[0]})

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.name == "HALT":
            _push({"type": "halt", "data": da.read_halt()})

    def on_deleted(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.name == "HALT":
            _push({"type": "halt", "data": None})


_observer: Optional[Observer] = None


def start_watcher(loop: asyncio.AbstractEventLoop):
    """Start the file watcher in a background thread."""
    global _observer
    _set_loop(loop)

    if _observer is not None:
        return

    handler = _DataFileHandler()
    _observer = Observer()

    data_dir = str(settings.DATA_DIR)
    _observer.schedule(handler, data_dir, recursive=False)

    log_json_dir = settings.LOG_DIR / "json"
    if log_json_dir.exists():
        _observer.schedule(handler, str(log_json_dir), recursive=False)

    _observer.daemon = True
    _observer.start()


def stop_watcher():
    global _observer
    if _observer:
        _observer.stop()
        _observer.join(timeout=2)
        _observer = None


async def ws_endpoint(ws: WebSocket):
    """WebSocket handler -- keeps connection alive and pushes file updates."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)
