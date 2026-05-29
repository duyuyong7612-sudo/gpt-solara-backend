"""FastAPI router exposing the orchestrator at /api/adu/orchestrator/*.

Wire into the running ChatAGI 8000 backend (`server_session.py`):

    from adu_orchestrator.routes import router as orch_router
    app.include_router(orch_router, prefix="/api")
"""
from __future__ import annotations

from typing import Optional

try:
    from fastapi import APIRouter
    from pydantic import BaseModel
    FASTAPI_OK = True
except Exception:  # pragma: no cover
    FASTAPI_OK = False

from . import dispatcher


router = None


if FASTAPI_OK:
    router = APIRouter(prefix="/adu/orchestrator", tags=["adu-orchestrator"])

    class RunReq(BaseModel):
        task: str
        mode: str = "auto"
        project_dir: Optional[str] = None
        safety_level: str = "normal"

    @router.post("/run")
    def run_endpoint(req: RunReq):
        return dispatcher.run_task(
            task=req.task,
            mode=req.mode,
            project_dir=req.project_dir,
            safety_level=req.safety_level,
        )

    @router.get("/info")
    def info():
        return {
            "ok": True,
            "service": "adu_orchestrator",
            "version": "0.1.0",
            "modes": ["code", "browser", "desktop", "auto"],
            "executors": {
                "code": "adu_code_agent (search / list / read); Codex CLI + Claude Code = stubs",
                "browser": "stub (browser-use / agent-browser planned)",
                "desktop": "local-agent :4317 (screenshot / mouse_position / active_window / click / double_click / type / paste / press)",
            },
        }
