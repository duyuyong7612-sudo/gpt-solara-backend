"""browser_executor — V1 stub for browser-use / agent-browser.

Returns a plan-only response. See orchestrator README for wiring.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def run(task: str,
        project_dir: Optional[str] = None,
        safety_level: str = "normal") -> Dict[str, Any]:
    return {
        "plan": [
            f"task: {task!r}",
            "browser executor not yet wired",
            "candidates: browser-use, agent-browser, Playwright + LLM",
        ],
        "actions": [],
        "result": (
            "browser executor is a V1 stub. See orchestrator README for the "
            "~6-line wiring to browser-use or agent-browser."
        ),
        "needs_user_confirmation": True,
        "engine": "browser_stub",
    }
