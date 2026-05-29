"""Adu Orchestrator dispatcher.

run_task(task, mode, project_dir, safety_level) →
  {plan, actions, result, needs_user_confirmation, mode, task, engine?, raw?}

mode = "code" | "browser" | "desktop" | "auto"
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .executors import code as code_exec
from .executors import browser as browser_exec
from .executors import desktop as desktop_exec


_DESKTOP_HINTS_CN = (
    "截图", "截屏", "看屏幕", "看一下屏幕", "前台", "当前应用", "当前窗口",
    "鼠标", "点击", "点一下", "双击", "右键", "输入", "打字", "粘贴",
    "按键", "热键", "回车", "退出键",
)
_DESKTOP_HINTS_EN = (
    "screenshot", "active window", "frontmost", "mouse position",
    "click", "double click", "right click", "type ", "paste ",
    "press ", "hotkey",
)
_BROWSER_HINTS = (
    "http://", "https://", "www.", "浏览器", "browser",
    "打开网页", "打开网站", "go to ", "navigate to ",
)


def _classify(task: str) -> str:
    """Heuristic auto-routing for mode='auto'. Conservative: code by default."""
    t = (task or "").strip()
    if not t:
        return "code"
    low = t.lower()
    if any(k in t for k in _DESKTOP_HINTS_CN) or any(k in low for k in _DESKTOP_HINTS_EN):
        return "desktop"
    if any(k in t for k in _BROWSER_HINTS) or any(k in low for k in _BROWSER_HINTS):
        return "browser"
    return "code"


def run_task(task: str,
             mode: str = "auto",
             project_dir: Optional[str] = None,
             safety_level: str = "normal") -> Dict[str, Any]:
    m = (mode or "auto").strip().lower()
    if m == "auto":
        m = _classify(task)

    kwargs = {"task": task, "project_dir": project_dir, "safety_level": safety_level}
    if m == "code":
        out = code_exec.run(**kwargs)
    elif m == "browser":
        out = browser_exec.run(**kwargs)
    elif m == "desktop":
        out = desktop_exec.run(**kwargs)
    else:
        out = {
            "plan": [],
            "actions": [],
            "result": f"unknown mode: {mode!r}",
            "needs_user_confirmation": True,
            "engine": "none",
        }
    # Normalize envelope
    out.setdefault("plan", [])
    out.setdefault("actions", [])
    out.setdefault("result", "")
    out.setdefault("needs_user_confirmation", False)
    out["mode"] = m
    out["task"] = task
    out["safety_level"] = safety_level
    if project_dir is not None:
        out["project_dir"] = project_dir
    return out
