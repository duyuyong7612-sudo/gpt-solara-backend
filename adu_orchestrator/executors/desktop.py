"""desktop_executor — talks to the local-agent on :4317.

V1 parses simple desktop intents from the task string and calls one
local-agent endpoint per task:

  screenshot / mouse_position / active_window
  click / double_click
  type / paste
  press

For richer multi-step desktop flows (Observe→Decide→Act→Verify), call
the ChatAGI brain at /api/brain/adu/act instead — this executor is the
thin direct path.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTP_OK = True
    _HTTP_ERR: Optional[str] = None
except Exception as _e:  # pragma: no cover
    HTTP_OK = False
    _HTTP_ERR = f"{type(_e).__name__}: {_e}"


LOCAL_AGENT_URL = os.environ.get("LOCAL_AGENT_URL", "http://127.0.0.1:4317").rstrip("/")
LOCAL_AGENT_TOKEN = os.environ.get("LOCAL_AGENT_TOKEN", "local-dev-token")


def _call(method: str, path: str,
          json_body: Optional[Dict[str, Any]] = None,
          timeout: float = 12.0) -> Dict[str, Any]:
    url = f"{LOCAL_AGENT_URL}{path}"
    headers = {"Authorization": f"Bearer {LOCAL_AGENT_TOKEN}"}
    try:
        with httpx.Client(timeout=timeout) as c:
            if method == "GET":
                r = c.get(url, headers=headers)
            else:
                r = c.post(url, headers=headers, json=json_body or {})
        if r.status_code < 400:
            try:
                data = r.json()
            except Exception:
                data = {"raw": (r.text or "")[:400]}
        else:
            data = {"error": (r.text or "")[:400]}
        return {"ok": r.status_code < 400, "status": r.status_code, "data": data}
    except Exception as e:
        return {"ok": False, "error": f"local_agent_unreachable: {type(e).__name__}: {e}"}


_XY_PAT = re.compile(r'(\d{1,5})\s*[,，、\s]+\s*(\d{1,5})')
_TYPE_PAT = re.compile(r'(?:输入|打字|键入|type)\s*[:：]?\s*(.+)$', re.I)
_PASTE_PAT = re.compile(r'(?:粘贴|paste)\s*[:：]?\s*(.+)$', re.I)
_PRESS_PAT = re.compile(
    r'(?:press|按)\s*((?:cmd|command|ctrl|alt|shift|option|⌘|⌃|⌥|⇧|[a-z0-9])'
    r'(?:\s*[+\s,，、]\s*(?:cmd|command|ctrl|alt|shift|option|⌘|⌃|⌥|⇧|[a-z0-9]+))*)',
    re.I,
)


def _parse_intent(task: str) -> Dict[str, Any]:
    t = (task or "").strip()
    low = t.lower()

    # Observe-class
    if any(k in t for k in ("截图", "截屏", "看屏幕", "看一下屏幕")) or "screenshot" in low:
        return {"verb": "screenshot"}
    if (("鼠标" in t) and any(k in t for k in ("在哪", "位置", "现在", "哪里"))) \
            or "mouse position" in low:
        return {"verb": "mouse_position"}
    if "active window" in low or "frontmost" in low \
            or any(k in t for k in ("前台", "当前应用", "当前窗口")):
        return {"verb": "active_window"}

    xy = _XY_PAT.search(t)
    xy_kw = ({"x": int(xy.group(1)), "y": int(xy.group(2))} if xy else {})

    # Click variants
    if "双击" in t or "double click" in low:
        return {"verb": "double_click", "button": "left", **xy_kw}
    if any(k in t for k in ("右键", "点右键", "右击")):
        return {"verb": "click", "button": "right", **xy_kw}
    if "点击" in t or "点一下" in t or "click" in low:
        return {"verb": "click", "button": "left", **xy_kw}

    # Type / paste — capture trailing text
    m = _TYPE_PAT.search(t)
    if m:
        text = m.group(1).strip().strip('"\'""')
        if text:
            return {"verb": "type", "text": text}
    m = _PASTE_PAT.search(t)
    if m:
        text = m.group(1).strip().strip('"\'""')
        if text:
            return {"verb": "paste", "text": text}

    # Press — single keywords first
    if "enter" in low or "回车" in t:
        return {"verb": "press", "keys": ["enter"]}
    if "escape" in low or " esc " in f" {low} " or "退出键" in t:
        return {"verb": "press", "keys": ["escape"]}
    if " tab" in f" {low}" or "制表键" in t:
        return {"verb": "press", "keys": ["tab"]}
    # Press with explicit combo
    mp = _PRESS_PAT.search(t)
    if mp:
        raw = mp.group(1)
        keys = [k.strip().lower() for k in re.split(r"[+\s,，、]+", raw) if k.strip()]
        sym = {"⌘": "cmd", "⌃": "ctrl", "⌥": "alt", "⇧": "shift", "command": "cmd",
               "control": "ctrl", "option": "alt"}
        keys = [sym.get(k, k) for k in keys]
        if keys:
            return {"verb": "press", "keys": keys}

    return {"verb": None}


def run(task: str,
        project_dir: Optional[str] = None,
        safety_level: str = "normal") -> Dict[str, Any]:
    if not HTTP_OK:
        return {
            "plan": [], "actions": [],
            "result": f"no http client available ({_HTTP_ERR})",
            "needs_user_confirmation": True,
            "engine": "local_agent",
        }

    intent = _parse_intent(task)
    v = intent.get("verb")

    if v is None:
        return {
            "plan": [f"task: {task!r}", "no desktop intent matched"],
            "actions": [],
            "result": (
                "desktop executor only handles screenshot / mouse_position / "
                "active_window / click / double_click / type / paste / press. "
                "For richer multi-step flows use /api/brain/adu/act."
            ),
            "needs_user_confirmation": True,
            "engine": "local_agent",
        }

    plan: List[str] = [f"local-agent: {v}"]

    if v == "screenshot":
        r = _call("POST", "/api/computer/screenshot", {})
    elif v == "mouse_position":
        r = _call("GET", "/api/computer/mouse")
    elif v == "active_window":
        r = _call("GET", "/api/computer/active_window")
    elif v in ("click", "double_click"):
        body: Dict[str, Any] = {"button": intent.get("button", "left")}
        if "x" in intent:
            body["x"] = intent["x"]
        if "y" in intent:
            body["y"] = intent["y"]
        if v == "double_click":
            body["count"] = 2
        r = _call("POST", "/api/computer/click", body)
    elif v == "type":
        r = _call("POST", "/api/computer/type", {"text": intent.get("text", "")})
    elif v == "paste":
        r = _call("POST", "/api/computer/paste", {"text": intent.get("text", "")})
    elif v == "press":
        r = _call("POST", "/api/computer/press", {"keys": intent.get("keys", [])})
    else:
        return {
            "plan": plan, "actions": [],
            "result": f"unsupported verb: {v}",
            "needs_user_confirmation": True,
            "engine": "local_agent",
        }

    ok = bool(r.get("ok"))
    return {
        "plan": plan,
        "actions": [{
            "tool": f"local_agent.{v}",
            "intent": intent,
            "ok": ok,
            "status": r.get("status"),
        }],
        "result": (
            f"{v} ok" if ok
            else f"{v} failed: {r.get('error') or (r.get('data') or {}).get('error') or r.get('data')}"
        ),
        "needs_user_confirmation": False,
        "engine": "local_agent",
        "raw": r,
    }
