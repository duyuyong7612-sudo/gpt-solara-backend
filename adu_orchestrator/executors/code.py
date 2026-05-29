"""code_executor — wraps the local adu_code_agent for project work.

Engine priority (per spec):
  1) local adu_code_agent  ← wired
  2) Codex CLI             ← stub (see _codex_cli_stub)
  3) Claude Code           ← stub (see _claude_code_stub)

V1 routes a few common natural-language intents to adu_code_agent
primitives (search / list / read). Free-form code tasks return a
plan-only response with needs_user_confirmation=true.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

try:
    from adu_code_agent import fs_tools as _aca_fs
    LOCAL_OK = True
    LOCAL_ERR: Optional[str] = None
except Exception as _err:  # pragma: no cover
    LOCAL_OK = False
    LOCAL_ERR = f"{type(_err).__name__}: {_err}"
    _aca_fs = None  # type: ignore


# Adapter stubs ----------------------------------------------------------

def _codex_cli_stub(task: str, project_dir: Optional[str]) -> Dict[str, Any]:
    return {
        "plan": [f"codex CLI would handle: {task!r}"],
        "actions": [],
        "result": "Codex CLI adapter not wired in V1. See orchestrator README "
                  "for the 6-line wiring stub.",
        "needs_user_confirmation": True,
        "engine": "codex_cli_stub",
    }


def _claude_code_stub(task: str, project_dir: Optional[str]) -> Dict[str, Any]:
    return {
        "plan": [f"claude code would handle: {task!r}"],
        "actions": [],
        "result": "Claude Code adapter not wired in V1. See orchestrator README.",
        "needs_user_confirmation": True,
        "engine": "claude_code_stub",
    }


# Intent parsing ---------------------------------------------------------

_SEARCH_PAT = re.compile(
    r'(?:search|查找|搜索|grep|找一下|找)\s+(?P<q>"[^"]+"|\'[^\']+\'|\S+)'
    r'(?:\s+(?:in|在)\s+(?P<scope>\S+))?',
    re.I,
)
_LIST_PAT = re.compile(r'(?:list|ls|列出|看一下目录)\s+(?P<scope>\S+)', re.I)
_READ_PAT = re.compile(
    r'(?:read|读)\s+(?P<path>\S+?)(?::(?P<s>\d+)?(?:-(?P<e>\d+))?)?\b',
    re.I,
)


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    return s


def _local_handle(task: str, project_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """Try to satisfy the task with adu_code_agent. Return None if nothing matches."""
    t = (task or "").strip()
    if not t:
        return None
    root = (project_dir or "").strip()

    m = _SEARCH_PAT.search(t)
    if m:
        query = _strip_quotes(m.group("q"))
        scope = (m.group("scope") or "").strip() or root
        if not scope:
            return {
                "plan": [f"search {query!r} (no scope)"],
                "actions": [],
                "result": "search intent matched but no project_dir / scope was provided",
                "needs_user_confirmation": True,
                "engine": "adu_code_agent",
            }
        result = _aca_fs.search_text(scope, query)
        return {
            "plan": [f"adu_code_agent.search_text({scope!r}, {query!r})"],
            "actions": [{
                "tool": "adu_code_agent.search_text",
                "scope": scope, "query": query,
                "ok": result.get("ok"),
                "match_count": result.get("match_count"),
            }],
            "result": (
                f"{result.get('match_count', 0)} match(es) for {query!r} in {scope}"
                if result.get("ok") else f"search failed: {result.get('error')}"
            ),
            "needs_user_confirmation": False,
            "engine": "adu_code_agent",
            "raw": result,
        }

    m = _LIST_PAT.search(t)
    if m:
        scope = m.group("scope")
        if scope in (".", "./"):
            scope = root or "."
        result = _aca_fs.list_files(scope)
        return {
            "plan": [f"adu_code_agent.list_files({scope!r})"],
            "actions": [{
                "tool": "adu_code_agent.list_files",
                "scope": scope,
                "ok": result.get("ok"),
                "count": result.get("count"),
            }],
            "result": (
                f"{result.get('count', 0)} file(s) under {scope}"
                if result.get("ok") else f"list failed: {result.get('error')}"
            ),
            "needs_user_confirmation": False,
            "engine": "adu_code_agent",
            "raw": result,
        }

    m = _READ_PAT.search(t)
    if m:
        path = m.group("path")
        start = int(m.group("s")) if m.group("s") else 1
        end = int(m.group("e")) if m.group("e") else None
        result = _aca_fs.read_file(path, start_line=start, end_line=end)
        return {
            "plan": [f"adu_code_agent.read_file({path!r}, {start}, {end})"],
            "actions": [{
                "tool": "adu_code_agent.read_file",
                "path": path, "ok": result.get("ok"),
            }],
            "result": (
                f"read {path} lines {result.get('start_line')}–{result.get('end_line')}"
                if result.get("ok") else f"read failed: {result.get('error')}"
            ),
            "needs_user_confirmation": False,
            "engine": "adu_code_agent",
            "raw": result,
        }

    return None


def run(task: str,
        project_dir: Optional[str] = None,
        safety_level: str = "normal") -> Dict[str, Any]:
    if not LOCAL_OK:
        return {
            "plan": [],
            "actions": [],
            "result": f"adu_code_agent unavailable — {LOCAL_ERR}",
            "needs_user_confirmation": True,
            "engine": "none",
        }
    out = _local_handle(task, project_dir)
    if out is not None:
        return out
    # Free-form code task → not yet wired; point to Codex CLI / Claude Code adapters.
    return {
        "plan": [
            f"task: {task!r}",
            "no built-in intent matched (search/list/read)",
            "next: wire Codex CLI adapter OR Claude Code adapter",
        ],
        "actions": [],
        "result": (
            "V1 code executor only handles 'search X [in PATH]', 'list PATH', "
            "and 'read PATH[:start-end]'. For free-form coding, integrate "
            "Codex CLI or Claude Code (see orchestrator README)."
        ),
        "needs_user_confirmation": True,
        "engine": "adu_code_agent",
    }
