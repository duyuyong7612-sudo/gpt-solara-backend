"""Claude Code CLI executor — automation engine for ChatAGI Adu Self / 自动化编程.

Default engine; Codex CLI remains as fallback via AUTOMATION_ENGINE=codex.

Exposes /api/adu/automation/health and /api/adu/automation/run. Reuses the
exact same safety primitives as codex_executor (path sandbox, risk-term gate,
output sanitization, safety prompt prefix, DEVNULL stdin, timeout, shell=False)
so behavior across engines is identical except for the underlying CLI binary
and model.

Never reads, prints, or transmits ANTHROPIC_API_KEY. The Claude Code CLI
manages its own credential storage; this module only sets ANTHROPIC_MODEL.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

try:
    import adu_project_registry as _project_registry  # type: ignore
except Exception:  # pragma: no cover
    _project_registry = None  # type: ignore

# Reuse the codex_executor safety primitives verbatim — same allow-paths,
# same risk terms, same redaction, same safety preamble. Zero drift.
from .codex_executor import (
    SAFETY_PREFIX,
    find_confirmation_terms,
    _is_path_allowed,
    _resolve_path_no_tilde,
    _configured_allow_paths,
    sanitize_output,
    run_codex_task,
    check_codex_available,
)


# ===== Engine / model selection (env-driven; safe defaults) ============
AUTOMATION_ENGINE = (os.getenv("AUTOMATION_ENGINE") or "claude").strip().lower()
AUTOMATION_MODEL  = (os.getenv("AUTOMATION_MODEL")  or "claude-opus-4-8").strip()

DEFAULT_CLAUDE_TIMEOUT_SEC = 900
MAX_CLAUDE_TIMEOUT_SEC = int(os.getenv("CLAUDE_MAX_TIMEOUT_SEC", "1800") or "1800")


def _claude_binary() -> Optional[str]:
    return shutil.which("claude")


def _claude_env(model: Optional[str] = None) -> Dict[str, str]:
    """Build subprocess env. Only sets ANTHROPIC_MODEL; never reads or
    forwards ANTHROPIC_API_KEY (the Claude CLI handles its own auth)."""
    env = os.environ.copy()
    chosen_model = (model or AUTOMATION_MODEL or "").strip() or "claude-opus-4-8"
    env["ANTHROPIC_MODEL"] = chosen_model
    return env


def check_claude_available(test_model: bool = False) -> Dict[str, Any]:
    """Probe Claude Code CLI: which claude + claude --version (+ optional
    no-tool model ping). Never returns or logs API key values."""
    path = _claude_binary()
    if not path:
        return {
            "ok": False,
            "error": "claude_not_found",
            "message": "Claude Code CLI not found on PATH. Install per https://docs.claude.com/claude-code",
            "claude_path": None,
            "version": "",
            "model_test_ok": False,
            "model": AUTOMATION_MODEL,
        }

    # 1) Version probe
    try:
        proc = subprocess.run(
            [path, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            shell=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": "claude_version_failed",
            "message": f"Claude CLI exists but `claude --version` failed: {e}",
            "claude_path": path,
            "version": "",
            "model_test_ok": False,
            "model": AUTOMATION_MODEL,
        }
    version = sanitize_output((proc.stdout or proc.stderr or "")).strip()

    data: Dict[str, Any] = {
        "ok": proc.returncode == 0,
        "error": None if proc.returncode == 0 else "claude_version_nonzero",
        "message": "Claude CLI is available" if proc.returncode == 0 else "claude --version returned non-zero",
        "claude_path": path,
        "version": version,
        "exit_code": proc.returncode,
        "model": AUTOMATION_MODEL,
        "model_test_ok": False,
    }

    # 2) Optional live model ping (opt-in only — keeps /health fast)
    if test_model and proc.returncode == 0:
        try:
            ping = subprocess.run(
                [path, "--print", "请只回复：automation claude 4.8 ok"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=45,
                shell=False,
                env=_claude_env(),
            )
            ping_stdout = sanitize_output(ping.stdout or "")
            ping_stderr = sanitize_output(ping.stderr or "")
            data["model_test_ok"] = ping.returncode == 0 and bool(ping_stdout.strip())
            data["model_test_exit_code"] = ping.returncode
            data["model_test_stdout"] = ping_stdout[:400]
            if not data["model_test_ok"]:
                data["model_test_stderr"] = ping_stderr[:400]
        except subprocess.TimeoutExpired:
            data["model_test_ok"] = False
            data["model_test_error"] = "claude_model_ping_timeout"
        except Exception as e:
            data["model_test_ok"] = False
            data["model_test_error"] = f"claude_model_ping_exception:{type(e).__name__}"

    return data


def run_claude_task(
    task: str,
    project_dir: str,
    timeout: int = DEFAULT_CLAUDE_TIMEOUT_SEC,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single Claude Code task headlessly. Return shape is compatible
    with run_codex_task so adu_self_sessions.record_codex_result and other
    downstream consumers keep working unchanged."""
    start = time.time()
    task = (task or "").strip()
    chosen_model = (model or AUTOMATION_MODEL or "").strip() or "claude-opus-4-8"

    def _envelope(**extra: Any) -> Dict[str, Any]:
        elapsed = time.time() - start
        base = {
            "engine": "claude",
            "provider": "anthropic",
            "model": chosen_model,
            "duration": round(elapsed, 3),
            "duration_ms": int(elapsed * 1000),
            "project_dir": project_dir,
            "timeout": timeout,
        }
        base.update(extra)
        return base

    if not task:
        return _envelope(
            ok=False, error="missing_task", message="task is required",
            stdout="", stderr="", exit_code=None, returncode=None,
            command=[], needs_user_confirmation=False,
        )

    try:
        timeout_i = int(timeout or DEFAULT_CLAUDE_TIMEOUT_SEC)
    except Exception:
        timeout_i = DEFAULT_CLAUDE_TIMEOUT_SEC
    timeout_i = max(1, min(timeout_i, MAX_CLAUDE_TIMEOUT_SEC))

    # Gate 1: risk-term confirmation
    hits = find_confirmation_terms(task)
    if hits:
        return _envelope(
            ok=False, error="needs_user_confirmation",
            message="Task contains risk terms and was not executed. Ask the user for explicit confirmation before running.",
            needs_user_confirmation=True, blocked_terms=hits,
            stdout="", stderr="", exit_code=None, returncode=None, command=[],
            timeout=timeout_i,
        )

    # Gate 2: project path sandbox
    allowed, reason, allow_paths = _is_path_allowed(project_dir)
    if not allowed:
        return _envelope(
            ok=False, error="project_dir_not_allowed", message=reason,
            allow_paths=allow_paths, stdout="", stderr="",
            exit_code=None, returncode=None, command=[],
            needs_user_confirmation=False, timeout=timeout_i,
        )
    project = _resolve_path_no_tilde(project_dir)
    if not project.exists() or not project.is_dir():
        return _envelope(
            ok=False, error="project_dir_missing",
            message=f"project_dir does not exist or is not a directory: {project}",
            stdout="", stderr="", exit_code=None, returncode=None, command=[],
            needs_user_confirmation=False, timeout=timeout_i,
            project_dir=str(project),
        )

    # Gate 3: CLI must be reachable
    available = check_claude_available(test_model=False)
    if not available.get("ok"):
        return _envelope(
            ok=False,
            error=available.get("error") or "claude_unavailable",
            message=available.get("message") or "Claude CLI not available",
            stdout=sanitize_output(str(available.get("stdout") or "")),
            stderr=sanitize_output(str(available.get("stderr") or "")),
            exit_code=available.get("exit_code"),
            returncode=available.get("exit_code"),
            command=[], needs_user_confirmation=False,
            timeout=timeout_i, project_dir=str(project),
        )

    claude_path = str(available.get("claude_path") or "claude")
    full_prompt = SAFETY_PREFIX + task
    cmd = [claude_path, "--print", full_prompt]

    # Command exposed to the caller never includes the raw prompt body, so
    # logs/UI cannot accidentally surface user prompt text. Length only.
    cmd_for_response = [claude_path, "--print", f"<task:{len(full_prompt)} chars>"]

    env = _claude_env(model=chosen_model)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_i,
            shell=False,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        return _envelope(
            ok=False, error="claude_exec_timeout",
            message=f"claude --print timed out after {timeout_i} seconds",
            stdout=sanitize_output(e.stdout or ""),
            stderr=sanitize_output(e.stderr or ""),
            exit_code=124, returncode=124,
            command=cmd_for_response, needs_user_confirmation=False,
            timeout=timeout_i, project_dir=str(project), timed_out=True,
        )
    except Exception as e:
        return _envelope(
            ok=False, error="claude_exec_exception",
            message=f"claude --print failed: {type(e).__name__}: {e}",
            stdout="", stderr="", exit_code=None, returncode=None,
            command=cmd_for_response, needs_user_confirmation=False,
            timeout=timeout_i, project_dir=str(project),
        )

    stdout = sanitize_output(proc.stdout or "")
    stderr = sanitize_output(proc.stderr or "")
    exit_code = int(proc.returncode)
    ok = exit_code == 0

    return _envelope(
        ok=ok,
        error=None if ok else "claude_exec_failed",
        message="Claude task completed" if ok else f"claude --print returned exit code {exit_code}",
        stdout=stdout, stderr=stderr,
        exit_code=exit_code, returncode=exit_code,
        command=cmd_for_response, needs_user_confirmation=False,
        timeout=timeout_i, project_dir=str(project),
    )


# ===== Engine selector + router ========================================
def _resolve_engine(requested: Optional[str]) -> str:
    eng = (requested or AUTOMATION_ENGINE or "claude").strip().lower()
    return "codex" if eng == "codex" else "claude"


if APIRouter is not None:
    router = APIRouter(prefix="/api/adu/automation", tags=["adu-automation"])

    class AutomationRunRequest(BaseModel):  # type: ignore[misc]
        task: str = Field(..., min_length=1, description="Task passed to the chosen automation engine")
        project_id: Optional[str] = Field(default=None, description="Registry id, e.g. gptsora / backend / little_beijing")
        project_dir: Optional[str] = Field(default=None, description="Legacy: absolute project directory used as subprocess cwd")
        timeout: int = Field(DEFAULT_CLAUDE_TIMEOUT_SEC, ge=1, le=MAX_CLAUDE_TIMEOUT_SEC)
        engine: Optional[str] = Field(default=None, description="Override AUTOMATION_ENGINE per-request: claude | codex")
        model: Optional[str] = Field(default=None, description="Override AUTOMATION_MODEL per-request (Claude engine only)")

    @router.get("/health")
    def automation_health(test_model: bool = False) -> Dict[str, Any]:
        engine = _resolve_engine(None)
        claude_chk = check_claude_available(test_model=test_model)
        codex_chk  = check_codex_available()

        primary_ok = claude_chk.get("ok", False) if engine == "claude" else codex_chk.get("ok", False)
        fallback_ok = codex_chk.get("ok", False) if engine == "claude" else claude_chk.get("ok", False)

        return {
            "ok": bool(primary_ok),
            "automation_engine": engine,
            "automation_model": AUTOMATION_MODEL,
            "claude_cli_exists": bool(claude_chk.get("claude_path")),
            "claude_model_test_ok": bool(claude_chk.get("model_test_ok")) if test_model else None,
            "codex_cli_exists": bool(codex_chk.get("codex_path")),
            "fallback_available": bool(fallback_ok),
            "claude": claude_chk,
            "codex": codex_chk,
            "allow_paths": _configured_allow_paths(),
        }

    @router.post("/run")
    def automation_run(req: AutomationRunRequest) -> Dict[str, Any]:
        # Resolve project_id → path (registry first)
        resolved_dir: Optional[str] = None
        resolved_id: Optional[str] = None
        if req.project_id and _project_registry is not None:
            resolved_id = _project_registry.normalize_project_id(req.project_id)
            if resolved_id:
                resolved_dir = _project_registry.resolve_path(resolved_id)
        if not resolved_dir:
            resolved_dir = (req.project_dir or "").strip() or None
        if not resolved_dir:
            if HTTPException is not None:
                raise HTTPException(status_code=422, detail={
                    "ok": False, "error": "missing_project",
                    "message": "Provide either project_id (registry) or project_dir (absolute path).",
                })
            return {"ok": False, "error": "missing_project"}

        engine = _resolve_engine(req.engine)
        if engine == "codex":
            result = run_codex_task(task=req.task, project_dir=resolved_dir, timeout=req.timeout)
            if isinstance(result, dict):
                result.setdefault("engine", "codex")
                result.setdefault("provider", "openai")
        else:
            result = run_claude_task(
                task=req.task,
                project_dir=resolved_dir,
                timeout=req.timeout,
                model=req.model,
            )

        if isinstance(result, dict):
            result.setdefault("project_id", resolved_id)
            result.setdefault("project_dir", resolved_dir)
        return result
else:  # pragma: no cover
    router = None  # type: ignore
