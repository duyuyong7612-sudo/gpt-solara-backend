from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional for standalone CLI tests. In server_session.py, load_dotenv() is already
# called before this module is imported.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - allows CLI import even if FastAPI is absent
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore

# 项目注册表(可选):允许调用方用 project_id 代替 project_dir。
# 不挂会让接口退回到必填 project_dir,不影响现有行为。
try:
    import adu_project_registry as _project_registry  # type: ignore
except Exception:  # pragma: no cover
    _project_registry = None  # type: ignore


DEFAULT_CODEX_TIMEOUT_SEC = 900
MAX_CODEX_TIMEOUT_SEC = int(os.getenv("CODEX_MAX_TIMEOUT_SEC", "1800") or "1800")
CODEX_SKIP_GIT_REPO_CHECK = (os.getenv("CODEX_SKIP_GIT_REPO_CHECK") or "1").strip().lower() not in ("0", "false", "no", "off")
CODEX_AUTH_MODE = (os.getenv("CODEX_AUTH_MODE") or "chatgpt").strip().lower()

# api_key mode:
# Official Codex non-interactive mode supports CODEX_API_KEY for `codex exec`.
# If CODEX_AUTH_MODE=api_key and CODEX_API_KEY is absent, bridge OPENAI_API_KEY
# into CODEX_API_KEY for the child process without printing the value.
CODEX_API_KEY_FALLBACK_ENV_NAMES = (
    "OPENAI_API_KEY",
)

# Keep this list intentionally narrow. Override in local .env with CODEX_ALLOW_PATHS
# when your local username/project path differs.
DEFAULT_ALLOW_PATHS = [
    "/Users/a12345/Desktop/little-beijing-edge-box",
    "/Users/a12345/Desktop/GPTsora",
    "/Users/a12345/Desktop/backend",
]

# Terms that must stop execution and ask the product/user layer for confirmation.
# Matching is case-insensitive for English terms and direct substring for Chinese.
CONFIRMATION_TERMS = [
    "rm -rf",
    "sudo",
    "git push",
    "npm publish",
    "pip install",
    "pip3 install",
    "npm install",
    "pnpm add",
    "yarn add",
    "brew install",
    "修改 .env",
    "修改.env",
    ".env",
    "删除",
    "payment",
    "付款",
    "api key",
    "api_key",
    "apikey",
]

SECRET_NAME_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|pwd|credential)")
ENV_ASSIGNMENT_RE = re.compile(
    r"(?im)^([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD|CREDENTIAL)[A-Z0-9_]*\s*=\s*)([^\s#]+).*$"
)
OPENAI_STYLE_SECRET_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{12,}|[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,})\b")

SAFETY_PREFIX = """You are running inside the ChatAGI Adu Codex executor.
Safety constraints:
- Do not run git push, npm publish, payment-related operations, destructive deletes, sudo, or dependency installs.
- Do not print, reveal, modify, or copy .env contents or API keys.
- Stay inside the current project directory.
- If a requested action requires the blocked operations above, stop and explain what confirmation is needed.

User task:
"""


def _split_allow_paths(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    # Accept comma, newline, or os.pathsep separated env values.
    parts: List[str] = []
    for chunk in re.split(r"[,\n]", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([p.strip() for p in chunk.split(os.pathsep) if p.strip()])
    return parts


def _configured_allow_paths() -> List[str]:
    env_paths = _split_allow_paths(os.getenv("CODEX_ALLOW_PATHS", ""))
    return env_paths or DEFAULT_ALLOW_PATHS


def _resolve_path_no_tilde(raw_path: str) -> Path:
    raw = (raw_path or "").strip()
    if not raw:
        raise ValueError("project_dir is required")
    if raw == "~" or raw.startswith("~/"):
        raise ValueError("project_dir cannot be '~' or use '~/' shorthand; use an explicit allowed project path")
    p = Path(raw)
    if not p.is_absolute():
        raise ValueError("project_dir must be an absolute path")
    return p.resolve(strict=False)


def _is_path_allowed(project_dir: str, allow_paths: Optional[List[str]] = None) -> Tuple[bool, str, List[str]]:
    try:
        project = _resolve_path_no_tilde(project_dir)
    except ValueError as e:
        return False, str(e), allow_paths or _configured_allow_paths()

    dangerous_exact = {Path("/").resolve(strict=False)}
    try:
        dangerous_exact.add(Path.home().resolve(strict=False))
    except Exception:
        pass
    if project in dangerous_exact:
        return False, "project_dir cannot be root or the user home directory", allow_paths or _configured_allow_paths()

    configured = allow_paths or _configured_allow_paths()
    allowed_roots: List[Path] = []
    for item in configured:
        try:
            if item.strip() in ("/", "~") or item.strip().startswith("~/"):
                continue
            allowed_roots.append(Path(item).resolve(strict=False))
        except Exception:
            continue

    for root in allowed_roots:
        try:
            project.relative_to(root)
            return True, "", [str(p) for p in allowed_roots]
        except ValueError:
            continue
    return False, f"project_dir is not in CODEX_ALLOW_PATHS: {project}", [str(p) for p in allowed_roots]


def find_confirmation_terms(task: str) -> List[str]:
    text = (task or "").strip()
    lower = text.lower()
    hits: List[str] = []
    for term in CONFIRMATION_TERMS:
        t = term.lower()
        if (t in lower) and term not in hits:
            hits.append(term)
    return hits


def _secret_env_values() -> List[str]:
    values: List[str] = []
    for k, v in os.environ.items():
        if not v or len(v) < 8:
            continue
        if SECRET_NAME_RE.search(k):
            values.append(v)
    # Longest first so partial values do not leave suffixes behind.
    return sorted(set(values), key=len, reverse=True)


def sanitize_output(text: str) -> str:
    """Best-effort redaction so executor responses do not leak API keys or .env values."""
    if not text:
        return ""
    out = str(text)
    for secret in _secret_env_values():
        out = out.replace(secret, "[REDACTED_SECRET]")
    out = ENV_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}[REDACTED]", out)
    out = OPENAI_STYLE_SECRET_RE.sub("[REDACTED_SECRET]", out)
    return out


def _codex_binary() -> Optional[str]:
    return shutil.which("codex")


def _codex_auth_mode() -> str:
    return "api_key" if CODEX_AUTH_MODE == "api_key" else "chatgpt"


def _chatgpt_auth_exists() -> bool:
    """Best-effort login-state check. Never read or return token contents."""
    home = Path.home()
    candidates = [
        home / ".codex" / "auth.json",
        home / "Library" / "Application Support" / "Codex",
    ]
    return any(p.exists() for p in candidates)


def _build_codex_subprocess_env() -> Tuple[Dict[str, str], str]:
    """Return env for `codex exec` and a non-secret auth source label.

    chatgpt mode intentionally removes CODEX_API_KEY and OPENAI_API_KEY so the
    local Codex CLI uses its saved ChatGPT login. api_key mode preserves the
    old CODEX_API_KEY-first behavior and only maps OPENAI_API_KEY when needed.
    Never include key values in API responses or logs.
    """
    env = os.environ.copy()
    if _codex_auth_mode() == "chatgpt":
        env.pop("CODEX_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        return env, "chatgpt_login"

    existing = (env.get("CODEX_API_KEY") or "").strip()
    if existing:
        return env, "CODEX_API_KEY"

    for name in CODEX_API_KEY_FALLBACK_ENV_NAMES:
        value = (env.get(name) or "").strip()
        if value:
            env["CODEX_API_KEY"] = value
            return env, f"{name}->CODEX_API_KEY"

    return env, ""


def _codex_auth_env_status() -> Dict[str, Any]:
    _, auth_source = _build_codex_subprocess_env()
    auth_mode = _codex_auth_mode()
    has_codex = bool((os.getenv("CODEX_API_KEY") or "").strip())
    has_openai = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    if auth_mode == "chatgpt":
        return {
            "auth_mode": "chatgpt",
            "auth_source": "chatgpt_login",
            "chatgpt_auth_exists": _chatgpt_auth_exists(),
            "has_codex_api_key": False,
            "has_openai_api_key": False,
            "openai_api_key_handling": "ignored_in_chatgpt_mode" if has_openai else "absent",
            "codex_api_key_handling": "ignored_in_chatgpt_mode" if has_codex else "absent",
        }
    return {
        "auth_mode": "api_key",
        "auth_source": auth_source or "none",
        "has_codex_api_key": has_codex,
        "has_openai_api_key": has_openai,
    }


def check_codex_available() -> Dict[str, Any]:
    path = _codex_binary()
    if not path:
        data = {
            "ok": False,
            "error": "codex_not_found",
            "message": "Codex CLI is not available. `which codex` returned empty; install/login Codex CLI and ensure it is on PATH.",
        }
        data.update(_codex_auth_env_status())
        return data
    try:
        proc = subprocess.run(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            shell=False,
        )
    except Exception as e:
        data = {
            "ok": False,
            "error": "codex_version_failed",
            "message": f"Codex CLI exists but `codex --version` failed: {e}",
            "codex_path": path,
        }
        data.update(_codex_auth_env_status())
        return data
    stdout = sanitize_output(proc.stdout or "")
    stderr = sanitize_output(proc.stderr or "")
    data = {
        "ok": proc.returncode == 0,
        "error": None if proc.returncode == 0 else "codex_version_nonzero",
        "message": "Codex CLI is available" if proc.returncode == 0 else "Codex CLI returned non-zero for --version",
        "codex_path": path,
        "version": (stdout or stderr).strip(),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": proc.returncode,
    }
    data.update(_codex_auth_env_status())
    if proc.returncode == 0 and _codex_auth_mode() == "chatgpt" and not data.get("chatgpt_auth_exists"):
        data.update({
            "ok": False,
            "error": "codex_chatgpt_not_logged_in",
            "message": "请先在电脑终端运行 codex login --device-auth 或 codex app 登录 ChatGPT。",
        })
    return data


def _looks_like_auth_error(stdout: str, stderr: str, exit_code: Optional[int] = None) -> bool:
    """Detect real Codex auth failures without false-positives from our own prompt.

    Codex writes the full prompt/tool trace to stderr. Our safety prompt contains
    phrases like "API keys", so broad markers such as "auth" or "api key" can
    incorrectly mark a successful run as an auth failure. Treat exit_code=0 as
    success, and only match precise failure text for non-zero runs.
    """
    try:
        if exit_code is not None and int(exit_code) == 0:
            return False
    except Exception:
        pass

    combined = f"{stdout}\n{stderr}".lower()
    markers = [
        "401 unauthorized",
        "missing bearer",
        "missing bearer or basic authentication",
        "unauthorized",
        "not logged in",
        "login required",
        "please login",
        "please log in",
        "authentication failed",
        "invalid api key",
        "invalid_api_key",
    ]
    return any(m in combined for m in markers)


def run_codex_task(task: str, project_dir: str, timeout: int = DEFAULT_CODEX_TIMEOUT_SEC) -> Dict[str, Any]:
    start = time.time()
    task = (task or "").strip()
    if not task:
        return {
            "ok": False,
            "error": "missing_task",
            "message": "task is required",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration": 0,
        }

    try:
        timeout_i = int(timeout or DEFAULT_CODEX_TIMEOUT_SEC)
    except Exception:
        timeout_i = DEFAULT_CODEX_TIMEOUT_SEC
    timeout_i = max(1, min(timeout_i, MAX_CODEX_TIMEOUT_SEC))

    hits = find_confirmation_terms(task)
    if hits:
        return {
            "ok": False,
            "error": "needs_user_confirmation",
            "message": "Task contains risk terms and was not executed. Ask the user for explicit confirmation before running.",
            "needs_user_confirmation": True,
            "blocked_terms": hits,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration": round(time.time() - start, 3),
        }

    allowed, reason, allow_paths = _is_path_allowed(project_dir)
    if not allowed:
        return {
            "ok": False,
            "error": "project_dir_not_allowed",
            "message": reason,
            "allow_paths": allow_paths,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration": round(time.time() - start, 3),
        }

    project = _resolve_path_no_tilde(project_dir)
    if not project.exists() or not project.is_dir():
        return {
            "ok": False,
            "error": "project_dir_missing",
            "message": f"project_dir does not exist or is not a directory: {project}",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration": round(time.time() - start, 3),
        }

    available = check_codex_available()
    if not available.get("ok"):
        available.update({
            "stdout": sanitize_output(str(available.get("stdout") or "")),
            "stderr": sanitize_output(str(available.get("stderr") or "")),
            "exit_code": available.get("exit_code"),
            "duration": round(time.time() - start, 3),
        })
        return available

    codex_path = str(available.get("codex_path") or "codex")
    codex_task = SAFETY_PREFIX + task
    codex_env, auth_source = _build_codex_subprocess_env()
    auth_mode = _codex_auth_mode()

    cmd = [codex_path, "exec"]
    if CODEX_SKIP_GIT_REPO_CHECK:
        cmd.append("--skip-git-repo-check")
    cmd.append(codex_task)

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
            env=codex_env,
        )
        stdout = sanitize_output(proc.stdout or "")
        stderr = sanitize_output(proc.stderr or "")
        exit_code = int(proc.returncode)
        auth_error = _looks_like_auth_error(stdout, stderr, exit_code)
        ok = exit_code == 0 and not auth_error
        if auth_error and auth_mode == "chatgpt":
            error_code = "codex_chatgpt_not_logged_in"
            message = "请先在电脑终端运行 codex login --device-auth 或 codex app 登录 ChatGPT。"
        else:
            error_code = None if ok else ("codex_not_logged_in_or_auth_failed" if auth_error else "codex_exec_failed")
            message = "Codex task completed" if ok else ("Codex CLI authentication failed. Set CODEX_AUTH_MODE=chatgpt for local ChatGPT login, or set CODEX_AUTH_MODE=api_key with CODEX_API_KEY/OPENAI_API_KEY." if auth_error else "Codex exec returned non-zero")
        return {
            "ok": ok,
            "error": error_code,
            "message": message,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "duration": round(time.time() - start, 3),
            "project_dir": str(project),
            "timeout": timeout_i,
            "needs_user_confirmation": False,
            "auth_mode": auth_mode,
            "auth_source": auth_source or "none",
            "skip_git_repo_check": CODEX_SKIP_GIT_REPO_CHECK,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": "codex_exec_timeout",
            "message": f"Codex exec timed out after {timeout_i} seconds",
            "stdout": sanitize_output(e.stdout or ""),
            "stderr": sanitize_output(e.stderr or ""),
            "exit_code": 124,
            "duration": round(time.time() - start, 3),
            "project_dir": str(project),
            "timeout": timeout_i,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": "codex_exec_exception",
            "message": f"Codex exec failed: {e}",
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "duration": round(time.time() - start, 3),
            "project_dir": str(project),
            "timeout": timeout_i,
        }


if APIRouter is not None:
    router = APIRouter(prefix="/api/adu/codex", tags=["adu-codex"])

    class CodexRunRequest(BaseModel):  # type: ignore[misc]
        task: str = Field(..., min_length=1, description="Task passed to `codex exec`")
        # 二者必填其一:project_id 优先(从 registry 查路径),project_dir 兼容老调用。
        project_id: Optional[str] = Field(default=None, description="Registry id, e.g. gptsora / backend / little_beijing")
        project_dir: Optional[str] = Field(default=None, description="Legacy: absolute local project directory used as subprocess cwd")
        timeout: int = Field(DEFAULT_CODEX_TIMEOUT_SEC, ge=1, le=MAX_CODEX_TIMEOUT_SEC)

    @router.get("/health")
    def codex_health() -> Dict[str, Any]:
        data = check_codex_available()
        data["allow_paths"] = _configured_allow_paths()
        data["skip_git_repo_check"] = CODEX_SKIP_GIT_REPO_CHECK
        return data

    @router.post("/run")
    def codex_run(req: CodexRunRequest) -> Dict[str, Any]:
        # 解析 project_id → 路径(registry 优先)
        resolved_dir: Optional[str] = None
        resolved_id: Optional[str] = None
        if req.project_id and _project_registry is not None:
            resolved_id = _project_registry.normalize_project_id(req.project_id)
            if resolved_id:
                resolved_dir = _project_registry.resolve_path(resolved_id)
        # 没有 project_id 或注册表未启用 → 回退到老的 project_dir
        if not resolved_dir:
            resolved_dir = (req.project_dir or "").strip() or None
        if not resolved_dir:
            # 既无合法 project_id 也无 project_dir,拒绝
            if HTTPException is not None:
                raise HTTPException(status_code=422, detail={
                    "ok": False,
                    "error": "missing_project",
                    "message": "Provide either project_id (registry) or project_dir (absolute path).",
                })
            return {"ok": False, "error": "missing_project"}

        result = run_codex_task(task=req.task, project_dir=resolved_dir, timeout=req.timeout)
        # 透出解析结果给调用方,方便 UI 显示"阿杜自动选择工程"
        if isinstance(result, dict):
            result.setdefault("project_id", resolved_id)
            result.setdefault("project_dir", resolved_dir)
        return result
else:  # pragma: no cover
    router = None  # type: ignore


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a ChatAGI Adu Codex CLI task safely.")
    parser.add_argument("--project", "--project-dir", dest="project_dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--timeout", type=int, default=DEFAULT_CODEX_TIMEOUT_SEC)
    args = parser.parse_args()
    result = run_codex_task(args.task, args.project_dir, args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())
