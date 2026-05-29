"""ChatAGI Adu — unified verifier。第一版只做"安全 + 快"的检查:

- verify_python_compile(paths)        真正 py_compile,只允许授权工程内的 .py
- verify_curl(url, method, body)      只允许本机 8000 / 4317,**不读响应敏感字段**
- verify_codex_health()               curl /api/adu/codex/health 的封装
- verify_backend_health()             curl /api/adu/health 或退到 /openapi.json
- verify_project_exists(project_id)   读 adu_project_registry,不做磁盘扫描
- verify_xcodebuild_command(text)     只返回推荐命令字符串,**不执行**

绝不做:
- 不执行 git / pip / brew / 任何包管理
- 不写文件 / 不删文件
- 不调外部网络(只允许 127.0.0.1 / localhost)
"""
from __future__ import annotations

import json
import py_compile
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import adu_project_registry as _registry  # type: ignore
except Exception:  # pragma: no cover
    _registry = None  # type: ignore


# ─── 通用返回壳 ────────────────────────────────────────────────────
def _check(name: str, ok: bool, message: str = "", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "ok": ok, "message": message}
    if details:
        out["details"] = details
    return out


def _wrap(checks: List[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": all(c.get("ok") for c in checks) if checks else False,
        "checks": checks,
    }
    if extra:
        payload.update(extra)
    return payload


# ─── 路径白名单 ────────────────────────────────────────────────────
def _project_roots() -> List[Path]:
    if _registry is None:
        return []
    roots: List[Path] = []
    for p in _registry.list_projects():
        if p.get("exists"):
            try:
                roots.append(Path(p["path"]).resolve())
            except Exception:
                continue
    return roots


def _is_inside_workspace(path: Path) -> bool:
    try:
        real = path.resolve()
    except Exception:
        return False
    for root in _project_roots():
        if real == root or root in real.parents:
            return True
    return False


# ─── 1. py_compile ────────────────────────────────────────────────
def verify_python_compile(paths: List[str]) -> Dict[str, Any]:
    """对 paths 里的 .py 跑 py_compile。只接受授权工作区内、后缀 .py 的文件。"""
    checks: List[Dict[str, Any]] = []
    for raw in paths or []:
        p = Path(raw)
        # 1) 必须是 .py
        if p.suffix != ".py":
            checks.append(_check(f"py_compile {raw}", False, "not_a_python_file"))
            continue
        # 2) 必须在授权工作区
        if not _is_inside_workspace(p):
            checks.append(_check(f"py_compile {raw}", False, "outside_workspace"))
            continue
        # 3) 必须存在 & 是文件
        if not p.is_file():
            checks.append(_check(f"py_compile {raw}", False, "missing"))
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            checks.append(_check(f"py_compile {raw}", True, "ok"))
        except py_compile.PyCompileError as e:
            # py_compile 自带的错误信息已含行号;不暴露绝对路径里的家目录是可以的(它本来就在 workspace 里)
            checks.append(_check(f"py_compile {raw}", False, str(e)[:400]))
        except Exception as e:
            checks.append(_check(f"py_compile {raw}", False, f"{type(e).__name__}: {e}"[:400]))
    return _wrap(checks)


# ─── 2. curl(本机限定) ────────────────────────────────────────────
_ALLOWED_CURL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
_ALLOWED_CURL_PORTS = {8000, 4317}
_ALLOWED_METHODS = {"GET", "POST"}


def _is_localhost_url(url: str) -> Tuple[bool, str]:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "invalid_url"
    if parsed.scheme not in ("http",):
        return False, "scheme_must_be_http"
    host = parsed.hostname or ""
    port = parsed.port or (80 if parsed.scheme == "http" else 0)
    if host not in _ALLOWED_CURL_HOSTS:
        return False, f"host_not_allowed:{host}"
    if port not in _ALLOWED_CURL_PORTS:
        return False, f"port_not_allowed:{port}"
    return True, "ok"


_BANNED_RESPONSE_KEYS = {
    "api_key", "openai_api_key", "codex_api_key", "anthropic_api_key", "sora_api_key",
    "secret", "secrets", "credentials", "credential",
    "token", "access_token", "refresh_token", "bearer",
    "password", "passwd",
}


def _scrub(obj: Any) -> Any:
    """递归把响应里的敏感字段值替换成 '***',不直接抛弃,保留 key 让调用方看出有这字段。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in _BANNED_RESPONSE_KEYS:
                out[k] = "***"
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def verify_curl(url: str, method: str = "GET", body: Optional[Dict[str, Any]] = None,
                timeout: int = 8) -> Dict[str, Any]:
    """本机 curl。响应体最大 8KB;敏感字段值会被替换成 '***'。"""
    ok_local, why = _is_localhost_url(url)
    if not ok_local:
        return _wrap([_check(f"curl {method} {url}", False, why)])
    m = (method or "GET").upper()
    if m not in _ALLOWED_METHODS:
        return _wrap([_check(f"curl {m} {url}", False, "method_not_allowed")])

    data = None
    headers = {"Accept": "application/json"}
    if body is not None and m == "POST":
        try:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            return _wrap([_check(f"curl {m} {url}", False, "invalid_body_json")])
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=m)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read(8 * 1024)
        text = raw.decode("utf-8", errors="replace")
        parsed: Any
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        scrubbed = _scrub(parsed) if isinstance(parsed, (dict, list)) else None
        return _wrap(
            [_check(f"curl {m} {url}", 200 <= status < 300, f"HTTP {status}")],
            extra={"http_status": status, "response_json": scrubbed, "response_text_head": text[:512] if scrubbed is None else None},
        )
    except urllib.error.HTTPError as e:
        body_head = ""
        try:
            body_head = e.read(2048).decode("utf-8", errors="replace")
        except Exception:
            pass
        return _wrap(
            [_check(f"curl {m} {url}", False, f"HTTP {e.code}")],
            extra={"http_status": e.code, "response_text_head": body_head},
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return _wrap([_check(f"curl {m} {url}", False, f"{type(e).__name__}: {e}"[:200])])


# ─── 3. codex health ─────────────────────────────────────────────
def verify_codex_health() -> Dict[str, Any]:
    return verify_curl("http://127.0.0.1:8000/api/adu/codex/health", method="GET", timeout=8)


# ─── 4. backend health ────────────────────────────────────────────
def verify_backend_health() -> Dict[str, Any]:
    # 没有专门的 /api/health 时,退到 /api/adu/projects(只读,稳定)。
    return verify_curl("http://127.0.0.1:8000/api/adu/projects", method="GET", timeout=6)


# ─── 5. project exists ───────────────────────────────────────────
def verify_project_exists(project_id: str) -> Dict[str, Any]:
    if _registry is None:
        return _wrap([_check(f"project {project_id}", False, "registry_unavailable")])
    pid = _registry.normalize_project_id(project_id)
    if not pid:
        return _wrap([_check(f"project {project_id}", False, "unknown_project")])
    proj = _registry.get_project(pid)
    if not proj:
        return _wrap([_check(f"project {pid}", False, "registry_lookup_failed")])
    exists = bool(proj.get("exists"))
    return _wrap(
        [_check(f"project {pid}", exists, "exists" if exists else "path_missing")],
        extra={"project_id": pid, "path": proj.get("path"), "name": proj.get("name")},
    )


# ─── 6. xcodebuild command(只返回推荐命令,不执行) ───────────────
def verify_xcodebuild_command(command_text: str = "") -> Dict[str, Any]:
    recommended = (
        'xcodebuild -workspace /Users/a12345/Desktop/GPTsora/GPTsora.xcworkspace '
        '-scheme "GPT Solara" -configuration Debug '
        "-destination 'generic/platform=iOS' "
        "CODE_SIGNING_ALLOWED=NO build"
    )
    rough_ok = bool(command_text) and ("xcodebuild" in command_text) and ("scheme" in command_text)
    return _wrap(
        [
            _check("xcodebuild_command_hint", True, "见 details.recommended_command;此层不执行",
                   details={
                       "recommended_command": recommended,
                       "user_command_passed_in": bool(command_text),
                       "user_command_looks_valid": rough_ok,
                   }),
        ],
    )
