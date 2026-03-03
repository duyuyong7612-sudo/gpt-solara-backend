# -*- coding: utf-8 -*-
"""
agent_loop_router.py — ChatAGI 高级编程版：工程内循环（B方案·可跑工程版 v1）

功能：
- POST /agent/run              启动一次“Plan→Patch→Test→(Retry)→Done”
- GET  /agent/events/{run_id}  SSE 推送日志/阶段/计划/测试/结果
- GET  /agent/result/{run_id}  查询最终状态
- GET  /agent/bundle/{run_id}.zip  下载补丁包（变更文件 + diff + report）

设计原则：
1) 后端执行（前端只展示/触发），实现“无人介入也能跑工程”
2) 受控工具链：命令白名单 + 超时 + 工作区隔离（/tmp）
3) 成本可控：最多重试 N 次，默认 dry_run=1（不污染线上代码）
4) 模型兼容：OpenAI 优先走 Responses API（兼容 gpt-5.x），失败再退回 Chat Completions；
   Anthropic 走 /v1/messages（非流式）。

⚠️ 重要说明：
- 此实现默认把“当前后端项目目录”复制到 /tmp/<run_id>/repo 再改。
- 测试命令默认使用 `python -m compileall -q .`（不依赖 pytest）。
  你可以通过 env 或请求参数覆盖（例如 pytest -q / npm test）。
"""

from __future__ import annotations

import os, re, json, time, uuid, shutil, queue, threading, hashlib, zipfile, subprocess, difflib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple

import requests
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

router = APIRouter(prefix="/agent", tags=["agent"])

# ----------------------------
# Config
# ----------------------------

BASE_DIR = Path(__file__).parent.resolve()
WORK_ROOT = Path(os.getenv("AGENT_WORK_ROOT") or "/tmp/chatagi_agent_runs").resolve()
WORK_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = (os.getenv("AGENT_MODEL") or os.getenv("CODER_MODEL") or "gpt-4o-mini").strip()
DEFAULT_PROVIDER = (os.getenv("AGENT_PROVIDER") or "openai").strip().lower()  # openai|anthropic

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("SORA_API_KEY") or "").strip()
ANTHROPIC_API_KEY = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
ANTHROPIC_BASE_URL = (os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
ANTHROPIC_VERSION = (os.getenv("ANTHROPIC_VERSION") or "2023-06-01").strip()

AGENT_MAX_TRIES = max(1, min(int(os.getenv("AGENT_MAX_TRIES") or "2"), 5))
AGENT_TEST_CMD_DEFAULT = (os.getenv("AGENT_TEST_CMD") or "python -m compileall -q .").strip()

AGENT_CMD_TIMEOUT_SEC = int(os.getenv("AGENT_CMD_TIMEOUT_SEC") or "240")
AGENT_MODEL_TIMEOUT_SEC = int(os.getenv("AGENT_MODEL_TIMEOUT_SEC") or "120")

# Allowed commands (regex list). Keep strict.
_ALLOWED_CMD_PATTERNS = [
    r"^python\s+-m\s+compileall(\s|$)",
    r"^pytest(\s|$)",
    r"^python(\s+-m\s+pytest)?(\s|$)",
    r"^npm\s+test(\s|$)",
    r"^pnpm\s+test(\s|$)",
    r"^yarn\s+test(\s|$)",
    r"^flutter\s+test(\s|$)",
    r"^xcodebuild(\s|$)",
]
_EXTRA = os.getenv("AGENT_ALLOWED_CMDS_REGEX") or ""
if _EXTRA.strip():
    for part in _EXTRA.split(";"):
        part = part.strip()
        if part:
            _ALLOWED_CMD_PATTERNS.append(part)

_ALLOWED_CMD_RE = re.compile("|".join(f"(?:{p})" for p in _ALLOWED_CMD_PATTERNS), re.I)

def _now() -> float:
    return time.time()

def _short(s: str, n: int = 3000) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "…")

def _safe_run_cmd(cmd: str, cwd: Path) -> Tuple[int, str]:
    cmd = (cmd or "").strip()
    if not cmd:
        return 2, "empty cmd"
    if not _ALLOWED_CMD_RE.search(cmd):
        return 2, f"blocked cmd: {cmd}"
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=AGENT_CMD_TIMEOUT_SEC,
            env=dict(os.environ),
        )
        out = (p.stdout or "") + ("\n" + (p.stderr or "") if p.stderr else "")
        out = out[-20000:]  # tail
        return int(p.returncode), out
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {AGENT_CMD_TIMEOUT_SEC}s"
    except Exception as e:
        return 2, f"run error: {e}"

def _list_text_files(root: Path, max_files: int = 1200) -> List[str]:
    ex_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", ".dart_tool", ".idea",
               ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "downloads", "agent_runs"}
    ex_ext = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".avi", ".zip", ".pdf", ".exe", ".dylib", ".so"}
    res = []
    for p in root.rglob("*"):
        if len(res) >= max_files:
            break
        if not p.is_file():
            continue
        if any(part in ex_dirs for part in p.parts):
            continue
        if p.suffix.lower() in ex_ext:
            continue
        try:
            if p.stat().st_size > 600_000:
                continue
        except Exception:
            continue
        try:
            res.append(str(p.relative_to(root)))
        except Exception:
            continue
    return sorted(res)

def _read_file(root: Path, rel: str) -> str:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError("path escape")
    return p.read_text("utf-8", errors="ignore")

def _write_file(root: Path, rel: str, content: str) -> None:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError("path escape")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")

def _unified_diff(a: str, b: str, fromfile: str, tofile: str) -> str:
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    return "".join(difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile))

# ----------------------------
# SSE event store
# ----------------------------

@dataclass
class RunState:
    run_id: str
    status: str  # queued|planning|patching|testing|done|error
    created_at: float
    updated_at: float
    tries: int = 0
    dry_run: bool = True
    test_cmd: str = ""
    goal: str = ""
    model: str = ""
    provider: str = ""
    last_error: str = ""
    changed_files: List[str] = None
    diff: str = ""
    report: str = ""

class _RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[str, RunState] = {}
        self._queues: Dict[str, "queue.Queue[str]"] = {}

    def create(self, *, goal: str, dry_run: bool, test_cmd: str, model: str, provider: str) -> RunState:
        rid = uuid.uuid4().hex
        st = RunState(
            run_id=rid,
            status="queued",
            created_at=_now(),
            updated_at=_now(),
            tries=0,
            dry_run=bool(dry_run),
            test_cmd=test_cmd,
            goal=goal,
            model=model,
            provider=provider,
            changed_files=[],
            diff="",
            report="",
        )
        with self._lock:
            self._states[rid] = st
            self._queues[rid] = queue.Queue()
        return st

    def q(self, rid: str) -> "queue.Queue[str]":
        with self._lock:
            q = self._queues.get(rid)
            if q is None:
                q = queue.Queue()
                self._queues[rid] = q
            return q

    def get(self, rid: str) -> Optional[RunState]:
        with self._lock:
            return self._states.get(rid)

    def update(self, rid: str, **kwargs: Any) -> None:
        with self._lock:
            st = self._states.get(rid)
            if not st:
                return
            for k, v in kwargs.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            st.updated_at = _now()

STORE = _RunStore()

def _emit(rid: str, typ: str, data: Any) -> None:
    payload = json.dumps({"type": typ, "data": data, "ts": _now()}, ensure_ascii=False)
    STORE.q(rid).put(payload)

def _sse_iter(rid: str) -> Iterator[bytes]:
    q = STORE.q(rid)
    yield f"event: meta\ndata: {json.dumps({'run_id': rid}, ensure_ascii=False)}\n\n".encode("utf-8")
    while True:
        try:
            msg = q.get(timeout=60)
            yield f"event: message\ndata: {msg}\n\n".encode("utf-8")
            st = STORE.get(rid)
            if st and st.status in ("done", "error"):
                yield f"event: done\ndata: {json.dumps(asdict(st), ensure_ascii=False)}\n\n".encode("utf-8")
                return
        except queue.Empty:
            yield b": keepalive\n\n"

# ----------------------------
# LLM helpers
# ----------------------------

def _openai_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

def _responses_input(sys: str, user: str) -> List[Dict[str, Any]]:
    def _blk(role: str, text: str) -> Dict[str, Any]:
        # assistant history would be output_text; we only use system+user here
        return {"type": "input_text", "text": (text or "").strip()}
    return [
        {"role": "system", "content": [_blk("system", sys)]},
        {"role": "user", "content": [_blk("user", user)]},
    ]

def _responses_extract_output_text(resp_obj: Dict[str, Any]) -> str:
    try:
        out = resp_obj.get("output") or []
        parts: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content") or []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    parts.append(str(c.get("text") or ""))
        return "".join(parts)
    except Exception:
        return ""

def _openai_responses(sys: str, user: str, model: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    url = "https://api.openai.com/v1/responses"
    payload = {
        "model": model,
        "input": _responses_input(sys, user),
        "max_output_tokens": 2000,
        "truncation": "auto",
    }
    r = requests.post(url, headers=_openai_headers(), json=payload, timeout=AGENT_MODEL_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"openai_responses_error {r.status_code}: {_short(r.text, 400)}")
    obj = r.json() if r.text else {}
    return _responses_extract_output_text(obj) or ""

def _openai_chatcompletions(sys: str, user: str, model: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": 1800,
    }
    r = requests.post(url, headers=_openai_headers(), json=payload, timeout=AGENT_MODEL_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"openai_chatcompletions_error {r.status_code}: {_short(r.text, 400)}")
    j = r.json()
    return (((j.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""

def _anthropic_messages(sys: str, user: str, model: str) -> str:
    if not ANTHROPIC_API_KEY:
        return ""
    url = f"{ANTHROPIC_BASE_URL}/v1/messages"
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 2200,
        "temperature": 0.2,
        "system": sys,
        "messages": [{"role": "user", "content": user}],
    }
    r = requests.post(
        url,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        data=json.dumps(payload),
        timeout=AGENT_MODEL_TIMEOUT_SEC,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"anthropic_error {r.status_code}: {_short(r.text, 400)}")
    j = r.json()
    blocks = j.get("content") or []
    out = ""
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            out += str(b.get("text") or "")
    return out

def _llm(provider: str, model: str, sys: str, user: str) -> str:
    provider = (provider or "openai").strip().lower()
    model = (model or DEFAULT_MODEL).strip()
    if provider == "anthropic":
        return _anthropic_messages(sys=sys, user=user, model=model)
    # openai
    try:
        txt = _openai_responses(sys=sys, user=user, model=model)
        if txt.strip():
            return txt
    except Exception:
        pass
    return _openai_chatcompletions(sys=sys, user=user, model=model)

def _extract_json_best_effort(text: str) -> Any:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"[\{\[]", t)
    if m:
        t = t[m.start():]
    try:
        return json.loads(t)
    except Exception:
        last = max(t.rfind("}"), t.rfind("]"))
        if last != -1:
            try:
                return json.loads(t[: last + 1])
            except Exception:
                return None
    return None

def _plan_prompt(file_list: List[str], goal: str) -> str:
    files_preview = "\n".join(file_list[:240])
    if len(file_list) > 240:
        files_preview += "\n…(more)"
    return f"""你是一个专业的软件工程自动化智能体。你要在一个代码仓库中完成用户目标，并尽量通过运行测试验证。

用户目标：
{goal}

仓库文件清单（节选）：
{files_preview}

请输出严格 JSON（不要解释，不要 markdown），结构如下：
{{
  "analysis": "你对问题的简短分析（<=120字）",
  "plan": [
    {{"step": 1, "title": "做什么", "details": "怎么做（<=160字）"}},
    {{"step": 2, "title": "...", "details": "..."}}
  ],
  "test_cmd": "建议运行的测试命令（如 python -m compileall -q . / pytest -q / npm test 等；留空表示不跑）",
  "files_to_read": ["相对路径1","相对路径2"],
  "files_to_edit": ["相对路径A","相对路径B"]
}}
"""

def _patch_prompt(goal: str, plan_json: Dict[str, Any], file_blobs: Dict[str, str], test_cmd: str, last_test: str) -> str:
    blobs = []
    for p, c in file_blobs.items():
        cc = c if len(c) <= 9000 else (c[:9000] + "\n…(truncated)")
        blobs.append(f"### {p}\n{cc}")
    ctx = "\n\n".join(blobs)

    return f"""你是一个专业的软件工程自动化智能体。请根据目标与计划，直接给出要写回仓库的“文件修改结果”。

用户目标：
{goal}

计划（JSON）：
{json.dumps(plan_json, ensure_ascii=False)}

建议测试命令：
{test_cmd}

最近一次测试输出（若为空表示还未测试）：
{last_test}

当前文件内容（节选）：
{ctx}

请输出严格 JSON（不要解释、不要 markdown），结构：
{{
  "summary": "本次修改摘要（<=120字）",
  "changes": [
    {{"path": "相对路径", "content": "修改后的完整文件内容（UTF-8文本）"}},
    ...
  ]
}}
约束：
- 只修改真正需要的文件
- content 必须是“完整文件内容”，不是 diff
- 如果需要新增文件，也可提供新的 path+content
"""

# ----------------------------
# Runner
# ----------------------------

def _copy_repo_to_workdir(run_id: str) -> Path:
    src = BASE_DIR
    dst = (WORK_ROOT / run_id / "repo").resolve()
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ex_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", ".dart_tool", ".idea",
               ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "downloads", "agent_runs"}
    ex_ext = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".avi", ".zip", ".pdf", ".exe", ".dylib", ".so"}

    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part in ex_dirs for part in rel.parts):
            continue
        if p.is_dir():
            continue
        if p.suffix.lower() in ex_ext:
            continue
        try:
            if p.stat().st_size > 1_000_000:
                continue
        except Exception:
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(p.read_bytes())
        except Exception:
            pass
    return dst

def _compute_changes_diff(root: Path, changes: List[Dict[str, str]]) -> Tuple[List[str], str]:
    changed_files = []
    diffs = []
    for ch in changes:
        path = (ch.get("path") or "").strip()
        if not path:
            continue
        newc = ch.get("content") or ""
        oldc = ""
        fpath = root / path
        if fpath.exists():
            try:
                oldc = fpath.read_text("utf-8", errors="ignore")
            except Exception:
                oldc = ""
        changed_files.append(path)
        diffs.append(_unified_diff(oldc, newc, f"a/{path}", f"b/{path}"))
    return sorted(set(changed_files)), "\n".join(diffs).strip()

def _apply_changes(root: Path, changes: List[Dict[str, str]]) -> None:
    for ch in changes:
        path = (ch.get("path") or "").strip()
        if not path:
            continue
        content = ch.get("content") or ""
        _write_file(root, path, content)

def _make_bundle(run_id: str, root: Path, changed_files: List[str], diff_text: str, report: str) -> Path:
    outp = (WORK_ROOT / run_id / f"bundle_{run_id}.zip").resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        outp.unlink()
    with zipfile.ZipFile(outp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("REPORT.txt", report or "")
        z.writestr("DIFF.patch", diff_text or "")
        for rel in changed_files:
            try:
                z.write(str((root / rel).resolve()), arcname=f"files/{rel}")
            except Exception:
                pass
    return outp

def _run_worker(run_id: str, goal: str, dry_run: bool, test_cmd: str, model: str, provider: str) -> None:
    _emit(run_id, "log", f"[agent] start run_id={run_id} dry_run={dry_run} provider={provider} model={model}")
    repo = _copy_repo_to_workdir(run_id)
    file_list = _list_text_files(repo)

    # --- PLAN ---
    STORE.update(run_id, status="planning", tries=0)
    _emit(run_id, "stage", "planning")

    plan_obj = None
    try:
        plan_text = _llm(provider, model, "你是一个可靠的编程智能体。输出必须是严格 JSON。", _plan_prompt(file_list, goal))
        plan_obj = _extract_json_best_effort(plan_text)
    except Exception as e:
        plan_obj = None
        _emit(run_id, "log", f"[agent] plan error: {e}")

    if not isinstance(plan_obj, dict):
        plan_obj = {
            "analysis": "模型输出解析失败或不可用，先给出执行骨架。",
            "plan": [{"step": 1, "title": "确认目标与仓库结构", "details": "列出关键文件并决定修改点。"}],
            "test_cmd": test_cmd,
            "files_to_read": [],
            "files_to_edit": [],
        }

    suggested_test = str(plan_obj.get("test_cmd") or "").strip()
    if suggested_test:
        test_cmd = suggested_test

    _emit(run_id, "plan", plan_obj)

    last_test = ""
    final_changes: List[Dict[str, str]] = []
    final_diff = ""
    final_report = ""
    changed_files: List[str] = []

    for attempt in range(1, AGENT_MAX_TRIES + 1):
        STORE.update(run_id, status="patching", tries=attempt)
        _emit(run_id, "stage", f"patching:{attempt}")

        files_to_read = plan_obj.get("files_to_read") or []
        if not isinstance(files_to_read, list):
            files_to_read = []
        if not files_to_read:
            for cand in ("server_session.py", "memory_module.py", "requirements.txt", "pyproject.toml", "package.json"):
                if cand in file_list:
                    files_to_read.append(cand)

        blobs: Dict[str, str] = {}
        for fp in files_to_read[:10]:
            fp = str(fp)
            if fp in file_list:
                try:
                    blobs[fp] = _read_file(repo, fp)
                except Exception:
                    pass

        patch_obj = None
        try:
            patch_text = _llm(provider, model, "你是一个可靠的编程智能体。输出必须是严格 JSON。", _patch_prompt(goal, plan_obj, blobs, test_cmd, last_test))
            patch_obj = _extract_json_best_effort(patch_text)
        except Exception as e:
            patch_obj = None
            _emit(run_id, "log", f"[agent] patch error: {e}")

        if not isinstance(patch_obj, dict) or not isinstance(patch_obj.get("changes"), list):
            STORE.update(run_id, status="error", last_error="patch_failed")
            _emit(run_id, "log", "[agent] patch_failed: 没有得到可解析的 changes JSON")
            return

        final_changes = patch_obj.get("changes") or []
        changed_files, final_diff = _compute_changes_diff(repo, final_changes)
        _emit(run_id, "patch_summary", {"summary": patch_obj.get("summary") or "", "changed_files": changed_files})

        _apply_changes(repo, final_changes)

        # --- TEST ---
        STORE.update(run_id, status="testing", tries=attempt)
        _emit(run_id, "stage", f"testing:{attempt}")

        if test_cmd.strip():
            code, out = _safe_run_cmd(test_cmd, cwd=repo)
            last_test = f"exit={code}\n{out}"
            _emit(run_id, "test", {"cmd": test_cmd, "exit": code, "output_tail": out[-4000:]})
            if code == 0:
                final_report = f"SUCCESS\nattempt={attempt}\ncmd={test_cmd}\n\n{patch_obj.get('summary') or ''}\n"
                break
            else:
                if attempt < AGENT_MAX_TRIES:
                    _emit(run_id, "log", "[agent] tests failed, retrying with failure context.")
                    continue
                final_report = f"FAILED\nattempts={attempt}\ncmd={test_cmd}\n\nlast_test:\n{_short(last_test, 12000)}\n"
        else:
            final_report = f"NO_TEST\nattempt={attempt}\n\n{patch_obj.get('summary') or ''}\n"
            break

    bundle = _make_bundle(run_id, repo, changed_files, final_diff, final_report)

    STORE.update(
        run_id,
        status="done",
        changed_files=changed_files,
        diff=final_diff,
        report=final_report,
    )
    _emit(run_id, "result", {"changed_files": changed_files, "bundle": str(bundle), "report": _short(final_report, 2000)})

# ----------------------------
# API
# ----------------------------

def _extract_plan_from_request(req: Request, body: Dict[str, Any]) -> str:
    for k in ("plan", "tier", "套餐"):
        v = body.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # header
    h = (req.headers.get("X-ChatAGI-Plan") or req.headers.get("x-chatagi-plan") or "").strip()
    if h:
        return h
    return ""

@router.post("/run")
async def agent_run(req: Request) -> JSONResponse:
    try:
        body = await req.json()
    except Exception:
        body = {}

    goal = str(body.get("goal") or body.get("task") or body.get("prompt") or "").strip()
    if not goal:
        raise HTTPException(status_code=400, detail="missing goal")

    # Gate: only advanced/coder plan should use
    plan = _extract_plan_from_request(req, body).lower()
    if plan and plan not in ("coder", "advanced", "高级编程版", "高级编程", "code"):
        raise HTTPException(status_code=403, detail="agent_run only for coder plan")

    dry_run = bool(body.get("dry_run", True))
    test_cmd = str(body.get("test_cmd") or AGENT_TEST_CMD_DEFAULT).strip()

    model = str(body.get("model") or DEFAULT_MODEL).strip()
    provider = str(body.get("provider") or DEFAULT_PROVIDER).strip().lower()

    st = STORE.create(goal=goal, dry_run=dry_run, test_cmd=test_cmd, model=model, provider=provider)
    t = threading.Thread(target=_run_worker, args=(st.run_id, goal, dry_run, test_cmd, model, provider), daemon=True)
    t.start()

    base = str(req.base_url).rstrip("/")
    return JSONResponse(
        {
            "ok": True,
            "run_id": st.run_id,
            "events_url": f"{base}/agent/events/{st.run_id}",
            "result_url": f"{base}/agent/result/{st.run_id}",
            "bundle_url": f"{base}/agent/bundle/{st.run_id}.zip",
            "dry_run": dry_run,
        }
    )

@router.get("/events/{run_id}")
async def agent_events(run_id: str) -> StreamingResponse:
    st = STORE.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    return StreamingResponse(_sse_iter(run_id), media_type="text/event-stream")

@router.get("/result/{run_id}")
async def agent_result(req: Request, run_id: str) -> JSONResponse:
    st = STORE.get(run_id)
    if not st:
        raise HTTPException(status_code=404, detail="run_id not found")
    base = str(req.base_url).rstrip("/")
    return JSONResponse(
        {
            "ok": True,
            "state": asdict(st),
            "bundle_url": f"{base}/agent/bundle/{run_id}.zip",
        }
    )

@router.get("/bundle/{run_id}.zip")
async def agent_bundle(run_id: str) -> FileResponse:
    p = (WORK_ROOT / run_id / f"bundle_{run_id}.zip").resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="bundle not ready")
    return FileResponse(str(p), filename=p.name, media_type="application/zip")

