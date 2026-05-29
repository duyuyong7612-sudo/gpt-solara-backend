"""ChatAGI Adu file search — 只在授权工作区内按文件名搜索。

POST /api/adu/files/search
  body: { "query": "server_session.py",
          "project_id": "backend",   # optional;不填则在所有授权工程内搜
          "max_results": 50,          # optional
          "case_sensitive": false }   # optional
  resp: { ok, project_id, scanned_projects, query, hit_count,
          hits: [ { project_id, project_name, name, relative_path,
                    absolute_path, size_bytes, modified } ] }

铁律(写在源头,后面任何 Codex/Agent 都不要松绑):
  - 只在 adu_project_registry 列出的工程目录里搜。绝不递归 /、$HOME、/Users。
  - 自动跳过敏感文件 / 目录(env / key / cred / git / node_modules / venv / build / Pods …)。
  - 不读文件内容,只列文件名与大小;**绝不返回 .env 之类文件的内容**。
  - 不跟随符号链接出工程根(防 symlink 逃逸到家目录或系统盘)。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
    _FASTAPI_OK = True
except Exception:  # pragma: no cover
    _FASTAPI_OK = False

try:
    import adu_project_registry as _registry  # type: ignore
except Exception:  # pragma: no cover
    _registry = None  # type: ignore


# ─── 安全黑名单 ────────────────────────────────────────────────────
# 目录名(任意层级里出现就整子树跳过)
SKIP_DIR_NAMES = {
    ".git", ".hg", ".svn",
    "node_modules", "bower_components",
    ".venv", "venv", "env", ".env.d",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "Pods", "Carthage", "build", "Build", "dist", "out", ".next", ".nuxt",
    "DerivedData", "ModuleCache.noindex",
    "xcuserdata",
    ".idea", ".vscode",
    ".terraform",
    "coverage", ".nyc_output",
    "credentials", "secrets",
}

# 文件名前缀 / 后缀 / 关键词:命中即跳过(也不会出现在 hits 里)
SKIP_FILE_PREFIXES = (
    ".env",      # .env, .env.local, .env.production…
    ".secret",
    ".credential",
    ".pgpass",
    ".npmrc",
    ".netrc",
)
SKIP_FILE_SUFFIXES = (
    ".key", ".pem", ".p12", ".pfx", ".keystore", ".jks",
    ".gpg", ".asc",
    ".cer", ".crt",
    ".swp", ".swo",
    ".log",                  # 别把巨大日志列出来
)
SKIP_FILE_KEYWORDS = (       # case-insensitive substring
    "id_rsa", "id_ed25519", "id_dsa",
    "credentials.json", "secrets.json", "secret.json",
    "service_account.json", "service-account.json",
    "private_key", "privatekey", "client_secret",
)

# 单次最多遍历的目录条目数,防极端工程超时
MAX_SCAN_ENTRIES = 8000
# 单次最多放进结果的命中数(在 max_results 之上再封顶)
HARD_HIT_CAP = 200
# 单文件名最大长度(防止异常输入)
MAX_QUERY_LEN = 200


def _is_sensitive_filename(name: str) -> bool:
    low = name.lower()
    if low.startswith(SKIP_FILE_PREFIXES):
        return True
    if low.endswith(SKIP_FILE_SUFFIXES):
        return True
    for kw in SKIP_FILE_KEYWORDS:
        if kw in low:
            return True
    return False


def _walk_project(root: Path, query: str, case_sensitive: bool) -> List[Dict[str, Any]]:
    """单工程内文件名搜索。返回 hits(已去敏)。"""
    hits: List[Dict[str, Any]] = []
    needle = query if case_sensitive else query.lower()
    scanned = 0
    root = root.resolve()

    # followlinks=False 防 symlink 出根
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # 跳过敏感目录(原地裁 dirnames 防进入)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")]

        # 防 symlink 逃出 root
        try:
            real_here = Path(dirpath).resolve()
            if not (real_here == root or root in real_here.parents):
                continue
        except Exception:
            continue

        for fn in filenames:
            scanned += 1
            if scanned > MAX_SCAN_ENTRIES:
                return hits
            if _is_sensitive_filename(fn):
                continue
            hay = fn if case_sensitive else fn.lower()
            if needle not in hay:
                continue
            full = Path(dirpath) / fn
            try:
                st = full.stat()
            except OSError:
                continue
            # 二次防 symlink:若解析后跳出 root,丢弃
            try:
                real_full = full.resolve()
                if root not in real_full.parents and real_full != root:
                    continue
            except Exception:
                continue
            rel = str(full.relative_to(root))
            hits.append({
                "name": fn,
                "relative_path": rel,
                "absolute_path": str(full),
                "size_bytes": int(st.st_size),
                "modified": int(st.st_mtime),
            })
            if len(hits) >= HARD_HIT_CAP:
                return hits
    return hits


def _search_one(project_id: str, query: str, case_sensitive: bool) -> Dict[str, Any]:
    if _registry is None:
        return {"project_id": project_id, "hits": [], "skipped": "registry_unavailable"}
    proj = _registry.get_project(project_id)
    if not proj:
        return {"project_id": project_id, "hits": [], "skipped": "unknown_project"}
    if not proj.get("exists"):
        return {"project_id": project_id, "hits": [], "skipped": "path_missing"}
    root = Path(proj["path"])
    started = time.time()
    raw = _walk_project(root, query, case_sensitive)
    # 把工程标签塞回每条 hit,前端可以直接展示
    for h in raw:
        h["project_id"] = project_id
        h["project_name"] = proj["name"]
    return {
        "project_id": project_id,
        "project_name": proj["name"],
        "absolute_root": str(root),
        "hits": raw,
        "scanned_ms": int((time.time() - started) * 1000),
    }


# ─── HTTP ─────────────────────────────────────────────────────────
if _FASTAPI_OK:
    router = APIRouter(prefix="/api/adu", tags=["adu-files"])

    class FileSearchRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=MAX_QUERY_LEN)
        project_id: Optional[str] = Field(default=None)
        max_results: int = Field(default=50, ge=1, le=HARD_HIT_CAP)
        case_sensitive: bool = Field(default=False)

    @router.post("/files/search")
    def files_search(req: FileSearchRequest) -> Dict[str, Any]:
        q = req.query.strip()
        if not q:
            raise HTTPException(status_code=422, detail={"ok": False, "error": "empty_query"})

        # 决定搜哪些工程
        scanned_projects: List[str] = []
        per_project: List[Dict[str, Any]] = []

        if req.project_id and _registry is not None:
            pid = _registry.normalize_project_id(req.project_id)
            if not pid:
                raise HTTPException(status_code=422, detail={
                    "ok": False, "error": "unknown_project", "project_id": req.project_id,
                })
            scanned_projects = [pid]
            per_project.append(_search_one(pid, q, req.case_sensitive))
        else:
            # 没指定工程 → 在所有"exists=true"的授权工程里搜
            if _registry is None:
                raise HTTPException(status_code=503, detail={"ok": False, "error": "registry_unavailable"})
            for p in _registry.list_projects():
                if not p["exists"]:
                    continue
                scanned_projects.append(p["id"])
                per_project.append(_search_one(p["id"], q, req.case_sensitive))

        # 合并 hits 并按工程内 hits 的最近修改时间排序,封顶 max_results
        all_hits: List[Dict[str, Any]] = []
        for ps in per_project:
            all_hits.extend(ps.get("hits", []))
        all_hits.sort(key=lambda h: h.get("modified", 0), reverse=True)
        capped = all_hits[: req.max_results]

        return {
            "ok": True,
            "query": q,
            "case_sensitive": req.case_sensitive,
            "project_id": req.project_id if req.project_id else None,
            "scanned_projects": scanned_projects,
            "hit_count": len(all_hits),
            "returned": len(capped),
            "hits": capped,
            "per_project": [
                {
                    "project_id": ps["project_id"],
                    "project_name": ps.get("project_name"),
                    "hit_count": len(ps.get("hits", [])),
                    "scanned_ms": ps.get("scanned_ms"),
                    "skipped": ps.get("skipped"),
                }
                for ps in per_project
            ],
        }
else:  # pragma: no cover
    router = None  # type: ignore
