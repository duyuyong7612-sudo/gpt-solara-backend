"""ChatAGI Adu — project registry + GET /api/adu/projects.

单一来源:planner / codex executor / iOS App 都从这里走,不要在多处硬编码 path。

依赖:仅 stdlib + fastapi/pydantic。挂载方式参考 adu_planner_router.py。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter
    _FASTAPI_OK = True
except Exception:  # pragma: no cover
    _FASTAPI_OK = False


# ─── Registry ────────────────────────────────────────────────────────
# 注册表是顺序敏感:keyword 命中数相同时按这里的顺序取第一个。
# 关键词大小写不敏感(infer_from_text 内做 lower)。
_PROJECTS: List[Dict[str, Any]] = [
    {
        "id": "gptsora",
        "name": "GPTsora",
        "path": "/Users/a12345/Desktop/GPTsora",
        "kinds": ["ios", "swift", "swiftui", "xcode"],
        "keywords": [
            "gptsora", "gpt sora", "sora",
            "homechatview", "home chat view", "homechat",
            "iphoneagent", "iphone agent",
            "swift", "swiftui", "xcode", "xcodeproj", "xcworkspace",
            "ios", "ipados", "ipad",
            "computercontrolhomeview", "engineeringagentpanel",
            "codexagentclient", "soravideoapp", "auth root view",
            "波纹键", "电脑控制页", "自动化编程面板",
            "前端", "ios 前端", "ios 端", "app", "app 端",
        ],
    },
    {
        "id": "backend",
        "name": "Backend",
        "path": "/Users/a12345/Desktop/backend",
        "kinds": ["python", "fastapi", "backend"],
        "keywords": [
            "backend", "后端", "服务端",
            "server_session", "server session", "session 后端",
            "fastapi", "uvicorn",
            "/api/", "api 路由", "router", "include_router",
            "adu_planner", "adu_codex", "adu_orchestrator",
            "adu_project_registry",
            "local-agent", "local_agent", "4317", "8000",
            "computer/action", "brain/computer",
            "py_compile", "python", ".py",
        ],
    },
    {
        "id": "little_beijing",
        "name": "Little Beijing",
        "path": "/Users/a12345/Desktop/little-beijing-edge-box",
        "kinds": ["ordering", "edge_box"],
        "keywords": [
            "little beijing", "little-beijing", "littlebeijing",
            "小北京", "北京餐厅", "餐厅", "ordering",
            "edge box", "edge-box", "edge_box",
            "menu.json", "rules.json", "store.json",
            "realtime/sdp", "openai realtime", "gpt-realtime",
            "餐馆", "点餐", "阿杜服务员", "服务员",
        ],
    },
    {
        "id": "chatagi_site",
        "name": "ChatAGI Site",
        "path": "/Users/a12345/Desktop/chatagi-site",
        "kinds": ["website", "static"],
        "keywords": [
            "chatagi-site", "chatagi site", "chatagi.site",
            "网站", "landing", "portal", "static site",
            "render static", "marketing site",
        ],
    },
]


def _exists(path: str) -> bool:
    try:
        return Path(path).is_dir()
    except Exception:
        return False


def list_projects() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in _PROJECTS:
        out.append({
            "id": p["id"],
            "name": p["name"],
            "path": p["path"],
            "kinds": list(p["kinds"]),
            "exists": _exists(p["path"]),
        })
    return out


def get_project(project_id: Optional[str]) -> Optional[Dict[str, Any]]:
    pid = (project_id or "").strip().lower()
    if not pid:
        return None
    for p in _PROJECTS:
        if p["id"] == pid:
            return {
                "id": p["id"],
                "name": p["name"],
                "path": p["path"],
                "kinds": list(p["kinds"]),
                "exists": _exists(p["path"]),
            }
    return None


def resolve_path(project_id: Optional[str]) -> Optional[str]:
    """project_id → 绝对路径。未知 / 空 → None。"""
    p = get_project(project_id)
    return p["path"] if p else None


def infer_from_text(text: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """纯关键词推断 project_id。供 planner 的兜底使用 —— LLM 没填时由这里补,
    LLM 填了的优先用 LLM 的。"""
    if not isinstance(text, str) or not text.strip():
        return default
    low = text.lower()
    best_id: Optional[str] = None
    best_hits: int = 0
    for p in _PROJECTS:
        hits = 0
        for kw in p["keywords"]:
            k = kw.lower()
            if k in low or kw in text:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_id = p["id"]
    return best_id or default


def normalize_project_id(value: Optional[str]) -> Optional[str]:
    """把 planner / iOS 传过来的 project_id 规整为 registry 内的标准 id。
    支持简单别名:'GPTsora'→'gptsora','Little Beijing'→'little_beijing' 等。"""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    # 直接命中
    for p in _PROJECTS:
        if p["id"] == v.lower():
            return p["id"]
    # 用 name / display 字符串匹配
    low = v.lower().replace("-", "").replace("_", "").replace(" ", "")
    for p in _PROJECTS:
        if p["id"].replace("_", "") == low:
            return p["id"]
        if p["name"].lower().replace(" ", "") == low:
            return p["id"]
    return None


# ─── HTTP ────────────────────────────────────────────────────────────
if _FASTAPI_OK:
    router = APIRouter(prefix="/api/adu", tags=["adu-projects"])

    @router.get("/projects")
    def get_projects():
        return {"ok": True, "projects": list_projects()}
else:  # pragma: no cover
    router = None  # type: ignore
