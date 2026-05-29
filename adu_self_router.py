"""ChatAGI Adu — self router(/api/adu/self/*)。

挂载方式(server_session.py):

    from adu_self_router import router as adu_self_router
    app.include_router(adu_self_router)

接口清单:
    GET  /api/adu/self/identity                 阿杜身份 + 长期目标 + 硬约束
    GET  /api/adu/self/capabilities             能力地图(全量 + 按 status 分桶)
    GET  /api/adu/self/evolution-memory         最近进化记忆(默认 limit=20)
    POST /api/adu/self/evolution-memory         追加一条进化记忆
    POST /api/adu/self/upgrade/plan             生成升级计划(plan_only,**不执行**)

安全约束(所有 endpoint 共享):
- 不读 .env / 不返回任何 key / token / credential
- 不调 Codex / 不写代码 / 不 git commit / 不部署
- 不递归扫描系统盘(只在授权工作区,由 verifier / file_search 内部限制)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import adu_self_identity as _identity
import adu_capability_map as _capmap
import adu_evolution_memory as _memory
import adu_self_loop as _loop
import adu_self_sessions as _sessions


router = APIRouter(prefix="/api/adu/self", tags=["adu-self"])


# ─── /identity ───────────────────────────────────────────────────
@router.get("/identity")
def get_identity() -> Dict[str, Any]:
    return _identity.get_identity()


# ─── /capabilities ────────────────────────────────────────────────
@router.get("/capabilities")
def get_capabilities() -> Dict[str, Any]:
    return {
        "ok": True,
        "capabilities": _capmap.list_capabilities(),
        "by_status": _capmap.summarize_by_status(),
    }


# ─── /evolution-memory ───────────────────────────────────────────
class EvolutionMemoryEvent(BaseModel):
    source: Optional[str] = Field(default="manual", max_length=64)
    problem: str = Field(default="", max_length=4000)
    root_cause: str = Field(default="", max_length=4000)
    fix: str = Field(default="", max_length=4000)
    verification: str = Field(default="", max_length=4000)
    cost: str = Field(default="", max_length=200)
    risk: str = Field(default="", max_length=200)
    lesson: str = Field(default="", max_length=4000)
    tags: List[str] = Field(default_factory=list)


@router.get("/evolution-memory")
def get_evolution_memory(limit: int = 20) -> Dict[str, Any]:
    if limit <= 0 or limit > 200:
        raise HTTPException(status_code=422, detail={"ok": False, "error": "limit_out_of_range"})
    events = _memory.list_recent(limit=limit)
    return {
        "ok": True,
        "count_total": _memory.memory_count(),
        "returned": len(events),
        "events": events,
        "summary": _memory.summarize_recent(limit=min(8, limit)),
        "file": _memory.memory_path(),
    }


@router.post("/evolution-memory")
def add_evolution_memory(req: EvolutionMemoryEvent) -> Dict[str, Any]:
    # 至少要有 problem 或 lesson,否则没有信息量
    if not (req.problem.strip() or req.lesson.strip()):
        raise HTTPException(status_code=422, detail={
            "ok": False, "error": "need_problem_or_lesson",
        })
    saved = _memory.add_memory(req.model_dump())
    return {"ok": True, "saved": saved, "count_total": _memory.memory_count()}


# ─── /upgrade/plan ───────────────────────────────────────────────
class UpgradePlanRequest(BaseModel):
    goal: str = Field(default="", max_length=2000)
    surface: str = Field(default="home_chat", max_length=64)
    mode: str = Field(default="plan_only", max_length=32)
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)
    system_state: Dict[str, Any] = Field(default_factory=dict)


@router.post("/upgrade/plan")
def upgrade_plan(req: UpgradePlanRequest) -> Dict[str, Any]:
    result = _loop.run_loop(req.model_dump())
    # run_loop 自带 ok 字段;execute mode 会被强制拒绝并 ok=False。
    return result


# ─── V2 self-upgrade sessions(闭环状态机)─────────────────────────
# 阶段:planned → codex_task_created → (running) → verified/failed → memory_written
# /sessions          创建一个新 session,同时生成计划
# /sessions/{id}     读取当前状态
# /sessions/{id}/select-upgrade   用户选了某条 recommended_upgrade → 把它的 codex_prompt 钉到 session
# /sessions/{id}/codex-result     Codex 跑完后回传摘要(永不发 stdout/stderr) → 生成 verification + 写 memory

class CreateSessionRequest(BaseModel):
    goal: str = Field(default="", max_length=2000)
    surface: str = Field(default="home_chat", max_length=64)
    mode: str = Field(default="plan_only", max_length=32)
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)
    system_state: Dict[str, Any] = Field(default_factory=dict)


@router.post("/sessions")
def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    # 仍走老的 plan 生成器(adu_self_loop → SelfUpgradePlanner),保持口径一致。
    plan = _loop.run_loop(req.model_dump())
    if not plan.get("ok"):
        # plan 生成本身失败(例如 mode=execute_with_confirmation 被拒)
        return {"ok": False, "error": plan.get("error", "plan_failed"), "plan": plan}
    sess = _sessions.create_session(
        goal=req.goal, surface=req.surface, mode=req.mode, plan=plan,
    )
    # 把 plan 的所有字段平铺出去 + 顶层加 session_id,这样 iOS 端只需要一个 Codable 结构
    out = dict(plan)
    out["session_id"] = sess["id"]
    out["session"] = sess
    return out


@router.get("/sessions/{session_id}")
def read_session(session_id: str) -> Dict[str, Any]:
    sess = _sessions.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail={"ok": False, "error": "session_not_found"})
    return {"ok": True, "session": sess}


class SelectUpgradeRequest(BaseModel):
    upgrade_id: str = Field(..., min_length=1, max_length=64)


@router.post("/sessions/{session_id}/select-upgrade")
def select_upgrade(session_id: str, req: SelectUpgradeRequest) -> Dict[str, Any]:
    sess = _sessions.select_upgrade(session_id, req.upgrade_id)
    if not sess:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "session_or_upgrade_not_found",
            "session_id": session_id, "upgrade_id": req.upgrade_id,
        })
    return {
        "ok": True,
        "session_id": sess["id"],
        "stage": sess.get("stage"),
        "selected_upgrade_id": sess.get("selected_upgrade_id"),
        "current_codex_prompt": sess.get("current_codex_prompt"),
        "session": sess,
    }


class UpdateStageRequest(BaseModel):
    stage: str = Field(..., min_length=1, max_length=32)
    message: Optional[str] = Field(default=None, max_length=400)


@router.post("/sessions/{session_id}/update-stage")
def update_stage(session_id: str, req: UpdateStageRequest) -> Dict[str, Any]:
    """V2.1:中间态上报(主要给 iOS 点 Run Codex 后报 running)。
    终态(memory_written / expired / failed)不允许回退,返回 409。"""
    sess, err = _sessions.update_stage(session_id, req.stage, req.message)
    if err == "invalid_stage":
        raise HTTPException(status_code=422, detail={
            "ok": False, "error": err,
            "allowed": sorted(_sessions.VALID_STAGES),
        })
    if err == "session_not_found":
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": err, "session_id": session_id,
        })
    if err == "stage_locked":
        raise HTTPException(status_code=409, detail={
            "ok": False, "error": err,
            "current_stage": (sess or {}).get("stage"),
            "session_id": session_id,
        })
    return {
        "ok": True,
        "session_id": session_id,
        "stage": (sess or {}).get("stage"),
        "session": sess,
    }


class CodexResultRequest(BaseModel):
    ok: bool
    exit_code: Optional[int] = None
    duration: Optional[float] = None
    summary: Optional[str] = Field(default=None, max_length=4000)
    needs_user_confirmation: bool = False
    blocked_terms: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None, max_length=400)
    http_status: Optional[int] = None
    # 兼容:即使 iOS 把短摘要塞这里,后端 sanitize 仍丢弃
    stdout_excerpt: Optional[str] = Field(default=None, max_length=400)
    stderr_excerpt: Optional[str] = Field(default=None, max_length=400)


@router.post("/sessions/{session_id}/codex-result")
def record_codex_result(session_id: str, req: CodexResultRequest) -> Dict[str, Any]:
    sess = _sessions.record_codex_result(session_id, req.model_dump())
    if not sess:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "session_not_found", "session_id": session_id,
        })
    return {
        "ok": True,
        "session_id": sess["id"],
        "stage": sess.get("stage"),
        "verification_result": sess.get("verification_result"),
        "memory_write_result": sess.get("memory_write_result"),
        "next_recommendation": sess.get("next_recommendation"),
        "session": sess,
    }
