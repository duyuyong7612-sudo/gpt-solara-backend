from __future__ import annotations

import asyncio
import json
import queue
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .executor import LocalComputerAgent
from .events import translate_event
from .registry import get_registry

router = APIRouter(prefix="/computer-agent", tags=["computer-agent"])
_agent = LocalComputerAgent()


# =========================================
# Task state / in-memory store
# =========================================

@dataclass
class AgentTask:
    task_id: str
    status: str
    goal: str
    target: str
    task_type: str = ""
    source: str = "local"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _TaskStore:
    def __init__(self) -> None:
        self.tasks: Dict[str, AgentTask] = {}
        self.queues: Dict[str, "queue.Queue[str]"] = {}

    def create(self, goal: str, target: str) -> AgentTask:
        task_id = uuid.uuid4().hex[:12]
        task = AgentTask(
            task_id=task_id,
            status="queued",
            goal=goal,
            target=target,
        )
        self.tasks[task_id] = task
        self.queues[task_id] = queue.Queue()
        return task

    def get(self, task_id: str) -> Optional[AgentTask]:
        return self.tasks.get(task_id)

    def q(self, task_id: str) -> "queue.Queue[str]":
        if task_id not in self.queues:
            self.queues[task_id] = queue.Queue()
        return self.queues[task_id]

    def update(self, task_id: str, **kwargs: Any) -> None:
        task = self.tasks.get(task_id)
        if not task:
            return
        for k, v in kwargs.items():
            if hasattr(task, k):
                setattr(task, k, v)
        task.updated_at = time.time()


STORE = _TaskStore()


def _emit(task_id: str, event_name: str, payload: Dict[str, Any]) -> None:
    msg = {"event": event_name, "payload": payload}
    lines = translate_event(msg)
    for line in lines:
        STORE.q(task_id).put(line)


def _sse_iter(task_id: str) -> Iterator[bytes]:
    q = STORE.q(task_id)
    yield f"event: meta\ndata: {json.dumps({'task_id': task_id}, ensure_ascii=False)}\n\n".encode("utf-8")
    while True:
        try:
            line = q.get(timeout=60)
            yield line.encode("utf-8")
            task = STORE.get(task_id)
            if task and task.status in ("done", "error", "aborted"):
                return
        except queue.Empty:
            yield b": keepalive\n\n"


# =========================================
# Request models
# =========================================

class ExecuteBody(BaseModel):
    goal: str = Field(..., min_length=1)
    target: str = Field("", min_length=0)
    async_mode: bool = True


class AbortBody(BaseModel):
    reason: str = "aborted_by_user"


# =========================================
# Internal runner
# =========================================

async def _run_task(task_id: str, goal: str, target: str) -> None:
    task = STORE.get(task_id)
    if not task:
        return

    normalized_target = get_registry().normalize_target(target)
    task_type = _agent.classify_task(goal, normalized_target)

    STORE.update(task_id, status="running", task_type=task_type)
    _emit(task_id, "action_started", {
        "action": task_type,
        "target": normalized_target,
    })

    try:
        result = await _agent.execute(goal, normalized_target)
        ok = bool(result.get("ok", False))

        _emit(task_id, "action_result", {
            "action": task_type,
            "target": normalized_target,
            "ok": ok,
            "summary": result.get("stage") or "",
            "raw": result,
            "count": result.get("count"),
            "line_count": result.get("line_count"),
        })

        if ok:
            STORE.update(task_id, status="done", result=result)
            _emit(task_id, "done", {
                "ok": True,
                "summary": "执行完成",
                "result": result,
            })
        else:
            err = str(result.get("error") or "execute_failed")
            STORE.update(task_id, status="error", result=result, error=err)
            _emit(task_id, "error", {
                "message": err,
            })

    except Exception as e:
        STORE.update(task_id, status="error", error=str(e))
        _emit(task_id, "error", {
            "message": str(e),
        })


# =========================================
# API routes
# =========================================

@router.get("/health")
async def health() -> Dict[str, Any]:
    reg = get_registry()
    return {
        "ok": True,
        "source": "local",
        "platform": reg.data.get("system", {}).get("platform"),
        "apps_count": len(reg.data.get("apps") or {}),
        "workspace_aliases_count": len(reg.data.get("workspace_aliases") or {}),
        "tasks_count": len(STORE.tasks),
    }


@router.post("/probe")
async def probe() -> Dict[str, Any]:
    reg = get_registry()
    data = reg.probe_and_persist()
    return {
        "ok": True,
        "source": "local",
        "registry": data,
    }


@router.get("/registry")
async def registry() -> Dict[str, Any]:
    reg = get_registry()
    return {
        "ok": True,
        "source": "local",
        "registry": reg.data,
    }


@router.post("/execute")
async def execute(body: ExecuteBody) -> Dict[str, Any]:
    goal = body.goal.strip()
    target = body.target.strip()

    task = STORE.create(goal, target)

    if body.async_mode:
        asyncio.create_task(_run_task(task.task_id, goal, target))
        return {
            "ok": True,
            "source": "local",
            "task_id": task.task_id,
            "status": "queued",
            "events_url": f"/computer-agent/events/{task.task_id}",
            "task_url": f"/computer-agent/task/{task.task_id}",
        }

    await _run_task(task.task_id, goal, target)
    final_task = STORE.get(task.task_id)
    return {
        "ok": bool(final_task and final_task.status == "done"),
        "source": "local",
        "task": final_task.to_dict() if final_task else None,
    }


@router.get("/task/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    task = STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task_not_found")
    return {
        "ok": True,
        "source": "local",
        "task": task.to_dict(),
    }


@router.get("/tasks")
async def list_tasks(limit: int = Query(default=20, ge=1, le=100)) -> Dict[str, Any]:
    items = sorted(STORE.tasks.values(), key=lambda x: x.created_at, reverse=True)[:limit]
    return {
        "ok": True,
        "source": "local",
        "tasks": [t.to_dict() for t in items],
    }


@router.get("/events/{task_id}")
async def events(task_id: str) -> StreamingResponse:
    task = STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task_not_found")
    return StreamingResponse(
        _sse_iter(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/abort/{task_id}")
async def abort(task_id: str, body: Optional[AbortBody] = None) -> Dict[str, Any]:
    task = STORE.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task_not_found")

    if task.status in ("done", "error", "aborted"):
        return {
            "ok": True,
            "source": "local",
            "task_id": task_id,
            "status": task.status,
        }

    STORE.update(task_id, status="aborted", error=(body.reason if body else "aborted"))
    _emit(task_id, "error", {
        "message": body.reason if body else "aborted_by_user",
    })

    return {
        "ok": True,
        "source": "local",
        "task_id": task_id,
        "status": "aborted",
    }


@router.get("/file")
async def get_file(path: str = Query(...)) -> FileResponse:
    resolved = Path(path).expanduser().resolve()

    allowed_prefixes = [
        Path.home(),
        Path("/tmp"),
    ]

    is_allowed = any(str(resolved).startswith(str(prefix.resolve())) for prefix in allowed_prefixes)
    if not is_allowed:
        raise HTTPException(status_code=403, detail="path_not_allowed")

    if ".." in str(path):
        raise HTTPException(status_code=403, detail="path_traversal_blocked")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="file_not_found")

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="not_a_file")

    return FileResponse(
        path=str(resolved),
        filename=resolved.name,
        media_type="application/octet-stream",
    )
