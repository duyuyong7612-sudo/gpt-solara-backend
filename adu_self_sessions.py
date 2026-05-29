"""ChatAGI Adu — self-upgrade session 状态机存储 + 业务逻辑。

闭环阶段:
    planned → codex_task_created → (running →) verified/failed → memory_written

- 持久化:/Users/a12345/Desktop/backend/data/adu_self_upgrade_sessions.json
- 单文件 JSON 数组,最多 100 条;原子写(临时文件 + os.replace)。
- 字段截断 4 KB / 列表上限 64,避免 stdout 等长字段污染。
- record_codex_result 内部直接调用 adu_evolution_memory.add_memory,
  让 evolution memory 自然带上 session_id / upgrade_id / stage 元数据。
- **绝不** 把完整 stdout / stderr / .env / 凭据写进 session 或 memory。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import adu_evolution_memory as _memory
except Exception:  # pragma: no cover
    _memory = None  # type: ignore


# ─── 文件 / 常量 ───────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "data"
_FILE = _DATA_DIR / "adu_self_upgrade_sessions.json"

_MAX_SESSIONS = 100
_FIELD_MAX = 4000
_SHORT_MAX = 400
_TINY_MAX = 200

VALID_STAGES = {"planned", "codex_task_created", "running",
                "verified", "failed", "memory_written", "expired"}

# V2.1:24 小时未更新的 planned/codex_task_created/running 会被标记为 expired。
# 已落到 memory_written / failed / expired 的不再变。
EXPIRY_THRESHOLD_SECONDS = 24 * 3600
_EXPIRABLE_STAGES = {"planned", "codex_task_created", "running"}

_LOCK = threading.Lock()


# ─── 工具 ──────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trunc(value: Any, limit: int = _FIELD_MAX) -> str:
    s = "" if value is None else str(value)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


def _ensure_file() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _FILE.exists():
        _atomic_write([])


def _atomic_write(items: List[Dict[str, Any]]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".adu_self_sess_", suffix=".json", dir=str(_DATA_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _FILE)
    except Exception:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _safe_load() -> List[Dict[str, Any]]:
    _ensure_file()
    try:
        with _FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        return []
    except OSError:
        return []


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """容忍 'Z' 结尾 / 缺时区的简化 ISO8601 解析。"""
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _maybe_expire_all() -> None:
    """V2.1:把超过 EXPIRY_THRESHOLD_SECONDS 仍停在中间态的 session 标 expired。
    每次 read(get/list/select/record)入口都跑一次,做完才返回最新视图。
    写入时只重写有变化的部分(尽量短锁)。"""
    with _LOCK:
        items = _safe_load()
        now = datetime.now(timezone.utc)
        changed = False
        for s in items:
            if s.get("stage") not in _EXPIRABLE_STAGES:
                continue
            last = _parse_iso(s.get("updated_at"))
            if last is None:
                continue
            if (now - last).total_seconds() <= EXPIRY_THRESHOLD_SECONDS:
                continue
            s["stage"] = "expired"
            s["expired_at"] = _now()
            s["updated_at"] = _now()
            s["expiry_reason"] = (
                f"stale_after_{EXPIRY_THRESHOLD_SECONDS // 3600}h"
            )
            changed = True
        if changed:
            _atomic_write(items)


# ─── CRUD ──────────────────────────────────────────────────────────
def create_session(goal: str, surface: str, mode: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    sid = uuid.uuid4().hex[:16]
    now = _now()
    sess: Dict[str, Any] = {
        "id": sid,
        "created_at": now,
        "updated_at": now,
        "goal": _trunc(goal, _SHORT_MAX),
        "surface": _trunc(surface, 64),
        "mode": _trunc(mode, 32),
        "stage": "planned",
        "selected_upgrade_id": None,
        # 保留 plan 里完整 recommended_upgrades(含 codex_prompt) —— select-upgrade 要用
        "recommended_upgrades": list(plan.get("recommended_upgrades") or []),
        "today_priorities": list(plan.get("today_priorities") or []),
        "plan_identity": _trunc(plan.get("identity", ""), _SHORT_MAX),
        "current_codex_prompt": None,
        "last_codex_result": None,
        "verification_result": None,
        "memory_write_result": None,
        "next_recommendation": None,
    }
    with _LOCK:
        items = _safe_load()
        items.append(sess)
        if len(items) > _MAX_SESSIONS:
            items = items[-_MAX_SESSIONS:]
        _atomic_write(items)
    return sess


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    _maybe_expire_all()
    for s in _safe_load():
        if s.get("id") == session_id:
            return s
    return None


def update_session(session_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    with _LOCK:
        items = _safe_load()
        idx = next((i for i, s in enumerate(items) if s.get("id") == session_id), None)
        if idx is None:
            return None
        sess = dict(items[idx])
        for k, v in patch.items():
            sess[k] = v
        sess["updated_at"] = _now()
        # stage 白名单兜底
        if sess.get("stage") not in VALID_STAGES:
            sess["stage"] = "planned"
        items[idx] = sess
        _atomic_write(items)
        return sess


def list_recent_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    _maybe_expire_all()
    return _safe_load()[-limit:][::-1]


# V2.1 update-stage:由 /api/adu/self/sessions/{id}/update-stage 调,
# 用户手动点 Run Codex 时把 session 从 codex_task_created → running。
# 终态(memory_written / expired / failed)不允许回退。
def update_stage(session_id: str, new_stage: str,
                 message: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """返回 (session, error_code)。"""
    if new_stage not in VALID_STAGES:
        return None, "invalid_stage"
    sess = get_session(session_id)
    if not sess:
        return None, "session_not_found"
    current = sess.get("stage")
    # 终态锁定:不允许从 memory_written / expired / failed 回退
    if current in {"memory_written", "expired", "failed"}:
        return sess, "stage_locked"
    patch: Dict[str, Any] = {"stage": new_stage}
    if message:
        patch["last_stage_message"] = _trunc(message, _SHORT_MAX)
    updated = update_session(session_id, patch)
    return updated, None


# ─── 业务 ──────────────────────────────────────────────────────────
def select_upgrade(session_id: str, upgrade_id: str) -> Optional[Dict[str, Any]]:
    """从 recommended_upgrades 中找到 upgrade,填 current_codex_prompt,推进到 codex_task_created。"""
    sess = get_session(session_id)
    if not sess:
        return None
    selected = next(
        (r for r in (sess.get("recommended_upgrades") or [])
         if r.get("id") == upgrade_id),
        None,
    )
    if not selected:
        return None
    return update_session(session_id, {
        "selected_upgrade_id": upgrade_id,
        "current_codex_prompt": selected.get("codex_prompt"),
        "stage": "codex_task_created",
    })


def _selected_upgrade(sess: Dict[str, Any]) -> Dict[str, Any]:
    sel = sess.get("selected_upgrade_id")
    for r in sess.get("recommended_upgrades") or []:
        if r.get("id") == sel:
            return r
    return {}


def _score_candidate(candidate: Dict[str, Any],
                     goal: str,
                     last_codex_result: Optional[Dict[str, Any]]) -> Tuple[int, List[str]]:
    """简单评分:返回 (score, reasons)。
    规则:
        - capability missing  优先
        - capability partial  次优先
        - risk_level=low      优先
        - estimated_cost=low  优先
        - 与近期失败原因关键词相关
        - 与用户当前 goal 关键词相关
    capability 状态从 adu_capability_map 实时读;若读不到则不加分。
    """
    score = 0
    reasons: List[str] = []

    risk = (candidate.get("risk_level") or "").lower()
    cost = (candidate.get("estimated_cost_level") or "").lower()
    cap_id = candidate.get("capability_id")
    title_low = (candidate.get("title") or "").lower()
    reason_low = (candidate.get("reason") or "").lower()

    if risk == "low":
        score += 10; reasons.append("风险低")
    elif risk == "high":
        score -= 10
    if cost == "low":
        score += 5; reasons.append("成本低")
    elif cost == "high":
        score -= 5

    # capability status
    if cap_id:
        try:
            import adu_capability_map as _capmap_local  # type: ignore
            cap = _capmap_local.get_capability(cap_id)
            if cap:
                st = cap.get("status")
                if st == "missing":
                    score += 100; reasons.append(f"能力 {cap_id} 仍 missing")
                elif st == "partial":
                    score += 50; reasons.append(f"能力 {cap_id} 仍 partial")
        except Exception:
            pass

    # 与近期失败相关
    if last_codex_result and (not last_codex_result.get("ok") or last_codex_result.get("needs_user_confirmation")):
        err_blob = " ".join([
            str(last_codex_result.get("error") or ""),
            " ".join(last_codex_result.get("blocked_terms") or []),
        ]).lower()
        # 取 3 个有信息量的关键词
        keywords = [w for w in err_blob.replace(",", " ").split() if len(w) >= 4][:3]
        if keywords and any(k in title_low or k in reason_low for k in keywords):
            score += 20; reasons.append("与近期失败相关")

    # 与 goal 关键词匹配
    if goal:
        goal_keywords = [w for w in goal.lower().replace(",", " ").split() if len(w) >= 3][:5]
        if goal_keywords and any(k in title_low or k in reason_low for k in goal_keywords):
            score += 10; reasons.append("与当前目标关键词匹配")

    return score, reasons


def _next_recommendation(sess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """V2.1 给"下一轮"打分挑最高分,排除已选中项。
    并列时按 recommended_upgrades 列表原序(早出现的优先)。"""
    sel = sess.get("selected_upgrade_id")
    recs = sess.get("recommended_upgrades") or []
    if not recs:
        return None

    goal = sess.get("goal", "")
    last_result = sess.get("last_codex_result")

    # 候选:除掉选中项;若 selected 不在表里(异常),全表参选
    candidates = [r for r in recs if r.get("id") != sel] if sel else list(recs)
    if not candidates:
        return None

    scored: List[Tuple[int, int, List[str], Dict[str, Any]]] = []  # (score, original_index, reasons, candidate)
    for idx, c in enumerate(candidates):
        s, reasons = _score_candidate(c, goal, last_result)
        scored.append((s, idx, reasons, c))
    # 高分优先;同分按出现顺序
    scored.sort(key=lambda x: (-x[0], x[1]))
    score, _idx, reasons, best = scored[0]

    return {
        "id": best.get("id"),
        "title": _trunc(best.get("title"), _TINY_MAX),
        "reason": _trunc(best.get("reason"), _SHORT_MAX),
        "risk_level": best.get("risk_level"),
        "estimated_cost_level": best.get("estimated_cost_level"),
        "suggested_mode": best.get("suggested_mode"),
        "project_id": best.get("project_id"),
        "score": score,
        "score_reason": "；".join(reasons) if reasons else "沿用推荐列表顺序",
        "score_reasons": reasons,
    }


def _verification_for(sess: Dict[str, Any], codex_result: Dict[str, Any]) -> Dict[str, Any]:
    """根据 Codex 结果 + 选中 upgrade 的 verification_steps,生成 verification_result。"""
    ok = bool(codex_result.get("ok"))
    needs_confirm = bool(codex_result.get("needs_user_confirmation"))
    exit_code = codex_result.get("exit_code")
    duration = codex_result.get("duration")
    err = codex_result.get("error")

    selected = _selected_upgrade(sess)
    steps = list(selected.get("verification_steps") or [])

    parts: List[str] = []
    if exit_code is not None:
        parts.append(f"exit={exit_code}")
    if duration is not None:
        parts.append(f"用时 {duration} 秒")
    head = "Codex 成功 " if (ok and not needs_confirm) else ("Codex 风控拦截 " if needs_confirm else "Codex 失败 ")
    summary = head + " ".join(parts)
    if needs_confirm:
        bt = codex_result.get("blocked_terms") or []
        if bt:
            summary += f" (blocked: {', '.join(bt[:6])})"
    if err and not (ok and not needs_confirm):
        summary += f" — {_trunc(err, _TINY_MAX)}"

    outcome = "ok" if (ok and not needs_confirm) else ("needs_user_confirmation" if needs_confirm else "failed")
    return {
        "summary": _trunc(summary, _SHORT_MAX),
        "outcome": outcome,
        "recommended_steps": steps,
        "checked_at": _now(),
    }


def _sanitize_codex_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """只留摘要字段。**绝不**保存完整 stdout / stderr。"""
    return {
        "ok": bool(raw.get("ok")),
        "exit_code": raw.get("exit_code"),
        "duration": raw.get("duration"),
        "summary": _trunc(raw.get("summary"), _SHORT_MAX),
        "needs_user_confirmation": bool(raw.get("needs_user_confirmation")),
        "blocked_terms": list(raw.get("blocked_terms") or [])[:16],
        "error": _trunc(raw.get("error"), _TINY_MAX),
        "http_status": raw.get("http_status"),
        # 显式忽略:stdout / stderr / stdout_excerpt / stderr_excerpt 不进库
    }


def _build_memory_event(sess: Dict[str, Any],
                        result: Dict[str, Any],
                        verification: Dict[str, Any],
                        next_rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    succeeded = result["ok"] and not result["needs_user_confirmation"]
    selected = _selected_upgrade(sess)
    selected_title = selected.get("title", "未知升级项")
    selected_risk = selected.get("risk_level", "")
    goal = sess.get("goal", "")
    next_title = (next_rec or {}).get("title") or "无"

    if succeeded:
        lesson = (
            f"本轮自我升级完成：目标={_trunc(goal, 80)}, "
            f"升级项={_trunc(selected_title, 80)}, "
            f"验证={_trunc(verification['summary'], 120)}, "
            f"下次建议={_trunc(next_title, 80)}"
        )
        fix = f"Codex ok exit={result.get('exit_code')}"
        if result.get("duration") is not None:
            fix += f", 用时 {result['duration']} 秒"
        return {
            "source": "self_upgrade",
            "problem": f"self-upgrade · {selected_title}",
            "root_cause": "",
            "fix": fix,
            "verification": verification["summary"],
            "cost": (f"用时 {result.get('duration')} 秒"
                     if result.get("duration") is not None else ""),
            "risk": selected_risk,
            "lesson": lesson,
            "tags": [
                "self_upgrade",
                f"session:{sess['id']}",
                f"upgrade:{sess.get('selected_upgrade_id')}",
                "stage:verified",
                "ok",
            ],
        }
    else:
        err_parts: List[str] = []
        if result.get("error"):
            err_parts.append(result["error"])
        if result.get("needs_user_confirmation"):
            err_parts.append("needs_user_confirmation")
        bt = result.get("blocked_terms") or []
        if bt:
            err_parts.append("blocked: " + ", ".join(bt[:6]))
        err_summary = " | ".join(err_parts) if err_parts else verification["outcome"]
        lesson = (
            f"本轮自我升级失败：目标={_trunc(goal, 80)}, "
            f"失败原因={_trunc(err_summary, 120)}, "
            f"下一步建议={_trunc(next_title, 80)}"
        )
        return {
            "source": "self_upgrade",
            "problem": f"self-upgrade · {selected_title}",
            "root_cause": _trunc(err_summary, _SHORT_MAX),
            "fix": "",
            "verification": verification["summary"],
            "cost": "",
            "risk": selected_risk,
            "lesson": lesson,
            "tags": [
                "self_upgrade",
                f"session:{sess['id']}",
                f"upgrade:{sess.get('selected_upgrade_id')}",
                "stage:failed",
                "failed",
            ],
        }


def record_codex_result(session_id: str, raw_codex_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """会话状态机的"汇合点":
       1) sanitize codex_result(只保留摘要,丢 stdout/stderr)
       2) 生成 verification_result
       3) 写 evolution memory(带 session/upgrade/stage tag)
       4) 计算 next_recommendation
       5) stage → memory_written
    """
    sess = get_session(session_id)
    if not sess:
        return None
    if not sess.get("selected_upgrade_id"):
        # 没选过升级项就汇报结果是逻辑错误;允许写但 stage 走 failed
        # (理论上 iOS 应该先 select-upgrade 才 Run)
        pass

    safe_result = _sanitize_codex_result(raw_codex_result)
    verification = _verification_for(sess, safe_result)
    next_rec = _next_recommendation(sess)

    # 写 evolution memory(摘要)
    memory_event = _build_memory_event(sess, safe_result, verification, next_rec)
    memory_write_result: Dict[str, Any]
    if _memory is not None:
        try:
            saved = _memory.add_memory(memory_event)
            memory_write_result = {"ok": True, "memory_id": saved.get("id")}
        except Exception as exc:  # 写失败也不阻塞 session 更新
            memory_write_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    else:
        memory_write_result = {"ok": False, "error": "evolution_memory_module_unavailable"}

    # stage:最终一律落到 memory_written(即使 verification.outcome=failed),
    # 表示"已落记忆";verification_result.outcome 字段携带 ok / failed / needs_user_confirmation。
    return update_session(session_id, {
        "last_codex_result": safe_result,
        "verification_result": verification,
        "memory_write_result": memory_write_result,
        "next_recommendation": next_rec,
        "stage": "memory_written",
    })


def session_path() -> str:
    return str(_FILE)
