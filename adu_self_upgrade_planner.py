"""ChatAGI Adu — self upgrade planner。

输入:
    {
        "goal":           "你自己检查系统哪里还能升级",
        "surface":        "home_chat" | "computer_control",
        "mode":           "plan_only" | "dry_run" | "execute"  # 第一版只接受 plan_only
        "recent_events":  [...],   # 可选,iOS 端 AduContextStore 摘要
        "system_state":   {...},   # 可选,任何上下文 key/value
    }

输出(见 docstring 末尾的样例)。

实现策略:
  1. 取 identity / capability_map / evolution_memory(本进程内 import)
  2. 把 capability_map 按 status 桶分:missing > partial > working/blocked
  3. 把"内置 8 项优先级"和 capability 缺口做交集,生成候选 upgrade 列表
  4. 给每个 upgrade 标注:
       - executor       —— 多数标 plan_only/verifier;只有真要改代码的标 codex
       - project_id     —— 默认 backend(自我进化属于后端)
       - risk_level     —— 默认 medium;触碰高风险(部署/git/付费)→ high
       - estimated_cost_level
       - suggested_mode —— 默认 plan_only
       - codex_prompt   —— 给用户日后点"执行"用的安全 prompt(永远以"先生成计划再问我"开头)
       - verification_steps —— py_compile / curl /xcodebuild 命令样例
  5. today_priorities 取前 3 个 risk!=high 且 status in {missing, partial} 的候选
  6. 整体返回 requires_user_confirmation=True,本轮强制不自动执行。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import adu_self_identity as _identity
except Exception:  # pragma: no cover
    _identity = None  # type: ignore

try:
    import adu_capability_map as _capmap
except Exception:  # pragma: no cover
    _capmap = None  # type: ignore

try:
    import adu_evolution_memory as _memory
except Exception:  # pragma: no cover
    _memory = None  # type: ignore


# ─── 内置候选 ─────────────────────────────────────────────────────
# 每个候选会和 capability_map 的某条匹配 —— 命中"missing/partial"才会出现在 recommended_upgrades。
_BUILTIN_CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": "cost_guard",
        "title": "降低 Codex 成本:接 chatgpt auth mode + 三档执行(plan_only/dry_run/execute)",
        "capability_id": "cost_guard",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "medium",
        "estimated_cost_level": "low",
        "suggested_mode": "plan_only",
        "verification_steps": [
            {"tool": "py_compile", "command": "python -m py_compile adu_orchestrator/codex_executor.py"},
            {"tool": "curl",       "command": "curl http://127.0.0.1:8000/api/adu/codex/health"},
        ],
    },
    {
        "id": "unified_adu_context",
        "title": "统一阿杜上下文:AduTimeline + recentSummary 反向写回后端 data/adu_event_log.jsonl",
        "capability_id": "memory",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "low",
        "estimated_cost_level": "low",
        "suggested_mode": "plan_only",
        "verification_steps": [
            {"tool": "py_compile", "command": "python -m py_compile adu_event_log.py"},
            {"tool": "curl",       "command": "curl -X POST http://127.0.0.1:8000/api/adu/events/append -d '{\"...\":\"...\"}'"},
        ],
    },
    {
        "id": "planner_route_stabilize",
        "title": "后端语义 planner 稳定化:加输入/输出回归测试,目标 confidence>=0.8 在 5 测试集",
        "capability_id": "planner_route",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "low",
        "estimated_cost_level": "low",
        "suggested_mode": "dry_run",
        "verification_steps": [
            {"tool": "py_compile", "command": "python -m py_compile adu_planner_router.py tests/test_planner_route.py"},
        ],
    },
    {
        "id": "project_registry_owner_meta",
        "title": "工程注册表加 owner / kind 元数据,让 verifier 能按工程类型给推荐验证步骤",
        "capability_id": "project_registry",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "low",
        "estimated_cost_level": "low",
        "suggested_mode": "dry_run",
        "verification_steps": [
            {"tool": "curl", "command": "curl http://127.0.0.1:8000/api/adu/projects"},
        ],
    },
    {
        "id": "codex_job_queue",
        "title": "Codex 异步 job 队列:把长任务从同步阻塞改成 jobs/{id} 轮询,避免 UI 卡住",
        "capability_id": "codex_automation",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "medium",
        "estimated_cost_level": "medium",
        "suggested_mode": "plan_only",
        "verification_steps": [
            {"tool": "curl", "command": "curl -X POST http://127.0.0.1:8000/api/adu/codex/run -d '{\"task\":\"echo\",\"project_id\":\"backend\",\"timeout\":10}'"},
        ],
    },
    {
        "id": "evolution_memory_autowrite",
        "title": "Evolution Memory 自动写回:Codex 失败 / planner 高风险拒绝 / verifier 失败时自动 add_memory",
        "capability_id": "evolution_memory",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "low",
        "estimated_cost_level": "low",
        "suggested_mode": "dry_run",
        "verification_steps": [
            {"tool": "curl", "command": "curl -X POST http://127.0.0.1:8000/api/adu/self/evolution-memory -d '{\"problem\":\"...\"}'"},
            {"tool": "curl", "command": "curl http://127.0.0.1:8000/api/adu/self/evolution-memory"},
        ],
    },
    {
        "id": "unified_verifier",
        "title": "Verifier 统一封装:py_compile / curl / xcodebuild 推荐命令 一个接口出",
        "capability_id": "verifier",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "low",
        "estimated_cost_level": "low",
        "suggested_mode": "dry_run",
        "verification_steps": [
            {"tool": "py_compile", "command": "python -m py_compile adu_verifier.py"},
        ],
    },
    {
        "id": "computer_observe_act_verify",
        "title": "电脑 observe→act→verify 闭环:每次 computer_action 后 0.3s 与 1.0s 各取一张截图 diff 比较,失败时报警",
        "capability_id": "computer_control",
        "executor": "codex",
        "project_id": "backend",
        "risk_level": "medium",
        "estimated_cost_level": "low",
        "suggested_mode": "plan_only",
        "verification_steps": [
            {"tool": "curl", "command": "curl -X POST http://127.0.0.1:8000/api/brain/computer/action -d '{\"action\":\"screenshot\"}'"},
        ],
    },
    {
        "id": "self_upgrade_router_iOS",
        "title": "iOS 接 /api/adu/self/upgrade/plan:self_upgrade_task 不再走 Codex 卡片",
        "capability_id": "self_upgrade",
        "executor": "codex",
        "project_id": "gptsora",
        "risk_level": "low",
        "estimated_cost_level": "low",
        "suggested_mode": "plan_only",
        "verification_steps": [
            {"tool": "xcodebuild", "command": (
                'xcodebuild -workspace /Users/a12345/Desktop/GPTsora/GPTsora.xcworkspace '
                '-scheme "GPT Solara" -configuration Debug '
                "-destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build"
            )},
        ],
    },
]


# ─── 安全 prompt 头 ─────────────────────────────────────────────────
def _safe_codex_prompt(title: str, capability: Optional[Dict[str, Any]], suggested_mode: str) -> str:
    """所有 codex_prompt 都以"hard constraints + plan 先行"开头。
    用户日后点了"确认执行"才会把这段 prompt 真正交给 /api/adu/codex/run。"""
    constraints = _identity.HARD_CONSTRAINTS if _identity else []
    cap_info = ""
    if capability:
        cap_info = (
            f"能力: {capability['id']} — {capability['name']} (当前 status={capability['status']})\n"
            f"已知问题: {capability.get('known_issues') or []}\n"
            f"建议下一步: {capability.get('next_upgrade','')}\n"
        )
    head = "\n".join([
        f"# Adu 自我升级任务:{title}",
        "",
        "## 硬约束(违反则拒绝执行)",
        *[f"- {c}" for c in constraints],
        "",
        "## 执行模式",
        f"- 当前 mode:{suggested_mode}",
        "- 第一步必须是只读扫描和方案输出,不要直接改文件。",
        "- 任何文件修改前,先把 diff 与影响范围打印出来等用户在 App 端确认。",
        "- 不要 git commit / push / 部署 / 改 .env / 安装新依赖 / 改 local-agent。",
        "",
        "## 上下文",
        cap_info,
    ])
    return head.strip() + "\n"


# ─── 主函数 ───────────────────────────────────────────────────────
def plan_upgrades(payload: Dict[str, Any]) -> Dict[str, Any]:
    goal = (payload.get("goal") or "").strip()
    surface = (payload.get("surface") or "home_chat").strip()
    mode_in = (payload.get("mode") or "plan_only").strip()
    # 第一版强制 plan_only —— execute/dry_run 在 codex_prompt 里做建议,但接口本身不允许执行。
    effective_mode = "plan_only"

    recent_events = payload.get("recent_events") or []
    if not isinstance(recent_events, list):
        recent_events = []
    system_state = payload.get("system_state") or {}
    if not isinstance(system_state, dict):
        system_state = {}

    identity_payload = _identity.get_identity() if _identity else {"identity": "(identity module unavailable)", "long_term_goals": []}
    cap_list = _capmap.list_capabilities() if _capmap else []
    cap_summary_by_status = _capmap.summarize_by_status() if _capmap else {}
    memory_summary = _memory.summarize_recent(8) if _memory else "(memory module unavailable)"
    memory_count = _memory.memory_count() if _memory else 0

    # current_state:能力地图按"missing → partial → working/blocked"排序后的一行摘要
    status_order = {"missing": 0, "partial": 1, "blocked": 2, "working": 3}
    cap_sorted = sorted(cap_list, key=lambda c: (status_order.get(c.get("status", "working"), 3), c.get("id", "")))
    current_state: List[str] = [
        f"{c['id']} · {c['name']} · status={c['status']} · stab={c['stability']:.2f}"
        for c in cap_sorted
    ]

    # findings:把"missing/partial"的能力 + 它的 known_issues 提炼成单行
    findings: List[str] = []
    for c in cap_sorted:
        if c["status"] in ("missing", "partial", "blocked"):
            issues = ", ".join(c.get("known_issues") or []) or c.get("next_upgrade", "")
            findings.append(f"[{c['status']}] {c['id']} — {issues}")

    # recommended_upgrades:遍历内置候选,如果对应 capability 是 missing/partial 才收
    recommended: List[Dict[str, Any]] = []
    for cand in _BUILTIN_CANDIDATES:
        cap = _capmap.get_capability(cand["capability_id"]) if _capmap else None
        if cap is None:
            continue
        if cap["status"] not in ("missing", "partial", "blocked"):
            continue
        reason_parts: List[str] = [f"该能力当前 status={cap['status']}"]
        if cap.get("known_issues"):
            reason_parts.append("已知问题: " + "; ".join(cap["known_issues"]))
        elif cap.get("next_upgrade"):
            reason_parts.append("规划: " + cap["next_upgrade"])
        rec = {
            "id": cand["id"],
            "title": cand["title"],
            "reason": " | ".join(reason_parts),
            "capability_id": cand["capability_id"],
            "executor": cand["executor"],
            "project_id": cand["project_id"],
            "risk_level": cand["risk_level"],
            "estimated_cost_level": cand["estimated_cost_level"],
            "suggested_mode": cand["suggested_mode"],
            "mode": effective_mode,            # 本轮接口实际给的 mode
            "requires_user_confirmation": True,
            "codex_prompt": _safe_codex_prompt(cand["title"], cap, cand["suggested_mode"]),
            "verification_steps": list(cand["verification_steps"]),
        }
        recommended.append(rec)

    # today_priorities:取前 3 个 risk!=high 且 status in {missing, partial} 的候选
    today_priorities = [
        {"id": r["id"], "title": r["title"], "risk_level": r["risk_level"], "executor": r["executor"]}
        for r in recommended
        if r["risk_level"] != "high"
    ][:3]

    # capability_map_summary:精简,给 iOS 当卡片标签
    capability_map_summary = [
        {"id": c["id"], "name": c["name"], "status": c["status"], "stability": c["stability"]}
        for c in cap_sorted
    ]

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": identity_payload.get("identity", ""),
        "long_term_goals": identity_payload.get("long_term_goals", []),
        "core_principles": identity_payload.get("core_principles", []),
        "hard_constraints": identity_payload.get("hard_constraints", []),
        "goal": goal,
        "surface": surface,
        "mode_requested": mode_in,
        "mode_effective": effective_mode,
        "current_state": current_state,
        "capability_map_summary": capability_map_summary,
        "capability_buckets": cap_summary_by_status,
        "findings": findings,
        "recommended_upgrades": recommended,
        "today_priorities": today_priorities,
        "evolution_memory_summary": memory_summary,
        "evolution_memory_count": memory_count,
        "requires_user_confirmation": True,
        "auto_executed": False,
        "message": (
            "已生成自我升级计划,默认不自动执行。请在 App 端逐项确认后,"
            "再把对应 codex_prompt 提交到 /api/adu/codex/run。"
        ),
    }
