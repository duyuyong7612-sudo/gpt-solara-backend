"""ChatAGI Adu — recursive self-improvement loop controller。

第一版:**plan_only**。
- 不调 /api/adu/codex/run。
- 不执行任何 shell。
- 不读 .env / 不写 git。
- 只在内存里把"identity + capability_map + evolution_memory → upgrade plan"串起来。

mode:
- plan_only                 默认。只生成计划返回前端。
- propose                   保留接口,与 plan_only 等价(为未来"自动给出 PR 草稿"留位置)。
- execute_with_confirmation 保留接口,本轮一律拒绝并提示用户走 /api/adu/codex/run 手动确认。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

try:
    import adu_self_upgrade_planner as _planner
except Exception:  # pragma: no cover
    _planner = None  # type: ignore


SUPPORTED_MODES = ("plan_only", "propose", "execute_with_confirmation")


def run_loop(payload: Dict[str, Any]) -> Dict[str, Any]:
    """主入口。返回 SelfUpgradePlanner.plan_upgrades() 的结果 + loop 元信息。"""
    mode_in = (payload.get("mode") or "plan_only").strip()
    if mode_in not in SUPPORTED_MODES:
        return {
            "ok": False,
            "error": "unsupported_mode",
            "mode_requested": mode_in,
            "allowed": list(SUPPORTED_MODES),
        }

    # 本版的硬约束:无论用户请求什么 mode,实际执行都只能是 plan_only。
    if mode_in == "execute_with_confirmation":
        return {
            "ok": False,
            "error": "execute_disabled_in_v1",
            "mode_requested": mode_in,
            "message": (
                "本轮自我进化系统只允许 plan_only。"
                "如需真正执行,请用 App 端逐项确认 codex_prompt,然后手动调 /api/adu/codex/run。"
            ),
        }

    if _planner is None:
        return {"ok": False, "error": "self_upgrade_planner_unavailable"}

    plan = _planner.plan_upgrades(payload)
    plan["loop"] = {
        "controller": "adu_self_loop",
        "version": 1,
        "mode_effective": "plan_only",
        "auto_executed": False,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return plan
