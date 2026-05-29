"""ChatAGI Adu — capability map (静态第一版,自我升级规划器据此分析能力缺口)。

后续可以把 status / last_verified / known_issues 改成 verifier 自动写回,
本轮就先用人工标注的快照。所有字段都不含敏感信息。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

CapabilityStatus = str  # "working" | "partial" | "missing" | "blocked"


def _cap(
    cap_id: str,
    name: str,
    status: CapabilityStatus,
    stability: float,
    description: str,
    known_issues: Optional[List[str]] = None,
    next_upgrade: str = "",
    owner_project: Optional[str] = None,
    last_verified: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": cap_id,
        "name": name,
        "status": status,
        "stability": max(0.0, min(1.0, stability)),
        "description": description,
        "known_issues": list(known_issues or []),
        "next_upgrade": next_upgrade,
        "owner_project": owner_project,                              # 用 registry id
        "last_verified": last_verified or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


CAPABILITIES: List[Dict[str, Any]] = [
    _cap(
        "home_chat", "主聊天 / 记忆 / 联网 / 文件图片理解",
        status="working", stability=0.9,
        description="HomeChatView 底部主输入栏走 /chat/prepare;recentSummary 已注入 system context。",
        known_issues=[],
        next_upgrade="把记忆切换成多用户独立 namespace,避免会话之间互相污染。",
        owner_project="gptsora",
    ),
    _cap(
        "computer_control", "电脑控制(动作)",
        status="working", stability=0.85,
        description="iOS → /api/brain/computer/action → local-agent 4317,/键盘/鼠标/截图 已通。",
        known_issues=["高风险输入(rm/sudo/格式化)目前只在 prompt 提示,后端尚未硬拦截"],
        next_upgrade="把高风险关键词在后端做硬拦截 + observe→act→verify 闭环。",
        owner_project="backend",
    ),
    _cap(
        "screenshot_polling", "实时截图(1.5s 间隔)",
        status="working", stability=0.95,
        description="ComputerControlHomeView 内部 Task 轮询,Image .id() 强制重绘。",
        known_issues=[],
        next_upgrade="加自适应间隔:用户输入活跃时降到 0.5s,空闲时回 3s。",
        owner_project="gptsora",
    ),
    _cap(
        "codex_automation", "Codex 自动化编程",
        status="working", stability=0.75,
        description="/api/adu/codex/run 支持 project_id;CodexAgentClient 已 GA。",
        known_issues=["成本偏高;无 job 队列,长任务同步阻塞;无 dry_run mode"],
        next_upgrade="加 chatgpt auth mode、cost_guard、plan_only/dry_run/execute 三档。",
        owner_project="backend",
    ),
    _cap(
        "planner_route", "后端语义 planner",
        status="partial", stability=0.7,
        description="/api/adu/planner/route 已上线,LLM 偶尔把短任务判成 ask_clarification,本地启发兜底中。",
        known_issues=["LLM 在'修 X' 短句上偶尔不稳;file_search_task 已加但还未持续验证"],
        next_upgrade="加 planner 输入/输出回归测试集,稳定 confidence>=0.8。",
        owner_project="backend",
    ),
    _cap(
        "project_registry", "授权工作区注册表",
        status="working", stability=0.85,
        description="adu_project_registry 已加 4 工程(gptsora / backend / little_beijing / chatagi_site)。",
        known_issues=["chatagi_site 目录尚未真正建立"],
        next_upgrade="加 owner / kind 元数据,supporting verifier 用得上。",
        owner_project="backend",
    ),
    _cap(
        "file_search", "授权工作区文件搜索",
        status="working", stability=0.8,
        description="/api/adu/files/search 黑名单 .env/*.key/*.pem 等;不读文件内容,只返回文件名+大小。",
        known_issues=["还没做内容关键词搜索(只支持文件名匹配)"],
        next_upgrade="加 grep 模式,只在授权工程内,跳过敏感文件,带行号摘要。",
        owner_project="backend",
    ),
    _cap(
        "self_upgrade", "递归自我进化",
        status="missing", stability=0.1,
        description="本轮正在做第一版 —— 只生成计划,不自动执行 Codex。",
        known_issues=["iOS 端 self_upgrade 仍走 Codex 卡片,未接 /api/adu/self/upgrade/plan"],
        next_upgrade="iOS 端 dispatchBackendPlannerRoute 的 .selfUpgradeTask 改成调 /api/adu/self/upgrade/plan。",
        owner_project="backend",
    ),
    _cap(
        "evolution_memory", "进化记忆",
        status="missing", stability=0.1,
        description="本轮第一版:adu_evolution_memory.py + data/adu_evolution_memory.json。",
        known_issues=["还没有自动写回机制 —— 目前只能手动 POST"],
        next_upgrade="让 Codex 任务结束 / planner 高风险拒绝时自动 add_memory(...)。",
        owner_project="backend",
    ),
    _cap(
        "cost_guard", "成本控制",
        status="missing", stability=0.0,
        description="还没有任何成本守门员;Codex 长任务可能烧钱。",
        known_issues=["未上 chatgpt auth mode;未上限额;无 plan_only/dry_run 分级"],
        next_upgrade="支持 OPENAI_CHATGPT_AUTH_MODE,把 codex/run 改成默认 dry_run。",
        owner_project="backend",
    ),
    _cap(
        "verifier", "统一验证层",
        status="partial", stability=0.5,
        description="py_compile / curl / xcodebuild 散落在各处,本轮新加 adu_verifier.py 做封装。",
        known_issues=["xcodebuild 还没真正接,只返回推荐命令字符串"],
        next_upgrade="把 EngineeringAgentPanel 的 health + run 结果都喂进 verifier 留底。",
        owner_project="backend",
    ),
    _cap(
        "realtime_voice", "黑色波纹实时语音",
        status="working", stability=0.8,
        description="OpenAI Realtime + WebRTC + 摄像头;perception payload 已带 adu_context_summary。",
        known_issues=["实时语音的 instructions 还没把 hard_constraints 注入"],
        next_upgrade="把 identity.hard_constraints 注入 realtime session instructions。",
        owner_project="backend",
    ),
    _cap(
        "memory", "长期记忆",
        status="partial", stability=0.6,
        description="主聊天侧已有记忆/上下文;AduContextStore 把工具事件合并进来。",
        known_issues=["AduContextStore 只在 iOS 端,后端没有持久化"],
        next_upgrade="把 AduContextStore 关键事件镜像到 data/adu_event_log.jsonl(只摘要,不存 stdout)。",
        owner_project="backend",
    ),
    _cap(
        "web_search", "联网检索",
        status="working", stability=0.8,
        description="/chat/prepare 已具备联网模式,主聊天可见站点图标。",
        known_issues=[],
        next_upgrade="给 planner 一个 web_business_task intent,把'订机票/查物流'引到这里。",
        owner_project="backend",
    ),
]


def list_capabilities() -> List[Dict[str, Any]]:
    return [dict(c) for c in CAPABILITIES]


def get_capability(cap_id: str) -> Optional[Dict[str, Any]]:
    for c in CAPABILITIES:
        if c["id"] == cap_id:
            return dict(c)
    return None


def summarize_by_status() -> Dict[str, List[str]]:
    """status → [cap_id, ...] 便于 SelfUpgradePlanner 快速找缺口。"""
    out: Dict[str, List[str]] = {"working": [], "partial": [], "missing": [], "blocked": []}
    for c in CAPABILITIES:
        out.setdefault(c["status"], []).append(c["id"])
    return out
