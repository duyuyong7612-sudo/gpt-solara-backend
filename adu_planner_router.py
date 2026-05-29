"""
ChatAGI Adu semantic task planner.

Mount this router from the main FastAPI app:

    from dev_agent_v1_server.adu_planner_router import router as adu_planner_router
    app.include_router(adu_planner_router)

The planner only classifies and prepares prompts. It never executes Codex,
computer actions, deployments, commits, shell commands, or code changes.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

# ✅ 项目注册表(单一来源)。导入失败也不要让 planner 挂掉 ——
# 这是兜底推断,LLM 仍然能填 project / project_id。
try:
    import adu_project_registry as _registry  # type: ignore
except Exception:  # pragma: no cover
    _registry = None  # type: ignore


router = APIRouter(prefix="/api/adu/planner", tags=["adu_planner"])

Intent = Literal[
    "computer_action",
    "codex_task",
    "claude_task",
    "chat_answer",
    "self_upgrade_task",
    "continue_previous_task",
    "confirm_pending_task",
    "cancel_pending_task",
    "edit_pending_task",
    "ask_clarification",
    "file_search_task",
]


class RecentMessage(BaseModel):
    role: str = Field(default="")
    content: str = Field(default="")


class PendingCodexTask(BaseModel):
    exists: bool = False
    project: Optional[str] = None
    prompt: Optional[str] = None
    summary: Optional[str] = None
    risk_level: Optional[str] = None
    created_at: Optional[str] = None


class PlannerRequest(BaseModel):
    text: str = Field(default="")
    surface: str = Field(default="home_chat")
    recent_messages: List[RecentMessage] = Field(default_factory=list)
    pending_codex_task: Optional[PendingCodexTask] = None
    selected_project: Optional[str] = None
    screen_state: Dict[str, Any] = Field(default_factory=dict)


class PlannerResponse(BaseModel):
    ok: bool = True
    intent: Intent
    confidence: float = 0.0
    surface: str = "home_chat"
    # 双字段并存:`project` 是历史显示名(GPTsora/Backend/LittleBeijing),
    # `project_id` 是 registry 的标准 id(gptsora/backend/little_beijing/chatagi_site),
    # 后者用于 /api/adu/codex/run 直接路由到工程目录。
    project: Optional[str] = None
    project_id: Optional[str] = None
    executor: str = "none"
    title: str = ""
    summary: str = ""
    computer_goal: str = ""
    codex_prompt: str = ""
    claude_prompt: str = ""
    chat_answer: str = ""
    risk_level: str = "low"
    needs_user_confirmation: bool = False
    blocked_terms: List[str] = Field(default_factory=list)
    next_action: str = "none"
    message: str = ""


SYSTEM_PROMPT = """
你是 ChatAGI 阿杜的任务规划器。
你的任务不是直接完成用户请求，而是判断用户当前输入应该交给哪个执行器。

你必须结合：
- 用户输入的真实语义
- 最近对话上下文
- 是否存在 pending Codex task
- 当前页面 surface
- 当前 selected_project
- 用户是否在确认、继续、取消、修改上一轮任务
- 当前系统状态 screen_state

必须返回严格 JSON。
不要输出 markdown。
不要输出解释性长文。
不要编造执行结果。
planner 只规划，不执行。

判断规则：
1. 如果用户说“继续、继续给 codex、接着做、就按这个、确认运行”，必须先看 pending_codex_task。
   - 如果存在 pending task：返回 continue_previous_task 或 confirm_pending_task。
   - 如果不存在 pending task：返回 ask_clarification。
2. 只有用户表达明确工程目标时，才返回 codex_task。
3. 普通解释、原因分析、概念说明，返回 chat_answer。
4. 明确电脑动作，返回 computer_action。
5. 自我升级、自我检查、递归自我改进类表达，返回 self_upgrade_task。
6. 不要因为文本里出现 “codex” 这个词就直接判断为 codex_task。
7. 高风险或不明确任务，needs_user_confirmation=true。
8. 如果用户输入太短或上下文不足，返回 ask_clarification。
9. 当用户用“修 / 改 / 加 / 接入 / 实现 / 修复 / 新增 + <文件名/类名/页面名/模块>”这类
   明确工程目标短句时（例:“修 HomeChatView 输入栏”、“改 server_session.py 的 planner mount”、
   “加一个 /api/xxx 路由”），即使句子很短也必须返回 codex_task,并按文件/类名归属填好 project_id。
   不要因为简短就回 ask_clarification。
10. 当用户用"找 / 帮我找 / 搜 / 在哪 / where / locate + <文件名>"这类纯查找(不要求修改)
    时,返回 file_search_task。executor 留 "none",summary 用一句话概括"在授权工作区搜 xxx";
    project_id 仍按文件名所属工程推断(server_session.py → backend, HomeChatView → gptsora 等)。

安全边界：
- 不直接执行 Codex。
- 不改代码。
- 不执行系统命令。
- 不做发布。
- 不做版本提交。
- 不读取敏感配置。
- 不返回任何密钥。

只返回一个 JSON object，字段必须包含：
ok,intent,confidence,surface,project,project_id,executor,title,summary,computer_goal,codex_prompt,
claude_prompt,chat_answer,risk_level,needs_user_confirmation,blocked_terms,next_action,message

关于 project 与 project_id：
- project 是显示名（如 GPTsora / Backend / LittleBeijing / ChatAGISite）。
- project_id 是后端 registry 的标准 id：
    gptsora        → iOS 前端 / SwiftUI / Xcode 项目
    backend        → Python FastAPI 后端 (server_session.py 等)
    little_beijing → Little Beijing 餐厅点餐 / Edge Box / Realtime
    chatagi_site   → ChatAGI 营销/落地网站
- 当 intent=codex_task 时,你必须根据用户文本判断并填写 project_id。
- 当 intent=computer_action / chat_answer / ask_clarification 时,可以留空。
""".strip()


def _pending(req: PlannerRequest) -> PendingCodexTask:
    return req.pending_codex_task or PendingCodexTask()


def _response(**kwargs: Any) -> PlannerResponse:
    base: Dict[str, Any] = {
        "ok": True,
        "intent": "ask_clarification",
        "confidence": 0.0,
        "surface": "home_chat",
        "project": None,
        "project_id": None,
        "executor": "none",
        "title": "",
        "summary": "",
        "computer_goal": "",
        "codex_prompt": "",
        "claude_prompt": "",
        "chat_answer": "",
        "risk_level": "low",
        "needs_user_confirmation": False,
        "blocked_terms": [],
        "next_action": "none",
        "message": "",
    }
    base.update(kwargs)
    return PlannerResponse(**base)


_DISPLAY_NAME_FOR_ID: Dict[str, str] = {
    "gptsora": "GPTsora",
    "backend": "Backend",
    "little_beijing": "LittleBeijing",
    "chatagi_site": "ChatAGISite",
}


def _finalize_project_id(req: PlannerRequest, resp: PlannerResponse) -> PlannerResponse:
    """
    后处理:确保 project_id 是 registry 标准 id;若 LLM 没给,按 project / 用户文本 / selected_project
    顺序兜底推断。computer_action / chat_answer / ask_clarification 不强填(留 None 也合法)。
    """
    if _registry is None:
        return resp

    # 先尝试规整 project_id —— 若 LLM 直接填了,优先采纳但归一化大小写
    pid = _registry.normalize_project_id(resp.project_id)

    # 否则从 display name 反查(LLM 常直接填 "GPTsora" / "Backend" 进 project)
    if not pid:
        pid = _registry.normalize_project_id(resp.project)

    # 还没有 → 用 selected_project
    if not pid:
        pid = _registry.normalize_project_id(req.selected_project)

    # 最后的兜底:用户文本关键词推断
    if not pid:
        # 仅对工程相关 intent 推断;闲聊/纯电脑动作不强行贴 project
        engineering_intents = {
            "codex_task",
            "claude_task",
            "self_upgrade_task",
            "confirm_pending_task",
            "continue_previous_task",
            "edit_pending_task",
            "file_search_task",
        }
        if resp.intent in engineering_intents:
            pid = _registry.infer_from_text(req.text)

    if pid:
        resp.project_id = pid
        # 同步刷新 display name(若 LLM 没填或填了别名)
        if not resp.project:
            resp.project = _DISPLAY_NAME_FOR_ID.get(pid, resp.project)
    return resp


def _project_for(text: str, selected: Optional[str]) -> str:
    selected = (selected or "").strip()
    if selected:
        return selected
    low = text.lower()
    if "little beijing" in low or "小北京" in text:
        return "LittleBeijing"
    if ("后端" in text or "backend" in low or "server_session" in low or "api" in low) and not any(
        k in text for k in ["不要改后端", "不改后端", "别改后端", "无需改后端"]
    ):
        return "Backend"
    return "GPTsora"


def _codex_prompt(text: str, project: str) -> str:
    backend_limit = "用户明确要求不要改后端；只处理 iOS 前端相关代码。" if any(
        k in text for k in ["不要改后端", "不改后端", "别改后端", "无需改后端"]
    ) else "只改本任务必要范围。"
    return (
        f"请在 {project} 项目中处理下面任务。\n\n"
        f"用户原始需求:\n{text}\n\n"
        "执行要求:\n"
        f"1. {backend_limit}\n"
        "2. 不改 local-agent、不改 Codex executor/run 逻辑、不改 Little Beijing、不改 billing。\n"
        "3. 不部署、不提交版本、不读取或输出密钥配置。\n"
        "4. 先阅读相关代码，再做最小实现；完成后运行可行的语法/编译验证。\n"
    )


def _self_upgrade_prompt(text: str) -> str:
    return (
        "请只读检查 ChatAGI 阿杜系统还能升级的地方，先输出计划，不自动执行。\n\n"
        f"用户原始需求:\n{text}\n\n"
        "要求:\n"
        "1. 分析前端、后端、agent、Codex runner 的边界和风险。\n"
        "2. 给出最小可行升级计划。\n"
        "3. 标记需要用户确认的步骤。\n"
        "4. 禁止读取或泄露 .env/API key，禁止部署、提交版本、删除数据。\n"
    )


def _fallback_plan(req: PlannerRequest, reason: str = "") -> PlannerResponse:
    text = (req.text or "").strip()
    low = text.lower()
    compact = "".join(text.split())
    pending = _pending(req)
    surface = req.surface or "home_chat"

    if not text or len(text) <= 1:
        return _response(
            intent="ask_clarification",
            confidence=0.55,
            surface=surface,
            needs_user_confirmation=True,
            next_action="ask_clarification",
            message="请再说明你想让我做什么。",
            summary=reason,
        )

    if compact in {"继续", "接着做", "继续给codex", "确认运行", "确认执行", "就按这个"} or any(
        k in text for k in ["继续给 codex", "继续给codex", "接着做", "确认运行", "就按这个"]
    ):
        if pending.exists:
            intent: Intent = "confirm_pending_task" if any(k in text for k in ["确认", "运行", "就按这个"]) else "continue_previous_task"
            return _response(
                intent=intent,
                confidence=0.9,
                surface=surface,
                project=pending.project or req.selected_project,
                executor="codex",
                title="继续待确认 Codex 任务",
                summary=pending.summary or pending.prompt or "",
                codex_prompt=pending.prompt or "",
                risk_level=pending.risk_level or "medium",
                needs_user_confirmation=True,
                next_action="show_codex_confirmation",
                message="已找到待继续的 Codex 任务，请确认后再运行。",
            )
        return _response(
            intent="ask_clarification",
            confidence=0.88,
            surface=surface,
            needs_user_confirmation=True,
            next_action="ask_clarification",
            message="当前没有待继续或待执行的 Codex 任务，请说明要处理的新任务。",
        )

    if any(k in text for k in ["取消", "别运行", "不要运行", "不用了", "算了"]):
        if pending.exists:
            return _response(
                intent="cancel_pending_task",
                confidence=0.86,
                surface=surface,
                project=pending.project or req.selected_project,
                executor="none",
                title="取消待执行任务",
                summary=pending.summary or "",
                next_action="none",
                message="已取消待执行的 Codex 任务。",
            )
        return _response(
            intent="ask_clarification",
            confidence=0.7,
            surface=surface,
            next_action="ask_clarification",
            message="当前没有待取消的 Codex 任务。",
        )

    if any(k in text for k in [
        "自我升级", "自己升级", "递归", "自我改进", "自我进化",
        "检查系统哪里还能升级", "哪里还能升级", "检查当前系统能力",
    ]):
        # ✅ V2 self-evolution:planner 不再自己写 Codex 长 prompt。
        #    直接把 next_action 改成 show_self_upgrade_plan,iOS 端应调
        #    POST /api/adu/self/upgrade/plan 拿"计划卡"。
        #    codex_prompt 留作 iOS 尚未接通时的兜底,但 message 显式提示新接口。
        prompt = _self_upgrade_prompt(text)
        return _response(
            intent="self_upgrade_task",
            confidence=0.9,
            surface=surface,
            project="Backend",
            executor="self_upgrade",
            title="自我升级检查计划",
            summary="用户要求阿杜检查系统可升级点 —— 应先调用 /api/adu/self/upgrade/plan 生成计划卡。",
            codex_prompt=prompt,                       # 兜底,iOS 旧版仍可显示
            risk_level="high",
            needs_user_confirmation=True,
            next_action="show_self_upgrade_plan",      # ← 新版前端应据此触发
            message=(
                "这是自我升级任务,应调用 POST /api/adu/self/upgrade/plan 生成计划卡。"
                "本轮 plan_only,不自动执行 Codex。"
            ),
        )

    if any(k in text for k in ["什么意思", "为什么", "解释", "原因", "含义", "是什么"]) or text.endswith(("?", "？")):
        answer = "这是一个普通解释问题，不应执行电脑动作或 Codex。"
        if "not_simple_computer_action" in low:
            answer = "not_simple_computer_action 通常表示输入没有被识别为可直接执行的简单电脑动作，因此系统会停在解释或规划层，避免误触发电脑控制。"
        return _response(
            intent="chat_answer",
            confidence=0.86,
            surface=surface,
            executor="chat",
            title="解释问题",
            summary=text,
            chat_answer=answer,
            next_action="answer_chat",
            message=answer,
        )

    if compact.startswith(("打开", "点击", "输入", "按", "切换到", "关闭")) and not any(k in text for k in ["修复", "代码", "页面", "编译"]):
        return _response(
            intent="computer_action",
            confidence=0.88,
            surface=surface,
            executor="computer",
            title="电脑动作",
            summary=text,
            computer_goal=text,
            next_action="execute_computer_action",
            message="已识别为电脑动作。",
        )

    # ─── file_search_task 兜底 ──────────────────────────────────────────
    # 用户说"找 / 帮我找 / 搜 / 在哪 / 查 + <文件名/标识符>" → 只做文件查找,不写代码。
    # 必须放在 codex_task 之前,避免被 "修/改/加" 启发覆盖。
    import re as _re_fs
    _fs_verb_pattern = r"(?:^|[\s,，。:：])(?:帮我?找|找一?下|找|搜一?下|搜索|搜|查一?下|查找|定位|在哪里?|where\s+is|locate)"
    _fs_token_pattern = r"\s*([A-Za-z_/.][A-Za-z0-9_/.\-]{2,}|\S+\.(?:py|swift|ts|tsx|js|jsx|json|m|h|md|sh|yaml|yml|txt|html|css))"
    _has_search_intent = bool(_re_fs.search(_fs_verb_pattern + _fs_token_pattern, text, flags=_re_fs.IGNORECASE))
    # 排除明显是"修改/创建/挂载"等动词:这些应继续走 codex_task
    _has_mutate_intent = any(k in text for k in ["修", "改", "加", "挂", "实现", "新增", "修复", "重写", "删除", "去掉"])
    if _has_search_intent and not _has_mutate_intent:
        project = _project_for(text, req.selected_project)
        return _response(
            intent="file_search_task",
            confidence=0.86,
            surface=surface,
            project=project,
            executor="none",
            title="文件搜索",
            summary=f"在授权工作区搜索:{text}",
            chat_answer="",
            risk_level="low",
            needs_user_confirmation=False,
            next_action="run_file_search",
            message="已识别为文件搜索请求,正在授权工作区里查找。",
        )

    has_engineering_goal = any(k in text for k in ["修复", "新增", "实现", "接入", "编译", "构建", "代码", "页面", "接口"])
    # 兜底:"修/改/加/挂 + 文件名/类名"这类简短工程指令也算明确工程目标(避免 LLM 不可用时被
    # 误判成 ask_clarification)。靠正则避免单字 "修" 误触发。
    if not has_engineering_goal:
        import re as _re
        if _re.search(r"(?:^|[\s,，。])(?:修|改|加|挂|接|加上|删除|去掉)\s*[A-Za-z_/.][A-Za-z0-9_/.\-]{2,}", text):
            has_engineering_goal = True
    if has_engineering_goal:
        project = _project_for(text, req.selected_project)
        return _response(
            intent="codex_task",
            confidence=0.84,
            surface=surface,
            project=project,
            executor="codex",
            title="自动化编程任务",
            summary=text,
            codex_prompt=_codex_prompt(text, project),
            risk_level="medium",
            needs_user_confirmation=True,
            next_action="show_codex_confirmation",
            message="已整理为 Codex 任务，请确认后再运行。",
        )

    return _response(
        intent="ask_clarification",
        confidence=0.52,
        surface=surface,
        needs_user_confirmation=True,
        next_action="ask_clarification",
        message="我还不确定要执行电脑动作、交给 Codex，还是只回答问题。请再补充一句目标。",
    )


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("planner model did not return a JSON object")
    return obj


def _call_openai_json(req: PlannerRequest) -> Dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("SORA_API_KEY")
    if not key:
        raise RuntimeError("missing model api key")

    model = os.getenv("ADU_PLANNER_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(req.model_dump(), ensure_ascii=False)},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(http_req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return _extract_json_object(content)


def _coerce_model_plan(req: PlannerRequest, obj: Dict[str, Any]) -> PlannerResponse:
    fallback = _fallback_plan(req)
    merged = fallback.model_dump()
    for key, value in obj.items():
        if key in merged and value is not None:
            merged[key] = value
    merged["ok"] = True
    merged["surface"] = merged.get("surface") or req.surface

    # ─── self_upgrade_task 安全收口 ─────────────────────────────────
    # LLM 偶尔会把这类任务标 needs_confirmation=false / risk=low,
    # 这违反了"递归自我进化默认 plan_only"原则。一律强制收紧:
    if merged.get("intent") == "self_upgrade_task":
        merged["needs_user_confirmation"] = True
        merged["risk_level"] = "high"
        merged["executor"] = "self_upgrade"
        merged["next_action"] = "show_self_upgrade_plan"
        if not (merged.get("message") or "").strip():
            merged["message"] = (
                "这是自我升级任务,应调用 POST /api/adu/self/upgrade/plan 生成计划卡。"
                "本轮 plan_only,不自动执行 Codex。"
            )

    return PlannerResponse(**merged)


@router.post("/route", response_model=PlannerResponse)
async def route(req: PlannerRequest) -> PlannerResponse:
    text = (req.text or "").strip()
    req.text = text
    if not text:
        return _finalize_project_id(req, _fallback_plan(req))

    try:
        obj = await asyncio.to_thread(_call_openai_json, req)
        resp = _coerce_model_plan(req, obj)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, ValueError, KeyError, json.JSONDecodeError):
        resp = _fallback_plan(req, reason="model_unavailable_fallback")
    return _finalize_project_id(req, resp)

