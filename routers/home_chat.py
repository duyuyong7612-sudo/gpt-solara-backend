# routers/home_chat.py
import os
import time
import uuid
import base64
import logging
import asyncio
from typing import List, Optional, Literal, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from openai import OpenAI

from media_store import store as media_store  # ✅ 只用绝对导入

log = logging.getLogger("home_chat")
router = APIRouter()

DEFAULT_MODEL = (os.getenv("HOME_CHAT_MODEL") or "gpt-5").strip()
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("HOME_CHAT_MAX_OUTPUT_TOKENS") or "1200")

HOME_CHAT_CONCURRENCY = int(os.getenv("HOME_CHAT_CONCURRENCY") or "4")
_SEM = asyncio.Semaphore(max(1, HOME_CHAT_CONCURRENCY))

ENABLE_CODE_INTERPRETER_DEFAULT = (os.getenv("HOME_CHAT_ENABLE_CODE_INTERPRETER") or "1").strip() in ("1", "true", "True")

MAX_MESSAGES = int(os.getenv("HOME_CHAT_MAX_MESSAGES") or "80")
MAX_CONTENT_CHARS = int(os.getenv("HOME_CHAT_MAX_CONTENT_CHARS") or "6000")

TRANSCRIBE_MODEL = (os.getenv("HOME_CHAT_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()

Role = Literal["system", "developer", "user", "assistant"]
AttachmentType = Literal["image", "audio"]


class Msg(BaseModel):
    role: Role = "user"
    content: str = ""  # ✅ 允许空（attachments-only）


class ChatAttachmentIn(BaseModel):
    type: AttachmentType
    id: str
    mime: Optional[str] = None
    url: Optional[str] = None


class ChatReq(BaseModel):
    model: Optional[str] = None
    messages: List[Msg] = Field(default_factory=list)
    attachments: List[ChatAttachmentIn] = Field(default_factory=list)

    enable_code_interpreter: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None


class ChatResp(BaseModel):
    text: str
    request_id: str
    model: str
    ms: int
    used_code_interpreter: bool = False
    user_transcript: Optional[str] = None


BASE_INSTRUCTIONS = """
You are Solara, a helpful assistant.
Output MUST be Markdown.
- When you output code, ALWAYS use fenced code blocks with a language tag, e.g. ```python
- If you used a tool, still provide a final human-readable answer.
""".strip()

CODE_MODE_INSTRUCTIONS = """
You have access to the python tool (Code Interpreter).
- Use it when it helps verify results, run calculations, or test code.
- ALWAYS return a final answer in Markdown after tool usage.
""".strip()


def _client() -> OpenAI:
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("SORA_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key)


def _map_role(r: str) -> str:
    # ✅ Responses API 推荐 developer/user/assistant
    # 你前端发 system 就当 developer 处理
    if r == "system":
        return "developer"
    if r == "developer":
        return "developer"
    if r == "assistant":
        return "assistant"
    return "user"


def _sanitize_messages(msgs: List[Msg], allow_empty_last_user: bool) -> List[Msg]:
    if not msgs and not allow_empty_last_user:
        raise HTTPException(status_code=400, detail="messages is empty")

    if len(msgs) > MAX_MESSAGES:
        msgs = msgs[-MAX_MESSAGES:]

    out: List[Msg] = []
    for i, m in enumerate(msgs):
        c = (m.content or "").strip()
        if len(c) > MAX_CONTENT_CHARS:
            c = c[:MAX_CONTENT_CHARS]

        if not c:
            if allow_empty_last_user and i == len(msgs) - 1 and m.role == "user":
                out.append(Msg(role=m.role, content=""))
            continue

        out.append(Msg(role=m.role, content=c))

    if not out and allow_empty_last_user:
        out = [Msg(role="user", content="")]

    if not out:
        raise HTTPException(status_code=400, detail="messages are empty after trimming")

    return out


def _to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            return {}
    d = {}
    for k in ("type", "content", "code", "outputs", "id"):
        try:
            v = getattr(x, k)
            if v is not None:
                d[k] = v
        except Exception:
            pass
    return d


def _extract_text(resp: Any) -> str:
    t = (getattr(resp, "output_text", None) or "").strip()
    if t:
        return t

    out = getattr(resp, "output", None) or []
    chunks: List[str] = []
    for item in out:
        try:
            content = getattr(item, "content", None) or []
        except Exception:
            content = _to_dict(item).get("content") or []
        for c in content or []:
            cd = _to_dict(c)
            tt = cd.get("text") or cd.get("transcript")
            if tt:
                chunks.append(str(tt))

    return "".join(chunks).strip()


def _wants_ci(req: ChatReq) -> bool:
    if req.enable_code_interpreter is not None:
        return bool(req.enable_code_interpreter)
    return ENABLE_CODE_INTERPRETER_DEFAULT


def _build_tools_and_include(use_ci: bool) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    tools: List[Dict[str, Any]] = []
    include: List[str] = []
    used_ci = False

    if use_ci:
        tools.append({
            "type": "code_interpreter",
            "container": {"type": "auto", "memory_limit": "4g"},
        })
        include.append("code_interpreter_call.outputs")
        used_ci = True

    return tools, include, used_ci


def _render_ci_markdown(resp: Any) -> str:
    out = getattr(resp, "output", None) or []
    parts: List[str] = []
    found = False

    for item in out:
        d = _to_dict(item)
        if d.get("type") != "code_interpreter_call":
            continue
        found = True
        code = d.get("code") or ""
        if code:
            parts.append("```python\n" + str(code).strip() + "\n```")
        outputs = d.get("outputs") or []
        for o in outputs:
            od = _to_dict(o)
            otype = od.get("type")
            if otype == "logs" and od.get("logs"):
                parts.append("```text\n" + str(od["logs"]).rstrip() + "\n```")
            elif od:
                parts.append("```json\n" + str(od) + "\n```")

    return "\n\n".join(parts).strip() if found else ""


async def _responses_create(payload: Dict[str, Any]) -> Any:
    def _call():
        c = _client()
        return c.responses.create(**payload)
    return await asyncio.to_thread(_call)


async def _transcribe_audio(path: str) -> str:
    def _call() -> str:
        c = _client()
        with open(path, "rb") as f:
            r = c.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=f,
                response_format="text",
            )
        if isinstance(r, str):
            return r
        if hasattr(r, "text"):
            return (getattr(r, "text") or "")
        return str(r)

    text = await asyncio.to_thread(_call)
    return (text or "").strip()


def _build_openai_input(msgs: List[Msg],
                        attachments: List[ChatAttachmentIn],
                        user_transcript: Optional[str]) -> List[Dict[str, Any]]:
    """
    ✅ 关键修复：
    - 历史消息全部用 content="纯字符串"（developer/user/assistant）
    - 只有最后一条 user（且带 attachments）才用 content=[{type: input_text/input_image...}]
    """
    out: List[Dict[str, Any]] = []

    # 找到最后一条 user 的索引（用于塞 attachments）
    last_user_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].role == "user":
            last_user_idx = i
            break

    for i, m in enumerate(msgs):
        role = _map_role(m.role)
        text = (m.content or "").strip()

        # 先把历史都当纯文本消息塞进去（非常稳）
        out.append({"role": role, "content": text})

    # 没有 user：补一个
    if last_user_idx < 0:
        out.append({"role": "user", "content": ""})
        last_user_idx = len(out) - 1

    # 如果没有 attachments，就不做“结构化 content”
    if not attachments and not user_transcript:
        return out

    # ✅ 把最后 user 这一条改为“结构化 content parts”
    parts: List[Dict[str, Any]] = []

    if out[last_user_idx].get("content"):
        parts.append({"type": "input_text", "text": out[last_user_idx]["content"]})

    # images
    for a in attachments:
        if a.type != "image":
            continue
        meta = media_store.get(a.id)
        if not meta:
            raise HTTPException(status_code=400, detail="image media not found or expired: %s" % a.id)
        raw = meta.path.read_bytes()
        mime = meta.mime or a.mime or "image/jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{b64}"})

    # transcript
    if user_transcript:
        parts.append({"type": "input_text", "text": user_transcript})

    if not parts:
        parts = [{"type": "input_text", "text": "用户发送了附件但没有文字。请先简要总结附件的关键信息，然后询问用户希望你做什么。"}]

    out[last_user_idx] = {"role": "user", "content": parts}
    return out


@router.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq, request: Request):
    rid = request.headers.get("x-request-id") or "hc_%s" % uuid.uuid4().hex[:12]
    model = (req.model or DEFAULT_MODEL).strip()
    max_out = req.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS

    allow_empty_last_user = bool(req.attachments)
    msgs = _sanitize_messages(req.messages, allow_empty_last_user=allow_empty_last_user)

    # audio -> transcript
    user_transcript: Optional[str] = None
    if req.attachments:
        transcripts: List[str] = []
        for a in req.attachments:
            if a.type != "audio":
                continue
            meta = media_store.get(a.id)
            if not meta:
                raise HTTPException(status_code=400, detail="audio media not found or expired: %s" % a.id)
            t = await _transcribe_audio(str(meta.path))
            if t:
                transcripts.append(t)
        if transcripts:
            user_transcript = "\n".join(transcripts).strip()

    use_ci = _wants_ci(req)
    tools, include, used_ci = _build_tools_and_include(use_ci)

    instructions = BASE_INSTRUCTIONS
    if used_ci:
        instructions += "\n\n" + CODE_MODE_INSTRUCTIONS

    openai_input = _build_openai_input(msgs, req.attachments or [], user_transcript=user_transcript)

    payload: Dict[str, Any] = {
        "model": model,
        "input": openai_input,
        "instructions": instructions,
        "max_output_tokens": max_out,
        "store": False,
    }

    if model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": "low"}

    if tools:
        payload["tools"] = tools
    if include:
        payload["include"] = include
    if req.temperature is not None:
        payload["temperature"] = float(req.temperature)

    t0 = time.time()
    try:
        async with _SEM:
            resp = await _responses_create(payload)

        text = _extract_text(resp)

        if not text and used_ci:
            text = _render_ci_markdown(resp)

        if not text:
            payload2 = dict(payload)
            payload2.pop("tools", None)
            payload2.pop("include", None)
            resp2 = await _responses_create(payload2)
            text = _extract_text(resp2)

        ms = int((time.time() - t0) * 1000)
        log.info("[{}] /chat model={} ms={} chars={} ci={} att={}".format(
            rid, model, ms, len(text), used_ci, len(req.attachments or [])
        ))

        if not text:
            raise HTTPException(status_code=502, detail="OpenAI returned no text output")

        return ChatResp(
            text=text,
            request_id=rid,
            model=model,
            ms=ms,
            used_code_interpreter=used_ci,
            user_transcript=user_transcript,
        )

    except HTTPException:
        raise
    except Exception as e:
        ms = int((time.time() - t0) * 1000)
        log.exception("[{}] /chat failed in {}ms: {}".format(rid, ms, e))
        raise HTTPException(status_code=500, detail=str(e))
