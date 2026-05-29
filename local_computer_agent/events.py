"""
local_computer_agent/events.py
==============================
电脑总代理事件翻译器

作用：
- 把本地代理执行过程中的内部事件
  翻译成前端 / SSE 可消费的事件流
- 不依赖 OpenClaw
- 用于：
    action_started
    action_result
    verify_started
    verify_passed
    verify_failed
    retrying
    status
    done
    error

使用方式：
    from local_computer_agent.events import translate_event, sse

    event = {
        "event": "action_started",
        "payload": {
            "action": "open_app",
            "target": "WeChat"
        }
    }
    lines = translate_event(event)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger("local_computer_agent.events")


def sse(event_type: str, data: Dict[str, Any]) -> str:
    """格式化为 SSE 字符串"""
    data_json = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data_json}\n\n"


def translate_event(event: Dict[str, Any]) -> List[str]:
    """
    把一个本地代理内部事件翻译为 0~N 条 SSE 字符串

    输入格式示例：
    {
        "event": "action_started",
        "payload": {
            "action": "open_app",
            "target": "WeChat"
        }
    }
    """
    event_name = event.get("event", "") or event.get("type", "")
    payload = event.get("payload", {}) or {}

    results: List[str] = []

    # ─────────────────────────────
    # 动作开始
    # ─────────────────────────────
    if event_name == "action_started":
        results.append(sse("action_started", {
            "type": "action_started",
            "action": payload.get("action", ""),
            "target": payload.get("target", ""),
            "summary": _summarize_action(payload),
            "status": "running",
        }))

    # ─────────────────────────────
    # 动作结果
    # ─────────────────────────────
    elif event_name == "action_result":
        ok = bool(payload.get("ok", False))
        results.append(sse("action_result", {
            "type": "action_result",
            "action": payload.get("action", ""),
            "target": payload.get("target", ""),
            "success": ok,
            "summary": payload.get("summary") or _summarize_result(payload),
            "raw": payload.get("raw") or {},
        }))

    # ─────────────────────────────
    # 验证开始（截图/视频帧验证）
    # ─────────────────────────────
    elif event_name == "verify_started":
        results.append(sse("verify_started", {
            "type": "verify_started",
            "action": payload.get("action", ""),
            "verify": payload.get("verify", ""),
            "status": "running",
            "summary": payload.get("summary") or "开始视觉验证",
        }))

    # ─────────────────────────────
    # 验证成功
    # ─────────────────────────────
    elif event_name == "verify_passed":
        results.append(sse("verify_passed", {
            "type": "verify_passed",
            "action": payload.get("action", ""),
            "verify": payload.get("verify", ""),
            "success": True,
            "summary": payload.get("summary") or "验证成功",
        }))

    # ─────────────────────────────
    # 验证失败
    # ─────────────────────────────
    elif event_name == "verify_failed":
        results.append(sse("verify_failed", {
            "type": "verify_failed",
            "action": payload.get("action", ""),
            "verify": payload.get("verify", ""),
            "success": False,
            "summary": payload.get("summary") or payload.get("reason") or "验证失败",
        }))

    # ─────────────────────────────
    # 重试
    # ─────────────────────────────
    elif event_name == "retrying":
        results.append(sse("retrying", {
            "type": "retrying",
            "action": payload.get("action", ""),
            "attempt": int(payload.get("attempt", 1)),
            "summary": payload.get("summary") or "准备重试",
        }))

    # ─────────────────────────────
    # 一般状态
    # ─────────────────────────────
    elif event_name == "status":
        results.append(sse("status", {
            "type": "status",
            "message": str(payload.get("message") or ""),
        }))

    # ─────────────────────────────
    # 最终完成
    # ─────────────────────────────
    elif event_name == "done":
        results.append(sse("done", {
            "type": "done",
            "ok": bool(payload.get("ok", True)),
            "summary": payload.get("summary") or "执行完成",
            "result": payload.get("result") or {},
        }))

    # ─────────────────────────────
    # 错误
    # ─────────────────────────────
    elif event_name == "error":
        msg = payload.get("message") or payload.get("error") or str(payload)
        results.append(sse("error", {
            "type": "error",
            "message": str(msg)[:1000],
        }))

    # ─────────────────────────────
    # 未处理事件
    # ─────────────────────────────
    else:
        if event_name and not str(event_name).startswith("_"):
            logger.debug("[LocalEventTranslator] Unhandled event: %s", event_name)

    return results


# ─────────────────────────────
# 辅助函数
# ─────────────────────────────

def _summarize_action(payload: Dict[str, Any]) -> str:
    action = str(payload.get("action") or "").strip()
    target = str(payload.get("target") or "").strip()

    if action == "open_app":
        return f"🟢 打开应用：{target or '?'}"
    if action == "open_path":
        return f"📂 打开目录：{target or '?'}"
    if action == "list_directory":
        return f"📋 列出目录：{target or '?'}"
    if action == "read_file":
        return f"📖 读取文件：{target or '?'}"
    if action == "paste_text":
        return "✍️ 粘贴文本"
    if action == "press_key":
        return f"⌨️ 按键：{target or '?'}"
    if action == "click_ui":
        return f"🖱️ 点击：{target or '?'}"

    return f"⚙️ 执行动作：{action or 'unknown'} {target}".strip()


def _summarize_result(payload: Dict[str, Any]) -> str:
    action = str(payload.get("action") or "").strip()
    ok = bool(payload.get("ok", False))
    target = str(payload.get("target") or "").strip()

    if ok:
        if action == "open_app":
            return f"✅ 已打开应用：{target or '?'}"
        if action == "open_path":
            return f"✅ 已打开目录：{target or '?'}"
        if action == "list_directory":
            count = payload.get("count")
            return f"✅ 已列出目录，共 {count if count is not None else '?'} 项"
        if action == "read_file":
            line_count = payload.get("line_count")
            return f"✅ 已读取文件，共 {line_count if line_count is not None else '?'} 行"
        return "✅ 动作执行成功"

    err = payload.get("error") or "unknown_error"
    return f"❌ 动作失败：{err}"