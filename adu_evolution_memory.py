"""ChatAGI Adu — evolution memory (问题 → 根因 → 修复 → 验证 → lesson)。

第一版:JSON 文件持久化,只追加,不覆盖。
位置:/Users/a12345/Desktop/backend/data/adu_evolution_memory.json

写入策略:
- 一次写一个事件,带 id / created_at / source。
- 不允许把任何"完整 stdout / stderr / .env / token"塞进 lesson —— 调用方有责任摘要。
  module 内只做一次"长度截断"保险,但不做语义识别。
- 文件锁极简:用临时文件 + os.replace 原子换文件,避免半写。

读出策略:
- list_recent(limit) 取最近 N 条。
- summarize_recent() 输出给 SelfUpgradePlanner 的压缩摘要(避免 prompt 爆)。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── 文件位置 ──────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "data"
_MEMORY_FILE = _DATA_DIR / "adu_evolution_memory.json"

# 单条事件里每个文本字段的最大字符数。超过会截断 + 加"…(truncated)"标记。
# 保护两件事:1) 不要让 .env/stdout 整段灌进去 2) 文件别无限增长。
_PER_FIELD_MAX = 4000
# 整个内存文件保留最近 N 条;再多就删掉最老的(JSON 文件一次重写)。
_MAX_RECORDS = 500

_LOCK = threading.Lock()

# ─── 内部工具 ──────────────────────────────────────────────────────
def _ensure_file() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not _MEMORY_FILE.exists():
        _atomic_write([])


def _atomic_write(records: List[Dict[str, Any]]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    # 写到同目录临时文件再 os.replace —— 半写也不会破坏现有文件。
    fd, tmp_path = tempfile.mkstemp(prefix=".adu_evo_mem_", suffix=".json", dir=str(_DATA_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _MEMORY_FILE)
    except Exception:
        # 失败时清理临时文件
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _safe_load() -> List[Dict[str, Any]]:
    _ensure_file()
    try:
        with _MEMORY_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # 文件被外部改坏 → 不删原文件,只在内存里当空处理。
        return []
    except json.JSONDecodeError:
        return []
    except OSError:
        return []


def _truncate_str(value: Any) -> str:
    s = "" if value is None else str(value)
    if len(s) <= _PER_FIELD_MAX:
        return s
    return s[:_PER_FIELD_MAX] + "…(truncated)"


def _normalize(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": event.get("id") or uuid.uuid4().hex[:12],
        "created_at": event.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": _truncate_str(event.get("source") or "manual"),
        "problem": _truncate_str(event.get("problem") or ""),
        "root_cause": _truncate_str(event.get("root_cause") or ""),
        "fix": _truncate_str(event.get("fix") or ""),
        "verification": _truncate_str(event.get("verification") or ""),
        "cost": _truncate_str(event.get("cost") or ""),
        "risk": _truncate_str(event.get("risk") or ""),
        "lesson": _truncate_str(event.get("lesson") or ""),
        "tags": list(event.get("tags") or [])[:16],
    }


# ─── 对外 API ──────────────────────────────────────────────────────
def add_memory(event: Dict[str, Any]) -> Dict[str, Any]:
    """追加一条进化记忆。返回最终落库的对象(已 normalize)。"""
    norm = _normalize(event)
    with _LOCK:
        records = _safe_load()
        records.append(norm)
        # 控制总量
        if len(records) > _MAX_RECORDS:
            records = records[-_MAX_RECORDS:]
        _atomic_write(records)
    return norm


def list_recent(limit: int = 20) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    records = _safe_load()
    return records[-limit:][::-1]   # 新的在前


def summarize_recent(limit: int = 8) -> str:
    """给 SelfUpgradePlanner / chat context 用的压缩摘要。
    每条 problem -> lesson 一行,避免拼接完整 stdout。"""
    rs = list_recent(limit=limit)
    if not rs:
        return "(暂无进化记忆)"
    lines: List[str] = ["[阿杜进化记忆 · 最近 %d 条]" % len(rs)]
    for r in rs:
        problem = (r.get("problem") or "").strip().splitlines()
        first_line = problem[0] if problem else ""
        head = first_line[:80] + ("…" if len(first_line) > 80 else "")
        lesson = (r.get("lesson") or "").strip().splitlines()
        lesson_head = lesson[0] if lesson else ""
        lesson_short = lesson_head[:100] + ("…" if len(lesson_head) > 100 else "")
        lines.append(f"- [{r.get('created_at','?')}] {head}  → lesson: {lesson_short or '(空)'}")
    return "\n".join(lines)


def memory_path() -> str:
    return str(_MEMORY_FILE)


def memory_count() -> int:
    return len(_safe_load())
