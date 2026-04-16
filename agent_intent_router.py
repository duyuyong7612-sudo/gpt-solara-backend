"""
agent_intent_router.py  v2.0
============================
可执行动作协议（四类基础工具 + 安全层）

工具类型：
  exec        — 终端命令执行
  read_file   — 文件读取（支持行范围）
  write_file  — 文件写入（限工作区）
  search_code — 代码搜索

安全层：
  A. 危险命令黑名单 → 自动拒绝
  B. 工作区白名单   → 文件操作限 SAFE_ROOTS
  C. 高危命令确认   → 写操作/删除需 confirm=true
  D. 超时限制       → exec 最大 180s

动作协议（嵌入 AI 回复）：
  [ADU_ACTION:exec|命令]
  [ADU_ACTION:read_file|路径|起始行|结束行]
  [ADU_ACTION:write_file|路径|内容]
  [ADU_ACTION:search_code|关键词|路径]
  [ADU_ACTION:agent_task|任务描述]
"""

import re
import json
import logging
import os
import asyncio
import subprocess
import shlex
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger("agent_intent")

# ══════════════════════════════════════════════════════════════════
# 安全配置
# ══════════════════════════════════════════════════════════════════

# 工作区白名单（文件读写只允许这些路径下）
SAFE_ROOTS: List[str] = [
    os.path.expanduser("~/Desktop/GPTsora"),
    os.path.expanduser("~/Desktop/backend"),
    os.path.expanduser("~/Desktop/chatterbox"),
    os.path.expanduser("~/Desktop/frontend"),
    os.path.expanduser("~/.openclaw/workspace"),
    "/tmp",
]

# 危险命令黑名单（包含任意一条即拒绝）
DANGEROUS_PATTERNS: List[str] = [
    "rm -rf /",
    "rm -rf ~",
    "sudo rm",
    "mkfs",
    "> /dev/",
    "dd if=/dev/zero",
    "dd if=/dev/urandom",
    "chmod -R 777 /",
    "chown -R",
    ":(){:|:&};:",  # fork bomb
    "sudo shutdown",
    "sudo reboot",
    "sudo halt",
    "launchctl unload /System",
    "defaults delete",
    "killall Finder",
    "killall SystemUIServer",
]

# 需要 confirm=true 才执行的高危操作
CONFIRM_REQUIRED_PATTERNS: List[str] = [
    "rm ",
    "rmdir",
    "git push",
    "git reset --hard",
    "git clean -fd",
    "DROP TABLE",
    "DELETE FROM",
    "truncate",
    "pkill",
    "kill -9",
]


def _is_dangerous(cmd: str) -> Tuple[bool, str]:
    """检查是否为危险命令，返回 (是否危险, 原因)"""
    cmd_lower = cmd.lower()
    for pat in DANGEROUS_PATTERNS:
        if pat.lower() in cmd_lower:
            return True, f"命令包含危险模式: '{pat}'"
    return False, ""


def _needs_confirm(cmd: str) -> Tuple[bool, str]:
    """检查是否需要用户确认"""
    cmd_lower = cmd.lower()
    for pat in CONFIRM_REQUIRED_PATTERNS:
        if pat.lower() in cmd_lower:
            return True, f"命令包含高危操作: '{pat}'，需要显式确认"
    return False, ""


def _is_path_safe(path: str) -> Tuple[bool, str]:
    """检查文件路径是否在安全工作区内"""
    resolved = str(Path(os.path.expanduser(path)).resolve())
    for root in SAFE_ROOTS:
        if resolved.startswith(root):
            return True, ""
    return False, f"路径 '{path}' 超出允许的工作区范围。允许路径: {', '.join(SAFE_ROOTS)}"


def _expand_path(p: str) -> str:
    if p.startswith("~/") or p == "~":
        return os.path.expanduser(p)
    return p


# ══════════════════════════════════════════════════════════════════
# AI System Prompt
# ══════════════════════════════════════════════════════════════════

AGENT_TOOLS_PROMPT = """
## 你的额外能力（静默使用，不要改变你的对话风格）

你可以操控用户的电脑。这只是你的一个附加能力，不是你的身份。
你依然是一个通用 AI 助手，有记忆、有个性、懂用户的项目。

### 重要：保持自然对话
- 不要说"我现在帮你执行命令"这种机械话
- 像一个懂编程的朋友一样说话，顺便帮忙干活
- 读完代码后给出你的理解和分析
- 执行完操作后用自然语言总结结果

### 可用动作（A/B 两类）

#### A. 四类基础工具（直接执行，秒级返回）
[ADU_ACTION:exec|命令]                         — 执行终端命令
[ADU_ACTION:read_file|路径]                    — 读取文件完整内容
[ADU_ACTION:read_file|路径|起始行|结束行]       — 读取指定行范围
[ADU_ACTION:write_file|路径|内容]              — 写入文件（限项目工作区）
[ADU_ACTION:search_code|关键词|路径]           — 搜索代码

#### B. 复杂任务（智能体全权处理，支持多步操作）
[ADU_ACTION:agent_task|任务描述]

复杂任务例子：
- [ADU_ACTION:agent_task|编译 GPTsora 项目并安装到已连接的 iPhone 上]
- [ADU_ACTION:agent_task|查看 Git 状态，如果有改动帮我 commit 并 push]
- [ADU_ACTION:agent_task|打开浏览器访问 github.com 并截图]
- [ADU_ACTION:agent_task|发微信给信玉说我回来了]
- [ADU_ACTION:agent_task|检查系统健康状态，包括磁盘、内存、CPU]

### 安全规则（必须遵守）
1. 文件写入只在项目工作区内（GPTsora/backend/chatterbox/frontend）
2. 删除文件/git push/kill 进程等操作请先告知用户再执行
3. 不执行系统破坏性命令（rm -rf /、sudo rm、格式化等）
4. 动作标记自然嵌入回复，不单独列出来
5. 不需要操作电脑时正常聊天，不输出任何动作标记
"""

# ══════════════════════════════════════════════════════════════════
# 动作解析
# ══════════════════════════════════════════════════════════════════

ACTION_PATTERN = re.compile(r'\[ADU_ACTION:([^\]]+)\]')


def extract_actions(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """从 AI 回复中提取动作标记，返回（清理后文本，动作列表）"""
    actions = []
    clean_lines = []

    for line in text.split('\n'):
        match = ACTION_PATTERN.search(line)
        if match:
            raw = match.group(1)
            parts = raw.split('|')
            action_type = parts[0].strip() if parts else ''
            params = [p.strip() for p in parts[1:]]
            actions.append({
                'type': action_type,
                'params': params,
                'raw': match.group(0),
            })
            cleaned = line.replace(match.group(0), '').strip()
            if cleaned:
                clean_lines.append(cleaned)
        else:
            clean_lines.append(line)

    return '\n'.join(clean_lines).strip(), actions


# ══════════════════════════════════════════════════════════════════
# 四类基础工具执行（含安全层）
# ══════════════════════════════════════════════════════════════════

async def _tool_exec(cmd: str, confirm: bool = False, timeout: int = 60) -> Dict[str, Any]:
    """
    工具1: 终端命令执行
    - 危险命令直接拒绝
    - 高危操作需 confirm=True
    - 超时保护
    """
    # 安全检查
    is_danger, danger_reason = _is_dangerous(cmd)
    if is_danger:
        return {
            "ok": False,
            "blocked": True,
            "reason": f"⛔ 安全拦截: {danger_reason}",
            "output": "",
        }

    needs_conf, conf_reason = _needs_confirm(cmd)
    if needs_conf and not confirm:
        return {
            "ok": False,
            "needs_confirm": True,
            "reason": f"⚠️ {conf_reason}。如确认执行，请回复【确认】后重试。",
            "output": "",
            "cmd": cmd,
        }

    # 长时命令加大超时
    if any(k in cmd for k in ['xcodebuild', 'pod install', 'npm install', 'pip install', 'cargo build']):
        timeout = max(timeout, 180)

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "TERM": "xterm-256color"},
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "output": f"⏰ 命令超时（{timeout}s）: {cmd[:80]}"}

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        return {
            "ok": True,
            "output": output,
            "exit_code": proc.returncode,
            "cmd": cmd,
        }
    except Exception as e:
        return {"ok": False, "output": f"执行失败: {e}"}


async def _tool_read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> Dict[str, Any]:
    """
    工具2: 文件读取
    - 路径安全检查（工作区限制）
    - 支持行范围
    - 大文件自动截断（最多500行）
    """
    expanded = _expand_path(path)
    safe, reason = _is_path_safe(expanded)
    if not safe:
        return {"ok": False, "output": f"⛔ {reason}"}

    if not os.path.exists(expanded):
        return {"ok": False, "output": f"❌ 文件不存在: {path}"}

    try:
        with open(expanded, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total = len(lines)
        file_name = os.path.basename(expanded)

        if start_line is not None and end_line is not None:
            # 行范围读取（1-based）
            sl = max(1, start_line) - 1
            el = min(total, end_line)
            content = ''.join(lines[sl:el])
            header = f"📄 {file_name} (第{start_line}-{end_line}行 / 共{total}行)\n\n"
        else:
            # 全文读取，超过500行截断
            MAX_LINES = 500
            if total > MAX_LINES:
                content = ''.join(lines[:MAX_LINES])
                header = f"📄 {file_name} (共{total}行，显示前{MAX_LINES}行)\n\n"
                content += f"\n\n... [文件过长，已截断。如需后续内容请用行范围读取: read_file|{path}|{MAX_LINES+1}|{min(total, MAX_LINES*2)}] ..."
            else:
                content = ''.join(lines)
                header = f"📄 {file_name} (共{total}行)\n\n"

        return {"ok": True, "output": header + content, "path": expanded, "total_lines": total}
    except Exception as e:
        return {"ok": False, "output": f"读取失败: {e}"}


async def _tool_write_file(path: str, content: str, confirm: bool = False) -> Dict[str, Any]:
    """
    工具3: 文件写入
    - 路径安全检查（严格工作区限制）
    - 自动备份（写入前 git commit）
    - 始终需要确认（高危操作）
    """
    expanded = _expand_path(path)
    safe, reason = _is_path_safe(expanded)
    if not safe:
        return {"ok": False, "output": f"⛔ {reason}"}

    if not confirm:
        file_exists = os.path.exists(expanded)
        action = "覆盖" if file_exists else "新建"
        return {
            "ok": False,
            "needs_confirm": True,
            "reason": f"⚠️ 即将{action}文件: {path} ({len(content)} 字符)。如确认，请回复【确认】后重试。",
            "output": "",
            "path": path,
        }

    try:
        # 自动创建目录
        os.makedirs(os.path.dirname(expanded), exist_ok=True)

        # 写入前 git 备份
        git_dir = next((r for r in SAFE_ROOTS if expanded.startswith(r) and os.path.exists(os.path.join(r, '.git'))), None)
        if git_dir:
            try:
                subprocess.run(
                    ['git', '-C', git_dir, 'add', '-A'],
                    capture_output=True, timeout=5
                )
                subprocess.run(
                    ['git', '-C', git_dir, 'commit', '-m', f'[ADU-BACKUP] before write {os.path.basename(expanded)}', '--allow-empty'],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass

        with open(expanded, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "ok": True,
            "output": f"✅ 已写入: {path} ({len(content)} 字符，{content.count(chr(10))+1} 行)",
            "path": expanded,
        }
    except Exception as e:
        return {"ok": False, "output": f"写入失败: {e}"}


async def _tool_search_code(keyword: str, search_path: str = "~/Desktop/GPTsora",
                             file_ext: str = "") -> Dict[str, Any]:
    """
    工具4: 代码搜索
    - grep -rn 带行号
    - 支持指定文件类型
    - 最多返回50条结果
    """
    expanded = _expand_path(search_path)
    safe, reason = _is_path_safe(expanded)
    if not safe:
        return {"ok": False, "output": f"⛔ {reason}"}

    if not keyword.strip():
        return {"ok": False, "output": "搜索关键词不能为空"}

    # 构建 grep 命令
    include = f'--include="*{file_ext}"' if file_ext else \
              '--include="*.swift" --include="*.py" --include="*.js" --include="*.ts" --include="*.json"'
    safe_kw = shlex.quote(keyword)
    cmd = (
        f'grep -rn {include} {safe_kw} "{expanded}" '
        f'--exclude-dir=".git" --exclude-dir="Pods" --exclude-dir="node_modules" '
        f'--exclude-dir=".venv" --exclude-dir="DerivedData" '
        f'2>/dev/null | head -50'
    )

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        output = stdout.decode("utf-8", errors="replace") if stdout else ""

        if not output.strip():
            return {"ok": True, "output": f"🔍 未找到 '{keyword}' 的匹配结果（路径: {search_path}）"}

        lines = output.strip().split('\n')
        header = f"🔍 搜索 '{keyword}' 找到 {len(lines)} 处匹配:\n\n"
        return {"ok": True, "output": header + output}
    except asyncio.TimeoutError:
        return {"ok": False, "output": "搜索超时（15s）"}
    except Exception as e:
        return {"ok": False, "output": f"搜索失败: {e}"}


# ══════════════════════════════════════════════════════════════════
# 动作分发 (Action Router)
# ══════════════════════════════════════════════════════════════════

async def execute_actions(
    actions: List[Dict[str, Any]],
    bridge,
    confirm: bool = False,
) -> List[Dict[str, Any]]:
    """分发并执行动作列表"""
    results = []

    for action in actions:
        t = action['type']
        p = action['params']
        result: Dict[str, Any] = {'type': t, 'ok': False, 'output': ''}

        try:
            # ── exec: 终端命令 ──
            if t == 'exec' and len(p) >= 1:
                cmd = p[0]
                timeout = int(p[1]) if len(p) > 1 and p[1].isdigit() else 60
                result = await _tool_exec(cmd, confirm=confirm, timeout=timeout)
                result['type'] = 'exec'

            # ── read_file: 文件读取 ──
            elif t == 'read_file' and len(p) >= 1:
                start = int(p[1]) if len(p) > 1 and p[1].isdigit() else None
                end   = int(p[2]) if len(p) > 2 and p[2].isdigit() else None
                result = await _tool_read_file(p[0], start, end)
                result['type'] = 'read_file'

            # ── write_file: 文件写入 ──
            elif t == 'write_file' and len(p) >= 2:
                content = '|'.join(p[1:])  # 内容中可能包含 |
                result = await _tool_write_file(p[0], content, confirm=confirm)
                result['type'] = 'write_file'

            # ── search_code: 代码搜索 ──
            elif t == 'search_code' and len(p) >= 1:
                search_path = p[1] if len(p) > 1 else '~/Desktop/GPTsora'
                file_ext    = p[2] if len(p) > 2 else ''
                result = await _tool_search_code(p[0], search_path, file_ext)
                result['type'] = 'search_code'

            # ── agent_task: 复杂任务 → OpenClaw AI ──
            elif t == 'agent_task' and len(p) >= 1:
                task_desc = '|'.join(p)
                result = await _execute_agent_task(bridge, task_desc)

            # ── 旧协议兼容 ──
            elif t in ('git', 'build', 'msg_send', 'browser_open', 'device_control',
                       'schedule_create', 'msg_platforms', 'device_list', 'schedule_list',
                       'skill_list', 'skill_invoke', 'system_info'):
                task = _legacy_action_to_task(t, p)
                result = await _execute_agent_task(bridge, task)

            else:
                result = {'type': t, 'ok': False, 'output': f'未知动作类型: {t}'}

        except Exception as e:
            result = {'type': t, 'ok': False, 'output': f'执行异常: {e}'}
            logger.error("[Intent] Action %s failed: %s", t, e, exc_info=True)

        results.append(result)
        logger.info("[Intent] %s → %s%s", t,
                    'OK' if result.get('ok') else 'FAIL',
                    f" (blocked)" if result.get('blocked') else f" (needs_confirm)" if result.get('needs_confirm') else "")

    return results


# ══════════════════════════════════════════════════════════════════
# /agent/tools HTTP 端点（供 iOS 直接调用，不走 AI 回复解析）
# ══════════════════════════════════════════════════════════════════

async def handle_agent_tools_request(body: dict) -> dict:
    """
    POST /agent/tools
    Body: {
      "tool": "exec" | "read_file" | "write_file" | "search_code",
      "params": {...},    # 工具参数
      "confirm": false,   # 高危操作确认标志
    }
    Returns: { "ok": bool, "output": str, ... }
    """
    tool = (body.get("tool") or "").strip()
    params = body.get("params") or {}
    confirm = bool(body.get("confirm", False))

    if not tool:
        return {"ok": False, "output": "缺少 tool 参数"}

    if tool == "exec":
        cmd = params.get("command", "")
        timeout = int(params.get("timeout", 60))
        if not cmd:
            return {"ok": False, "output": "缺少 command 参数"}
        return await _tool_exec(cmd, confirm=confirm, timeout=timeout)

    elif tool == "read_file":
        path = params.get("path", "")
        start = params.get("start_line")
        end = params.get("end_line")
        if not path:
            return {"ok": False, "output": "缺少 path 参数"}
        return await _tool_read_file(path,
                                     int(start) if start else None,
                                     int(end) if end else None)

    elif tool == "write_file":
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return {"ok": False, "output": "缺少 path 参数"}
        return await _tool_write_file(path, content, confirm=confirm)

    elif tool == "search_code":
        keyword = params.get("keyword", "")
        search_path = params.get("path", "~/Desktop/GPTsora")
        file_ext = params.get("ext", "")
        if not keyword:
            return {"ok": False, "output": "缺少 keyword 参数"}
        return await _tool_search_code(keyword, search_path, file_ext)

    else:
        return {"ok": False, "output": f"未知工具: {tool}。支持: exec, read_file, write_file, search_code"}


# ══════════════════════════════════════════════════════════════════
# OpenClaw AI 任务（agent_task B 类）
# ══════════════════════════════════════════════════════════════════

async def _execute_agent_task(bridge, task_description: str) -> Dict[str, Any]:
    logger.info("[AgentTask] → %s", task_description[:100])

    try:
        safe_msg = shlex.quote(task_description)
        cmd = f'openclaw agent --agent main -m {safe_msg} --json --timeout 300'

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=360)
        raw = stdout.decode('utf-8', errors='replace').strip()
        if not raw and stderr:
            raw = stderr.decode('utf-8', errors='replace').strip()

        try:
            data = json.loads(raw)
            payloads = data.get('result', {}).get('payloads', [])
            texts = [p.get('text', '') for p in payloads if p.get('text')]
            output = '\n\n'.join(texts) if texts else raw
            return {'type': 'agent_task', 'ok': data.get('status') == 'ok', 'output': output, 'task': task_description}
        except json.JSONDecodeError:
            return {'type': 'agent_task', 'ok': bool(raw), 'output': raw or '无输出', 'task': task_description}

    except asyncio.TimeoutError:
        return {'type': 'agent_task', 'ok': False,
                'output': '⏰ 任务超过6分钟，请到 OpenClaw 控制台查看进度。', 'task': task_description}
    except Exception as e:
        logger.error("[AgentTask] Failed: %s", e)
        # fallback: 直接本地执行
        simple = _task_to_simple_command(task_description)
        if simple:
            r = await _tool_exec(simple, timeout=120)
            if r.get('ok'):
                r['task'] = task_description
                return r
        return {'type': 'agent_task', 'ok': False, 'output': f'执行失败: {e}', 'task': task_description}


def _task_to_simple_command(task: str) -> Optional[str]:
    t = task.lower()
    if '系统' in t and ('健康' in t or '状态' in t):
        return 'uname -a && uptime && df -h / && vm_stat | head -5'
    if 'git' in t and 'status' in t:
        return 'cd ~/Desktop/GPTsora && git status'
    if 'git' in t and 'log' in t:
        return 'cd ~/Desktop/GPTsora && git log --oneline -10'
    if '磁盘' in t or 'disk' in t:
        return 'df -h ~'
    if '内存' in t or 'memory' in t:
        return 'vm_stat | head -8'
    return None


def _legacy_action_to_task(t: str, p: list) -> str:
    if t == 'git':
        return f"git {p[0] if p else 'status'}" + (f" commit: {p[2]}" if len(p) > 2 else "")
    elif t == 'build':
        return f"编译 {p[0] if p else '~/Desktop/GPTsora'} 并安装到手机"
    elif t == 'msg_send' and len(p) >= 3:
        return f"通过 {p[0]} 发消息给 {p[1]}: {p[2]}"
    elif t == 'browser_open' and p:
        return f"打开浏览器访问 {p[0]}"
    elif t == 'system_info':
        return "检查系统健康状态（磁盘/内存/CPU）"
    return t


# ══════════════════════════════════════════════════════════════════
# 结果格式化
# ══════════════════════════════════════════════════════════════════

def format_action_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return ''
    parts = ['\n\n---\n📡 **执行结果：**\n']
    for r in results:
        # 需要确认
        if r.get('needs_confirm'):
            parts.append(f'⚠️ **需要确认**\n> {r.get("reason", "请确认后重试")}')
            parts.append('')
            continue
        # 被拦截
        if r.get('blocked'):
            parts.append(f'⛔ **安全拦截**\n> {r.get("reason", "危险操作已被阻止")}')
            parts.append('')
            continue

        icon = '✅' if r.get('ok') else '❌'
        label = {
            'exec': '⌨️ 终端命令',
            'read_file': '📄 文件读取',
            'write_file': '✏️ 文件写入',
            'search_code': '🔍 代码搜索',
            'agent_task': '🦾 智能体任务',
        }.get(r.get('type', ''), r.get('type', '工具'))

        output = (r.get('output') or '').strip()
        parts.append(f'{icon} **{label}**')
        if output:
            if '\n' in output or len(output) > 100:
                # 超长输出截断
                if len(output) > 3000:
                    output = output[:3000] + f'\n... [输出过长，已截断 {len(output)} 字符]'
                parts.append(f'```\n{output}\n```')
            else:
                parts.append(f'> {output}')
        parts.append('')

    return '\n'.join(parts)


# ══════════════════════════════════════════════════════════════════
# 入口函数
# ══════════════════════════════════════════════════════════════════

def inject_agent_system_prompt(messages: list, bridge_connected: bool = False) -> list:
    if not bridge_connected:
        return messages
    new_messages = list(messages)
    for i, msg in enumerate(new_messages):
        if msg.get('role') in ('system', 'developer'):
            new_messages[i] = {**msg, 'content': msg['content'] + '\n\n' + AGENT_TOOLS_PROMPT}
            return new_messages
    new_messages.insert(0, {'role': 'system', 'content': '你是 ChatAGI-阿杜，一个全能 AI 助手。\n\n' + AGENT_TOOLS_PROMPT})
    return new_messages


async def process_agent_actions(
    ai_text: str, bridge, confirm: bool = False
) -> Tuple[str, str]:
    clean_text, actions = extract_actions(ai_text)
    if not actions:
        return ai_text, ''
    results = await execute_actions(actions, bridge, confirm=confirm)
    return clean_text, format_action_results(results)


def process_agent_actions_sync(
    ai_text: str, bridge, confirm: bool = False
) -> Tuple[str, str]:
    clean_text, actions = extract_actions(ai_text)
    if not actions:
        return ai_text, ''
    logger.info("[Intent] Found %d actions (sync)", len(actions))
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                execute_actions(actions, bridge, confirm=confirm), loop
            )
            results = future.result(timeout=360)
        else:
            results = loop.run_until_complete(execute_actions(actions, bridge, confirm=confirm))
    except RuntimeError:
        results = asyncio.run(execute_actions(actions, bridge, confirm=confirm))
    except Exception as e:
        logger.error("[Intent] sync failed: %s", e)
        return clean_text, f'\n\n> ⚠️ 执行失败: {e}'
    return clean_text, format_action_results(results)
