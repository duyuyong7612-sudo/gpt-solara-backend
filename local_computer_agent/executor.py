from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from .registry import get_registry


class LocalComputerAgent:
    def __init__(self):
        self.registry = get_registry()

    def classify_task(self, goal: str, target: str) -> str:
        g = (goal or "").strip().lower()
        t = self.registry.normalize_target(target).lower()

        # 先判定列目录
        if any(k in g for k in [
            "列出", "列表", "目录内容", "文件列表", "看看里面有什么",
            "list", "list directory", "show files", "dir"
        ]):
            return "list_directory"

        # 读文件
        if any(k in g for k in ["读取", "查看文件", "打开文件", "read file", "read"]) and ('.' in t or '/' in t):
            return "read_file"

        # 目录类
        if any(k in g for k in ["目录", "文件夹", "工程", "项目", "folder", "directory", "path"]):
            return "open_path"

        # 打开/启动
        if any(k in g for k in ["打开", "启动", "切换到", "open", "launch", "start"]):
            if self.registry.resolve_workspace(t):
                return "open_path"
            return "open_app"

        return "open_path" if self.registry.resolve_workspace(t) else "open_app"

    def _run(self, args: List[str]) -> Dict[str, Any]:
        try:
            p = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
            )
            return {
                "ok": True,
                "stdout": (p.stdout or "").strip(),
                "stderr": (p.stderr or "").strip(),
            }
        except subprocess.CalledProcessError as e:
            return {
                "ok": False,
                "error": str(e),
                "stdout": (e.stdout or "").strip() if hasattr(e, "stdout") else "",
                "stderr": (e.stderr or "").strip() if hasattr(e, "stderr") else "",
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
            }

    def _open_path_sync(self, target: str) -> Dict[str, Any]:
        resolved = self.registry.resolve_workspace(target) or target
        system_name = platform.system().lower()

        if system_name == "darwin":
            if shutil.which("open") is None:
                return {"ok": False, "error": "macos_open_not_found"}
            res = self._run(["open", resolved])
        elif system_name == "windows":
            res = self._run(["cmd", "/c", "start", "", resolved])
        else:
            opener = shutil.which("xdg-open") or shutil.which("gio")
            if not opener:
                return {"ok": False, "error": "linux_opener_not_found"}
            cmd = [opener, resolved] if Path(opener).name != "gio" else [opener, "open", resolved]
            res = self._run(cmd)

        return {
            "ok": bool(res.get("ok")),
            "source": "local",
            "engine": "local",
            "stage": "system_open_path" if res.get("ok") else "system_open_path_failed",
            "active": resolved,
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            **({} if res.get("ok") else {"error": res.get("error", "open_path_failed")}),
        }

    def _open_app_sync(self, target: str) -> Dict[str, Any]:
        app = self.registry.resolve_app(target)
        if not app:
            return {
                "ok": False,
                "source": "local",
                "engine": "local",
                "stage": "system_open_app_failed",
                "error": "app_not_resolved",
            }

        system_name = platform.system().lower()

        if system_name == "darwin":
            if shutil.which("open") is None:
                return {"ok": False, "error": "macos_open_not_found"}
            res = self._run(["open", "-a", app])
        elif system_name == "windows":
            res = self._run(["cmd", "/c", "start", "", app])
        else:
            opener = app if shutil.which(app) else None
            if not opener:
                return {"ok": False, "error": f"app_not_found:{app}"}
            res = self._run([opener])

        return {
            "ok": bool(res.get("ok")),
            "source": "local",
            "engine": "local",
            "stage": "system_open_app" if res.get("ok") else "system_open_app_failed",
            "active": app,
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            **({} if res.get("ok") else {"error": res.get("error", "open_app_failed")}),
        }

    def _list_directory_sync(self, target: str) -> Dict[str, Any]:
        resolved = self.registry.resolve_workspace(target) or target
        p = Path(resolved).expanduser()

        if not p.exists():
            return {
                "ok": False,
                "source": "local",
                "engine": "local",
                "stage": "list_directory_failed",
                "error": "path_not_found",
                "active": str(p),
            }

        if not p.is_dir():
            return {
                "ok": False,
                "source": "local",
                "engine": "local",
                "stage": "list_directory_failed",
                "error": "not_a_directory",
                "active": str(p),
            }

        try:
            items = []
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "path": str(item.resolve()),
                })

            return {
                "ok": True,
                "source": "local",
                "engine": "local",
                "stage": "list_directory",
                "active": str(p.resolve()),
                "count": len(items),
                "items": items[:300],
            }
        except Exception as e:
            return {
                "ok": False,
                "source": "local",
                "engine": "local",
                "stage": "list_directory_failed",
                "error": str(e),
                "active": str(p),
            }

    def _read_file_sync(self, target: str) -> Dict[str, Any]:
        p = Path(target).expanduser()

        if not p.exists():
            resolved = self.registry.resolve_workspace(target)
            if resolved:
                p = Path(resolved)

        if not p.exists() or not p.is_file():
            return {
                "ok": False,
                "source": "local",
                "engine": "local",
                "stage": "read_file_failed",
                "error": "file_not_found",
            }

        try:
            text = p.read_text("utf-8", errors="replace")
            return {
                "ok": True,
                "source": "local",
                "engine": "local",
                "stage": "read_file",
                "path": str(p.resolve()),
                "content": text[:40000],
                "line_count": len(text.splitlines()),
            }
        except Exception as e:
            return {
                "ok": False,
                "source": "local",
                "engine": "local",
                "stage": "read_file_failed",
                "error": str(e),
            }

    async def execute(self, goal: str, target: str) -> Dict[str, Any]:
        started = time.time()
        normalized_target = self.registry.normalize_target(target)
        task_type = self.classify_task(goal, normalized_target)

        if task_type == "open_path":
            out = await asyncio.to_thread(self._open_path_sync, normalized_target)
        elif task_type == "list_directory":
            out = await asyncio.to_thread(self._list_directory_sync, normalized_target)
        elif task_type == "read_file":
            out = await asyncio.to_thread(self._read_file_sync, normalized_target)
        else:
            out = await asyncio.to_thread(self._open_app_sync, normalized_target)

        out["duration"] = round(time.time() - started, 2)
        out["task_type"] = task_type
        out["goal"] = goal
        out["target"] = normalized_target
        return out
