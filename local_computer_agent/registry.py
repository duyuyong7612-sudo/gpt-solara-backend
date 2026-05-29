from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional

from .system_probe import probe_system

REGISTRY_PATH = Path(os.getenv('CHATAGI_LOCAL_AGENT_REGISTRY') or (Path.home() / '.chatagi' / 'local_agent_registry.json'))
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

_SPECIAL_SPACES_RE = re.compile(r'[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u3000]+')
_SUFFIX_RE = re.compile(r'(目录|文件夹|项目|工程|app|应用)$', re.I)


class LocalAgentRegistry:
    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.data: Dict = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text('utf-8'))
                return
            except Exception:
                pass
        self.data = {'system': {}, 'apps': {}, 'workspace_aliases': {}, 'memory': {}}
        self.save()

    def save(self) -> None:
        with self._lock:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), 'utf-8')

    def probe_and_persist(self) -> Dict:
        info = probe_system()
        self.data['system'] = {k: v for k, v in info.items() if k not in ('apps', 'workspace_aliases')}
        self.data['apps'] = info.get('apps', {})
        self.data['workspace_aliases'] = info.get('workspace_aliases', {})
        self.data.setdefault('memory', {})['last_probe_ok'] = True
        self.data['memory']['last_probe_platform'] = info.get('platform')
        self.save()
        return self.data

    @staticmethod
    def normalize_target(target: str) -> str:
        t = (target or '').strip()
        t = _SPECIAL_SPACES_RE.sub(' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        t = _SUFFIX_RE.sub('', t).strip()
        return t

    def resolve_workspace(self, target: str) -> Optional[str]:
        t = self.normalize_target(target).lower().replace(' ', '')
        aliases = self.data.get('workspace_aliases') or {}
        if t in aliases:
            return aliases[t]
        for k, v in aliases.items():
            kk = str(k).lower().replace(' ', '')
            if kk == t:
                return v
        p = Path(target).expanduser()
        if p.exists():
            return str(p.resolve())
        return None

    def resolve_app(self, target: str) -> Optional[str]:
        t = self.normalize_target(target).lower().replace(' ', '')
        apps = self.data.get('apps') or {}
        if t in apps:
            return apps[t]
        for k, v in apps.items():
            if str(k).lower().replace(' ', '') == t:
                return v
        raw = self.normalize_target(target)
        return raw if raw else None


_registry: Optional[LocalAgentRegistry] = None


def get_registry() -> LocalAgentRegistry:
    global _registry
    if _registry is None:
        _registry = LocalAgentRegistry()
    return _registry