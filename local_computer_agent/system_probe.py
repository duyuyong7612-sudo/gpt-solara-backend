from __future__ import annotations

import getpass
import os
import platform
import shutil
import socket
from pathlib import Path
from typing import Dict, List


def _existing(paths: List[Path]) -> List[str]:
    out: List[str] = []
    for p in paths:
        try:
            if p.exists():
                out.append(str(p.resolve()))
        except Exception:
            pass
    return out


def _detect_shell() -> str:
    return os.getenv('SHELL') or os.getenv('COMSPEC') or ''


def _detect_desktop_paths(home: Path) -> Dict[str, str]:
    candidates = {
        'home': home,
        'desktop': home / 'Desktop',
        'documents': home / 'Documents',
        'downloads': home / 'Downloads',
        'applications_user': home / 'Applications',
    }
    result: Dict[str, str] = {}
    for k, p in candidates.items():
        try:
            if p.exists():
                result[k] = str(p.resolve())
        except Exception:
            pass
    return result


def _detect_installed_apps_macos(home: Path) -> Dict[str, str]:
    roots = [Path('/Applications'), home / 'Applications']
    mapping: Dict[str, str] = {}
    aliases = {
        'wechat': 'WeChat.app',
        '微信': 'WeChat.app',
        'safari': 'Safari.app',
        '浏览器': 'Safari.app',
        'finder': 'Finder.app',
        'terminal': 'Terminal.app',
        '终端': 'Terminal.app',
        'notes': 'Notes.app',
        '备忘录': 'Notes.app',
        'xcode': 'Xcode.app',
    }
    for root in roots:
        try:
            if not root.exists():
                continue
            for item in root.iterdir():
                if item.suffix.lower() == '.app':
                    mapping[item.stem.lower()] = item.stem
        except Exception:
            continue
    for alias, app_bundle in aliases.items():
        stem = app_bundle[:-4]
        if stem.lower() in mapping:
            mapping[alias.lower()] = stem
    return mapping


def _detect_installed_apps_windows() -> Dict[str, str]:
    aliases = {
        'wechat': 'WeChat',
        '微信': 'WeChat',
        'edge': 'msedge',
        '浏览器': 'msedge',
        'notepad': 'notepad',
        '记事本': 'notepad',
        'terminal': 'wt',
        '终端': 'wt',
    }
    return aliases


def _detect_installed_apps_linux() -> Dict[str, str]:
    aliases = {
        'wechat': 'wechat',
        '微信': 'wechat',
        'browser': 'xdg-open',
        '浏览器': 'xdg-open',
        'terminal': 'x-terminal-emulator',
        '终端': 'x-terminal-emulator',
    }
    return aliases


def probe_system() -> Dict:
    home = Path.home()
    system_name = platform.system().lower()
    info: Dict = {
        'platform': system_name,
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'hostname': socket.gethostname(),
        'username': getpass.getuser(),
        'shell': _detect_shell(),
        'python_executable': shutil.which('python3') or shutil.which('python') or '',
        'paths': _detect_desktop_paths(home),
        'available_roots': _existing([home, home / 'Desktop', home / 'Documents', home / 'Downloads']),
        'detected_at': __import__('time').time(),
    }

    if system_name == 'darwin':
        info['apps'] = _detect_installed_apps_macos(home)
    elif system_name == 'windows':
        info['apps'] = _detect_installed_apps_windows()
    else:
        info['apps'] = _detect_installed_apps_linux()

    workspace_aliases: Dict[str, str] = {}
    desktop = info['paths'].get('desktop')
    if desktop:
        try:
            for item in Path(desktop).iterdir():
                if item.is_dir():
                    workspace_aliases[item.name.lower()] = str(item.resolve())
                    workspace_aliases[item.name.replace(' ', '').lower()] = str(item.resolve())
        except Exception:
            pass

    for alias_key in ('backend', 'frontend', 'gptsora', 'chatagi'):
        if alias_key not in workspace_aliases:
            guess = Path(desktop or str(home)) / alias_key
            if guess.exists():
                workspace_aliases[alias_key] = str(guess.resolve())
    info['workspace_aliases'] = workspace_aliases
    return info