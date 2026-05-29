from __future__ import annotations

import logging
from .registry import get_registry

log = logging.getLogger('local_computer_agent')


def bootstrap_local_agent() -> None:
    reg = get_registry()
    try:
        data = reg.probe_and_persist()
        platform_name = data.get('system', {}).get('platform')
        apps = len(data.get('apps') or {})
        aliases = len(data.get('workspace_aliases') or {})
        log.info('[LocalAgent] boot probe complete platform=%s apps=%s aliases=%s', platform_name, apps, aliases)
    except Exception as e:
        log.warning('[LocalAgent] boot probe failed: %s', e)
