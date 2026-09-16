"""Resolve portable source paths and private artifact configuration."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def settings() -> dict:
    location = Path(os.environ.get('RESEARCH_PATHS_CONFIG', ROOT / 'paths.local.yaml'))
    if not location.is_file():
        return {}
    value = yaml.safe_load(location.read_text())
    if not isinstance(value, dict):
        raise ValueError('Local path configuration must be a mapping')
    return value


def path(value: str) -> str:
    if not isinstance(value, str) or not value.startswith('@'):
        return value
    if value == '@python':
        return sys.executable
    if value == '@remote_registered':
        if not settings().get('ssh', {}).get('host'):
            return value
        return ssh_host() + ':' + str(Path(path('@remote_data')) / '03_registered_nifti')
    if value.startswith('@private:'):
        return private_value(value.split(':', 1)[1], required=True)
    if value.startswith('@legacy:'):
        _, source, relative, field = value.split(':', 3)
        roots = settings().get('legacy_config_roots', {})
        if source not in roots:
            raise ValueError(
                f'Historical compatibility metadata is required for {source}/{relative}. '
                'Set legacy_config_roots in paths.local.yaml to a private copy of the '
                'original configuration. Values are read in memory and are not exported.'
            )
        filename = (Path(roots[source]).expanduser() / relative).resolve()
        if not filename.is_relative_to(Path(roots[source]).expanduser().resolve()):
            raise ValueError('Historical configuration escaped its configured root')
        result = yaml.safe_load(filename.read_text())
        for part in field.split('.'):
            result = result[int(part)] if isinstance(result, list) else result[part]
        return result
    config = settings()
    overrides = config.get('path_overrides', {})
    if value in overrides:
        target = Path(overrides[value]).expanduser()
        return str(target if target.is_absolute() else ROOT / target)
    alias, _, suffix = value[1:].partition('/')
    if alias == 'repo':
        root = ROOT
    else:
        defaults = {'artifacts': ROOT / 'artifacts', 'cache': ROOT / '.cache',
                    'workspace': ROOT / 'external/workspace', 'data': ROOT / 'external/data',
                    'weights': ROOT / 'external/weights', 'external': ROOT / 'external',
                    'remote_data': Path('/path/to/remote/organized'),
                    'external_storage': ROOT / 'external/storage'}
        if alias not in defaults:
            raise ValueError(f'Unknown portable path alias: {alias}')
        root = Path(config.get('paths', {}).get(alias, defaults[alias])).expanduser()
        if not root.is_absolute():
            root = ROOT / root
    return str(root / suffix) if suffix else str(root)


def resolve(value, *, resolve_assets: bool = True):
    if isinstance(value, dict):
        return {key: resolve(item, resolve_assets=resolve_assets) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, resolve_assets=resolve_assets) for item in value]
    if isinstance(value, str) and value.startswith(('@legacy:', '@private:')) and not resolve_assets:
        return value
    return path(value) if isinstance(value, str) else value


def load_yaml(stream, *, resolve_assets: bool = True):
    return resolve(yaml.safe_load(stream), resolve_assets=resolve_assets)


def ssh_arguments() -> list[str]:
    value = settings().get('ssh', {}).get('arguments', ['ssh', '-o', 'BatchMode=yes'])
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError('ssh.arguments must be a nonempty string list')
    return value


def ssh_host() -> str:
    value = settings().get('ssh', {}).get('host')
    if not isinstance(value, str) or not value.strip() or value.startswith('-'):
        raise ValueError('Set ssh.host in paths.local.yaml before remote data preparation')
    return value


def private_value(name: str, *, required: bool = False) -> str:
    values = settings().get('cohort', {})
    if isinstance(values.get(name), str) and values[name].strip():
        return values[name]
    if required:
        raise ValueError(f'Set the private cohort field {name} in paths.local.yaml')
    # Import-only placeholders also let synthetic unit tests avoid real identifiers.
    placeholders = {'mu_missing_baseline': 'PatientID_0000/Timepoint_1',
                    'mu_missing_late': 'PatientID_9999/Timepoint_3'}
    if name not in placeholders:
        raise ValueError(f'Unknown private cohort field: {name}')
    return placeholders[name]


def layout() -> dict:
    return json.loads((ROOT / 'release-layout.json').read_text())['projects']
