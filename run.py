"""Run an original research entrypoint with only this release's source paths."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from research_release import ROOT, layout


def main() -> int:
    projects = layout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', choices=sorted(projects))
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Python arguments, for example: -m pytest tests/test_flow_path.py')
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply -m MODULE or a script path followed by its arguments')
    workdir = (ROOT / projects[args.project]).resolve()
    code_paths = [ROOT, workdir]
    if (workdir / 'src').is_dir() and args.project != 'pillar':
        code_paths.insert(1, workdir / 'src')
    if 'symm' in projects and args.project != 'symm':
        code_paths.append(ROOT / projects['symm'] / 'src')
    env = os.environ.copy()
    # Do not inherit PYTHONPATH entries pointing at the original research workspace.
    env['PYTHONPATH'] = os.pathsep.join(str(p.resolve()) for p in code_paths)
    env.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    env.setdefault('HF_HUB_OFFLINE', '1')
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    return subprocess.call([sys.executable, *command], cwd=workdir, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
