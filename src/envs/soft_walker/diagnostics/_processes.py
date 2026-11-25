"""Process mechanics for diagnostic stages; watchdog and failure policies stay with callers."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO


def start_logged_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: TextIO,
    process_group: bool = True,
) -> subprocess.Popen[bytes]:
    """Start one stage with merged output and the caller's session isolation policy."""
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=process_group,
    )


def kill_process(process: subprocess.Popen[bytes], *, process_group: bool = True) -> int:
    """Kill and reap a stage using its original process-group or process policy."""
    if process_group:
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    return process.wait()


def terminate_process(process: subprocess.Popen[bytes], *, process_group: bool = True) -> int:
    """Allow five seconds after termination, then kill and reap the stage."""
    if process_group:
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        return process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        return kill_process(process, process_group=process_group)
