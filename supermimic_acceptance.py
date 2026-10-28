#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-command SuperMimic acceptance on a lab cluster.

Lives next to an RLinf checkout (never inside it) and needs only the Python 3
standard library to start::

    workspace/
    |-- RLinf/                          # git clone -b feat/supermimic ...
    |-- supermimic_acceptance.py        # this file
    |-- supermimic_acceptance.yaml
    `-- acceptance_results/<run id>/    # summary.md, manifest.json, logs/, ...

    python3 supermimic_acceptance.py                 # full acceptance
    python3 supermimic_acceptance.py --mode smoke
    python3 supermimic_acceptance.py --dry-run

It installs SuperMimic into a brand-new virtual environment with RLinf's own
``requirements/install.sh``, then runs unit tests, real MuJoCo Warp training,
checkpoint/resume, evaluation, profiling, multi-GPU NCCL, multi-node Slurm,
and numeric diagnostics. Every stage is written to disk when it starts and
when it ends; the first failing required stage stops the run.

The script never runs a Git write command and never edits the RLinf checkout.
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

SCRIPT_VERSION = "1.0.0"
MANIFEST_SCHEMA = 1
MARKER_ENV = "SUPERMIMIC_ACCEPTANCE_RUN"

PASS = "PASS"
FAIL = "FAIL"
BLOCKED = "BLOCKED"
NOT_RUN = "NOT_RUN"
NOT_APPLICABLE = "NOT_APPLICABLE"
RUNNING = "RUNNING"
PENDING = "PENDING"
FINAL_STATES = {PASS, FAIL, BLOCKED, NOT_RUN, NOT_APPLICABLE}


class AcceptanceError(Exception):
    """A configuration or environment problem detected before running a stage."""


class Interrupted(Exception):
    """SIGINT or SIGTERM arrived; the run must record and stop."""

    def __init__(self, signum: int):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


# --------------------------------------------------------------------------
# Minimal YAML subset
# --------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _split_inline(text: str) -> list[str]:
    parts, buf, quote, depth = [], [], None, 0
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return parts


def parse_scalar(text: str, where: str = "") -> Any:
    """Parse one YAML scalar or flow collection of scalars."""
    s = text.strip()
    if s in {"", "~", "null", "Null", "NULL"}:
        return None
    if s in {"true", "True", "TRUE"}:
        return True
    if s in {"false", "False", "FALSE"}:
        return False
    if s.startswith("[") and s.endswith("]"):
        return [parse_scalar(p, where) for p in _split_inline(s[1:-1])]
    if s.startswith("{") and s.endswith("}"):
        result = {}
        for part in _split_inline(s[1:-1]):
            key, sep, value = part.partition(":")
            if not sep:
                raise AcceptanceError(
                    f"{where}: flow mapping entry needs 'key: value': {part!r}"
                )
            result[parse_scalar(key, where)] = parse_scalar(value, where)
        return result
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        body = s[1:-1]
        if s[0] == '"':
            body = body.replace('\\"', '"').replace("\\\\", "\\")
        else:
            body = body.replace("''", "'")
        return body
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", s):
        return float(s)
    if s[0] in "&*!|>%@`":
        raise AcceptanceError(
            f"{where}: unsupported YAML syntax {s!r} (anchors, tags, block scalars)"
        )
    return s


def parse_yaml(text: str, source: str = "<yaml>") -> dict:
    """Parse the YAML subset this tool documents: nested mappings, scalar lists.

    Anchors, tags, multi-line scalars, and lists of mappings are rejected with
    an error rather than misread.
    """
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise AcceptanceError(f"{source}:{number}: indent with spaces, not tabs")
        line = _strip_comment(raw)
        if line.strip():
            lines.append((number, len(line) - len(line.lstrip(" ")), line.strip()))

    def block(i: int, indent: int) -> tuple[Any, int]:
        if i >= len(lines):
            return {}, i
        if lines[i][2].startswith("- ") or lines[i][2] == "-":
            items = []
            while (
                i < len(lines) and lines[i][1] == indent and lines[i][2].startswith("-")
            ):
                number, _, content = lines[i]
                value = content[1:].strip()
                if ":" in value and not value.startswith(("[", "{", "'", '"')):
                    raise AcceptanceError(
                        f"{source}:{number}: lists of mappings are not supported"
                    )
                items.append(parse_scalar(value, f"{source}:{number}"))
                i += 1
            return items, i
        result = {}
        while i < len(lines) and lines[i][1] == indent:
            number, _, content = lines[i]
            if content.startswith("-"):
                raise AcceptanceError(
                    f"{source}:{number}: list item where a key was expected"
                )
            key, sep, rest = content.partition(":")
            if not sep or (rest and not rest.startswith(" ")):
                raise AcceptanceError(
                    f"{source}:{number}: expected 'key: value', got {content!r}"
                )
            key = key.strip()
            if key in result:
                raise AcceptanceError(f"{source}:{number}: duplicate key {key!r}")
            rest = rest.strip()
            i += 1
            if rest:
                result[key] = parse_scalar(rest, f"{source}:{number}")
            elif i < len(lines) and lines[i][1] > indent:
                result[key], i = block(i, lines[i][1])
            else:
                result[key] = None
        if i < len(lines) and lines[i][1] > indent:
            raise AcceptanceError(f"{source}:{lines[i][0]}: unexpected indentation")
        return result, i

    if not lines:
        return {}
    data, end = block(0, lines[0][1])
    if end != len(lines):
        raise AcceptanceError(
            f"{source}:{lines[end][0]}: indentation does not match its parent"
        )
    if not isinstance(data, dict):
        raise AcceptanceError(f"{source}: the top level must be a mapping")
    return data


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "paths": {
        "repo": None,
        "venv": "supermimic_envs/{run_id}",
        "results": "acceptance_results",
        "tools": ".supermimic_tools",
    },
    "data": {"lafan1": None, "bones_seed": None},
    "install": {
        "mode": "fresh",
        "python": None,
        "use_mirror": False,
        "no_root": True,
        "cache": "isolated",
        "cache_dir": None,
        "bootstrap_uv": True,
        "pip_index_url": None,
        "min_free_gb": 30,
        "env": {},
    },
    "scheduler": {
        "mode": "auto",
        "cpu_stages": "local",
        "poll_seconds": 30,
        "max_queue_wait_minutes": 120,
    },
    "slurm": {
        "partition": None,
        "account": None,
        "qos": None,
        "gpu_request": "gpus-per-node",
        "extra_sbatch_args": [],
        "jobs": {
            "cpu": {"cpus": None, "mem": None, "time": None},
            "gpu1": {"cpus": None, "mem": None, "time": None},
            "gpu2": {"cpus": None, "mem": None, "time": None},
            "multinode": {"cpus": None, "mem": None, "time": None},
            "learning": {"cpus": None, "mem": None, "time": None},
            "long": {"cpus": None, "mem": None, "time": None},
        },
    },
    "cluster": {"env": {}, "rlinf_timeout_minutes": 30, "nccl_debug": "INFO"},
    "smoke": {
        "num_clips": 40,
        "velocity_overrides": [
            "env.num_envs=8",
            "env.replay_length=64",
            "algorithm.wm_window_batch=64",
            "algorithm.imagine_batch=8",
            "algorithm.total_updates=2",
        ],
    },
    "evaluation": {"g2_trials": 4},
    "profiling": {
        "num_envs": [256, 1024],
        "steps": 30,
        "warmup": 20,
        "update_iters": 1,
    },
    "multi_gpu": {
        "enabled": True,
        "overrides": [
            "env.num_envs=64",
            "env.replay_length=200",
            "algorithm.wm_window_batch=256",
            "algorithm.imagine_batch=256",
        ],
        "total_updates": 50,
        "fault_timeout_minutes": 10,
    },
    "paper_scale": {
        "enabled": True,
        "min_gpu_memory_gb": 40,
        "profile_num_envs": 4096,
        "train_updates": 20,
        "two_gpu_recipe": True,
    },
    "multinode": {
        "enabled": True,
        "nodes": None,
        "gpus_per_node": None,
        "ray_port": None,
        "total_updates": 100,
        "overrides": [
            "env.num_envs=512",
            "algorithm.wm_window_batch=512",
            "algorithm.imagine_batch=512",
        ],
    },
    "learning_signal": {
        "enabled": True,
        "config": "supermimic_lafan1",
        "gpus": 1,
        "updates": 2000,
        "val_interval": 500,
        "seed": 0,
        "variants": {"baseline": []},
    },
    "long_training": {
        "enabled": False,
        "motion": {
            "config": "supermimic_lafan1",
            "gpus": None,
            "total_updates": None,
            "save_interval": None,
            "val_interval": None,
        },
        "velocity": {
            "enabled": True,
            "config": "supermimic_velocity",
            "gpus": None,
            "total_updates": None,
            "save_interval": None,
            "val_interval": None,
        },
    },
    "bones": {"enabled": False, "gpus": 1, "train_updates": 100},
    "timeouts_minutes": {
        "install": 240,
        "config": 30,
        "unit_test_file": 60,
        "smoke": 60,
        "checkpoint": 60,
        "evaluation": 120,
        "profiling": 90,
        "two_gpu": 120,
        "multi_gpu": 180,
        "paper_scale": 240,
        "multinode": 360,
        "learning_signal": 1440,
        "long_training": 4320,
    },
}


def deep_merge(base: dict, override: dict, where: str = "") -> dict:
    """Merge ``override`` into a copy of ``base``; unknown keys are errors."""
    out = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        path = f"{where}.{key}" if where else key
        if key not in out:
            free_maps = {"env", "variants"}
            parent = where.rsplit(".", 1)[-1]
            if parent in free_maps:
                out[key] = value
                continue
            raise AcceptanceError(
                f"unknown configuration key {path!r}; see supermimic_acceptance.yaml"
            )
        if isinstance(out[key], dict) and key not in {"env", "variants"}:
            if value is None:
                continue
            if not isinstance(value, dict):
                raise AcceptanceError(
                    f"configuration key {path!r} must be a mapping, got {value!r}"
                )
            out[key] = deep_merge(out[key], value, path)
        else:
            out[key] = value
    return out


def load_config(path: Path | None) -> dict:
    """Defaults merged with the YAML file, if one is given or found."""
    if path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))
    if not path.is_file():
        raise AcceptanceError(f"configuration file {path} does not exist")
    return deep_merge(
        DEFAULT_CONFIG, parse_yaml(path.read_text(encoding="utf-8"), str(path))
    )


def check_config_types(cfg: dict) -> list[str]:
    """Return problems with value types and enums (empty list when fine)."""
    problems = []

    def expect(path: str, ok: bool, what: str) -> None:
        if not ok:
            problems.append(f"{path} must be {what}")

    expect(
        "install.mode", cfg["install"]["mode"] in {"fresh", "reuse"}, "fresh or reuse"
    )
    expect(
        "install.cache",
        cfg["install"]["cache"] in {"isolated", "shared"},
        "isolated or shared",
    )
    expect(
        "scheduler.mode",
        cfg["scheduler"]["mode"] in {"auto", "local", "slurm"},
        "auto, local or slurm",
    )
    expect(
        "scheduler.cpu_stages",
        cfg["scheduler"]["cpu_stages"] in {"local", "slurm"},
        "local or slurm",
    )
    for key in ("poll_seconds", "max_queue_wait_minutes"):
        value = cfg["scheduler"][key]
        expect(
            f"scheduler.{key}",
            isinstance(value, (int, float)) and value > 0,
            "a positive number",
        )
    expect(
        "slurm.gpu_request",
        cfg["slurm"]["gpu_request"] in {"gpus-per-node", "gres"},
        "gpus-per-node or gres",
    )
    for key in ("velocity_overrides",):
        expect(
            f"smoke.{key}",
            isinstance(cfg["smoke"][key], list),
            "a list of Hydra overrides",
        )
    for section in ("multi_gpu", "multinode"):
        expect(
            f"{section}.overrides",
            isinstance(cfg[section]["overrides"], list),
            "a list",
        )
    expect(
        "profiling.num_envs",
        isinstance(cfg["profiling"]["num_envs"], list),
        "a list of integers",
    )
    expect(
        "learning_signal.variants",
        isinstance(cfg["learning_signal"]["variants"], dict)
        and all(
            isinstance(v, list) for v in cfg["learning_signal"]["variants"].values()
        ),
        "a mapping of name -> list of overrides",
    )
    for section, keys in (
        ("learning_signal", ("gpus", "updates", "val_interval")),
        ("multi_gpu", ("total_updates",)),
        ("smoke", ("num_clips",)),
    ):
        for key in keys:
            value = cfg[section][key]
            expect(
                f"{section}.{key}",
                isinstance(value, int) and value > 0,
                "a positive integer",
            )
    for key, value in cfg["timeouts_minutes"].items():
        expect(
            f"timeouts_minutes.{key}",
            isinstance(value, (int, float)) and value > 0,
            "a positive number",
        )
    for key in ("env",):
        expect(f"install.{key}", isinstance(cfg["install"][key], dict), "a mapping")
        expect(f"cluster.{key}", isinstance(cfg["cluster"][key], dict), "a mapping")
    return problems


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def make_run_dir(results: Path, sha: str) -> tuple[str, Path]:
    """Create a unique run directory; ``mkdir`` without exist_ok is the lock."""
    results.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{sha[:8] or 'nogit'}_{secrets.token_hex(2)}"
        run_dir = results / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        for sub in (
            "logs",
            "metrics",
            "evaluations",
            "checkpoints",
            "env",
            "scripts",
            "slurm",
        ):
            (run_dir / sub).mkdir()
        return run_id, run_dir
    raise AcceptanceError(f"could not create a unique run directory under {results}")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    sid: str
    name: str
    title: str
    resource: str  # none | cpu | gpu1 | gpu2 | multinode | learning | long
    modes: frozenset
    deps: tuple = ()
    required: bool = True


_BOTH = frozenset({"smoke", "full"})
_FULL = frozenset({"full"})
STAGES: list[StageSpec] = [
    StageSpec("00", "preflight", "Preflight、Git 版本、硬件与资源记录", "none", _BOTH),
    StageSpec(
        "01", "install", "全新隔离环境安装与依赖声明完整性", "cpu", _BOTH, ("00",)
    ),
    StageSpec(
        "02",
        "config",
        "依赖导入、环境隔离、Hydra 配置与数据检查",
        "cpu",
        _BOTH,
        ("01",),
    ),
    StageSpec(
        "03", "unit_tests", "SuperMimic 单元测试与适用公共回归", "gpu1", _BOTH, ("02",)
    ),
    StageSpec(
        "04", "smoke_train", "真实单 GPU MuJoCo Warp Smoke 训练", "gpu1", _BOTH, ("03",)
    ),
    StageSpec(
        "05",
        "checkpoint_resume",
        "Checkpoint 保存、校验、强杀与 Resume",
        "gpu1",
        _BOTH,
        ("04",),
    ),
    StageSpec(
        "06",
        "evaluation",
        "Motion / Velocity 独立评测与 only_eval",
        "gpu1",
        _BOTH,
        ("04",),
    ),
    StageSpec("07", "profiling", "单 GPU 显存与吞吐 Profiling", "gpu1", _FULL, ("04",)),
    StageSpec(
        "08",
        "nccl_2gpu",
        "双 GPU 真实 NCCL、参数同步与计数一致",
        "gpu2",
        _FULL,
        ("04",),
    ),
    StageSpec(
        "09",
        "multigpu_resume_fault",
        "多 GPU Checkpoint、Resume、换卡数与单 rank 故障",
        "gpu2",
        _FULL,
        ("08",),
    ),
    StageSpec(
        "10", "paper_scale", "论文规模单 GPU 显存与吞吐可行性", "gpu1", _FULL, ("07",)
    ),
    StageSpec("10b", "paper_2gpu", "论文双卡配方可行性", "gpu2", _FULL, ("08", "10")),
    StageSpec(
        "11", "multinode", "多节点 Ray/NCCL/Slurm 运行", "multinode", _FULL, ("08",)
    ),
    StageSpec(
        "12", "learning_signal", "短期学习信号与数值诊断", "learning", _FULL, ("04",)
    ),
    StageSpec(
        "13",
        "long_training",
        "长时间 Motion / Velocity 训练与评测",
        "long",
        _FULL,
        ("12",),
    ),
    StageSpec(
        "13b", "bones_seed", "BONES-SEED 真实数据训练与评测", "learning", _FULL, ("04",)
    ),
    StageSpec("14", "summary", "最终结果汇总", "none", _BOTH),
]
STAGE_BY_ID = {s.sid: s for s in STAGES}
EXEC_ORDER = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "10",
    "08",
    "09",
    "10b",
    "11",
    "12",
    "13",
    "13b",
    "14",
]


@dataclass
class StageResult:
    status: str
    message: str = ""
    exit_code: int | None = None
    results: dict = field(default_factory=dict)
    diagnosis: list = field(default_factory=list)


# --------------------------------------------------------------------------
# Process control
# --------------------------------------------------------------------------


@dataclass
class CommandResult:
    code: int | None
    seconds: float
    timed_out: bool = False
    killed_by_watch: bool = False
    tail: list = field(default_factory=list)
    output: str = ""


_ACTIVE_GROUPS: set[int] = set()
_SLURM_JOBS: set[str] = set()


def _ancestors() -> set[int]:
    """This process and its parents; a Slurm worker's batch shell carries the marker too."""
    chain, pid = set(), os.getpid()
    while pid > 1 and pid not in chain:
        chain.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                pid = int(handle.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return chain


def marker_pids(run_id: str) -> list[int]:
    """PIDs whose environment carries this run's marker (only our processes)."""
    needle = f"{MARKER_ENV}={run_id}".encode()
    found = []
    spared = _ancestors()
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not entry.isdigit() or int(entry) in spared:
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as handle:
                if needle in handle.read().split(b"\0"):
                    found.append(int(entry))
        except OSError:
            continue
    return found


def cleanup_run_processes(run_id: str, grace: float = 10.0) -> list[int]:
    """Stop process groups we started and every process tagged with our run ID.

    Ray daemons started by our drivers inherit the tag, so they are found too.
    Nothing without the tag is touched (no global ``ray stop`` or ``pkill``).
    """
    for pgid in list(_ACTIVE_GROUPS):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            _ACTIVE_GROUPS.discard(pgid)
    pids = marker_pids(run_id)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + grace
    while time.time() < deadline and marker_pids(run_id):
        time.sleep(0.5)
    leftover = marker_pids(run_id)
    for pid in leftover:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for pgid in list(_ACTIVE_GROUPS):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        _ACTIVE_GROUPS.discard(pgid)
    time.sleep(0.2)
    return pids


def _kill_group(proc: subprocess.Popen, hard: bool = False) -> None:
    """SIGTERM then SIGKILL one process group we started (``hard``: SIGKILL only)."""
    steps = (
        ((signal.SIGKILL, 30),)
        if hard
        else ((signal.SIGTERM, 10), (signal.SIGKILL, 30))
    )
    for sig, wait in steps:
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict,
    log: Path,
    timeout_s: float,
    watch: Callable[[str], bool] | None = None,
    watch_every: float = 2.0,
    echo: bool = True,
) -> CommandResult:
    """Run ``cmd`` in its own process group, streaming output into ``log``.

    ``watch`` sees the output so far every ``watch_every`` seconds; returning
    True kills the group (used for deliberate kill -9 tests). On timeout the
    whole group is killed.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as handle:
        handle.write(f"\n$ {shlex.join(cmd)}\n# cwd: {cwd}\n# start: {now_iso()}\n")
        handle.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            handle.write(f"# could not start: {error}\n")
            return CommandResult(
                code=127, seconds=0.0, tail=[str(error)], output=str(error)
            )
        _ACTIVE_GROUPS.add(proc.pid)
        chunks: list[str] = []
        lock = threading.Lock()

        def pump() -> None:
            for line in iter(proc.stdout.readline, ""):
                with lock:
                    chunks.append(line)
                    handle.write(line)
                    handle.flush()
                if echo:
                    sys.stdout.write(line)
                    sys.stdout.flush()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = killed = False
        last_watch = 0.0
        try:
            while proc.poll() is None:
                if time.time() - start > timeout_s:
                    timed_out = True
                    break
                if watch and time.time() - last_watch >= watch_every:
                    last_watch = time.time()
                    with lock:
                        so_far = "".join(chunks)
                    if watch(so_far):
                        killed = True
                        break
                time.sleep(0.2)
        except BaseException:
            _kill_group(proc)
            raise
        finally:
            if timed_out or killed:
                _kill_group(proc, hard=killed)
            _ACTIVE_GROUPS.discard(proc.pid)
        reader.join(timeout=30)
        if not reader.is_alive():
            proc.stdout.close()
        seconds = time.time() - start
        code = proc.returncode
        note = (
            "timeout" if timed_out else ("killed by test" if killed else f"exit {code}")
        )
        with lock:
            handle.write(f"\n# end: {now_iso()} ({note}, {seconds:.1f} s)\n")
    output = "".join(chunks)
    return CommandResult(
        code=code,
        seconds=seconds,
        timed_out=timed_out,
        killed_by_watch=killed,
        tail=output.splitlines()[-60:],
        output=output,
    )


# --------------------------------------------------------------------------
# Log parsing and diagnosis
# --------------------------------------------------------------------------

_METRIC_LINE = re.compile(r"supermimic (metrics|eval) (\{.*\})")


def parse_metric_dict(text: str) -> dict | None:
    """Parse a logged ``{...}`` metrics dict; nan/inf become float values."""
    safe = re.sub(
        r"(?<![\w'\"])(-?)(nan|inf)(?![\w'\"])",
        lambda m: f"'__{m.group(1)}{m.group(2)}__'",
        text,
    )
    try:
        raw = ast.literal_eval(safe)
    except (ValueError, SyntaxError):
        return None
    out = {}
    for key, value in raw.items():
        if isinstance(value, str) and value.startswith("__") and value.endswith("__"):
            value = float(value.strip("_"))
        out[key] = value
    return out


def parse_training_log(output: str) -> dict:
    """Metrics and eval dicts the SuperMimic runner logged, in order."""
    metrics, evals = [], []
    for line in output.splitlines():
        match = _METRIC_LINE.search(line)
        if not match:
            continue
        parsed = parse_metric_dict(match.group(2))
        if parsed is None:
            continue
        (metrics if match.group(1) == "metrics" else evals).append(parsed)
    return {"metrics": metrics, "evals": evals}


DIAGNOSES = [
    (
        r"No module named '([\w.]+)'",
        "缺少 Python 模块 {0}：若出现在新环境中，说明依赖声明不完整，需要修复 requirements 后用全新环境重测",
    ),
    (
        r"(\S+) \S+ (?:has requirement|requires) (.+?), but you have (.+)",
        "依赖冲突：{0} 需要 {1}，实际为 {2}",
    ),
    (
        r"No solution found when resolving dependencies",
        "uv 无法解出依赖：依赖声明之间存在冲突",
    ),
    (
        r"CUDA out of memory|OutOfMemoryError",
        "GPU 显存不足（CUDA OOM）：降低规模或使用显存更大的 GPU",
    ),
    (
        r"NCCL.*(?:timeout|Watchdog|timed out)|ProcessGroupNCCL.*Timeout",
        "NCCL 通信超时：检查网络接口（NCCL_SOCKET_IFNAME）、防火墙与各 rank 是否存活",
    ),
    (
        r"NCCL error|ncclInternalError|ncclSystemError|ncclUnhandledCudaError",
        "NCCL 错误：查看 NCCL_DEBUG=INFO 输出定位接口或驱动问题",
    ),
    (
        r"CUDA error|cudaError|CUDA driver version is insufficient",
        "CUDA 运行时错误：检查驱动版本与 torch/warp 的 CUDA 构建是否匹配",
    ),
    (
        r"node running low on memory|memory usage threshold",
        "Ray 内存监控因主机内存不足杀掉了 worker",
    ),
    (
        r"Killed\b|exit code -9|signal 9",
        "进程被 SIGKILL 杀死（可能是系统 OOM killer 或作业超时）",
    ),
    (r"(\w+(?:Error|Exception)): (.+)", "Python 异常 {0}: {1}"),
]


def diagnose(text: str) -> list[str]:
    """Likely causes found in a log tail, most specific first."""
    found = []
    for pattern, template in DIAGNOSES:
        for match in re.finditer(pattern, text):
            message = template.format(*[g.strip()[:200] for g in match.groups()])
            if message not in found:
                found.append(message)
            if len(found) >= 6:
                return found
    return found


def all_finite(values: Iterable) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


# --------------------------------------------------------------------------
# Run state and reports
# --------------------------------------------------------------------------


class RunState:
    """``manifest.json`` plus ``summary.md``, rewritten atomically on every change."""

    def __init__(self, run_dir: Path, manifest: dict):
        self.run_dir = run_dir
        self.manifest = manifest

    @classmethod
    def create(cls, run_dir: Path, run_id: str, header: dict) -> RunState:
        stages = {}
        for spec in STAGES:
            stages[spec.sid] = {
                "id": spec.sid,
                "name": spec.name,
                "title": spec.title,
                "resource": spec.resource,
                "status": PENDING,
                "message": "",
                "exit_code": None,
                "seconds": None,
                "started_at": None,
                "finished_at": None,
                "log": f"logs/{spec.sid}_{spec.name}.log",
                "placement": None,
                "commands": [],
                "results": {},
                "diagnosis": [],
            }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "script_version": SCRIPT_VERSION,
            "run_id": run_id,
            "started_at": now_iso(),
            "finished_at": None,
            "exit_code": None,
            "first_failure": None,
            "stage_order": EXEC_ORDER,
            "stages": stages,
            "slurm_jobs": [],
            "conclusions": {},
            **header,
        }
        state = cls(run_dir, manifest)
        state.save()
        return state

    @classmethod
    def load(cls, run_dir: Path) -> RunState:
        return cls(
            run_dir, json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        )

    def reload(self) -> None:
        self.manifest = json.loads(
            (self.run_dir / "manifest.json").read_text(encoding="utf-8")
        )

    def save(self) -> None:
        atomic_write(
            self.run_dir / "manifest.json",
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
        )
        atomic_write(self.run_dir / "summary.md", render_summary(self.manifest))

    def stage(self, sid: str) -> dict:
        return self.manifest["stages"][sid]

    def set(self, sid: str, status: str, message: str = "", **extra) -> None:
        record = self.stage(sid)
        record["status"] = status
        record["message"] = message
        record.update(extra)
        if (
            status in FINAL_STATES
            and record["finished_at"] is None
            and status in {NOT_RUN, NOT_APPLICABLE, BLOCKED}
        ):
            record["finished_at"] = now_iso()
        self.save()

    def start(self, sid: str, placement: str) -> None:
        record = self.stage(sid)
        record.update(
            status=RUNNING,
            started_at=now_iso(),
            started_epoch=time.time(),
            placement=placement,
            message="运行中",
        )
        self.save()

    def add_command(
        self, sid: str, cmd: list[str], cwd: Path, result: CommandResult
    ) -> None:
        self.stage(sid)["commands"].append(
            {
                "cmd": shlex.join(cmd),
                "cwd": str(cwd),
                "exit_code": result.code,
                "seconds": round(result.seconds, 1),
                "timed_out": result.timed_out,
                "killed_by_test": result.killed_by_watch,
            }
        )
        self.save()

    def finish(self, sid: str, result: StageResult) -> None:
        record = self.stage(sid)
        record.update(
            status=result.status,
            message=result.message,
            exit_code=result.exit_code,
            finished_at=now_iso(),
            results=result.results,
            diagnosis=result.diagnosis,
        )
        if record.get("started_epoch"):
            record["seconds"] = round(time.time() - record["started_epoch"], 1)
        if result.status == FAIL and not self.manifest.get("first_failure"):
            self.manifest["first_failure"] = sid
        self.save()


STATUS_ICON = {
    PASS: "✅",
    FAIL: "❌",
    BLOCKED: "⛔",
    NOT_RUN: "⏸",
    NOT_APPLICABLE: "➖",
    RUNNING: "🔄",
    PENDING: "…",
}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
        return text if len(text) <= 160 else text[:157] + "..."
    return str(value)


def render_summary(m: dict) -> str:
    """The human report; regenerated from the manifest after every change."""
    lines = []
    if m.get("simulation"):
        lines += [
            f"> **模拟运行（scenario: `{m['simulation']}`）**：阶段内容是脚本自测用的 Mock，"
            "只证明验收脚本本身的编排、落盘与失败处理，**不是**任何真实集群验收结果。",
            "",
        ]
    lines += [f"# SuperMimic 集群验收报告 `{m['run_id']}`", ""]
    git = m.get("git", {})
    env = m.get("environment", {})
    sched = m.get("scheduler", {})
    lines += [
        "## 运行信息",
        "",
        f"- 模式：`{m.get('mode')}`；调度：`scheduler.mode={sched.get('mode')}`，"
        f"启动时在 Slurm 作业内：{sched.get('in_allocation')}",
        f"- 驱动：{env.get('driver') or '未检测'}",
        f"- 开始：{m.get('started_at')}；结束：{m.get('finished_at') or '运行中'}；退出码：{m.get('exit_code')}",
        f"- 命令：`{m.get('command_line', '')}`",
        f"- 脚本版本：{m.get('script_version')}；配置文件：`{m.get('config_file')}`",
        f"- 仓库：`{m.get('repo')}`；结果目录：`{m.get('run_dir')}`",
        f"- Git：commit `{git.get('commit', '未知')}`，分支 `{git.get('branch', '未知')}`，"
        f"未提交改动 {git.get('dirty_files', '未知')} 个",
        f"- 主机：{env.get('hostname', '?')}；CPU {env.get('cpu_count', '?')} 核；内存 {env.get('mem_gb', '?')} GB；"
        f"系统 Python {env.get('system_python', '?')}",
        f"- GPU（启动节点可见）：{env.get('gpus', '未检测')}",
        "",
    ]
    install = m["stages"].get("01", {})
    if (install.get("results") or {}).get("install_command"):
        r = install["results"]
        lines += [
            "## 环境安装",
            "",
            f"- 状态：{install['status']}；venv：`{r.get('venv')}`；缓存模式：`{r.get('cache_mode')}`（`{r.get('cache_dir')}`）",
            f"- 安装命令：`{r.get('install_command', '')}`",
            f"- 关键版本：{_fmt(r.get('versions', {}))}",
            "",
        ]
    lines += [
        "## 阶段结果",
        "",
        "| 阶段 | 内容 | 状态 | 退出码 | 耗时(s) | 结果 | 日志 |",
        "|---|---|---|---|---|---|---|",
    ]
    for sid in m.get("stage_order", list(m["stages"])):
        s = m["stages"][sid]
        message = s.get("message", "").replace("|", "\\|").replace("\n", " ")
        if len(message) > 140:
            message = message[:137] + "..."
        lines.append(
            f"| {sid} | {s['title']} | {STATUS_ICON.get(s['status'], '')} {s['status']} | "
            f"{'' if s.get('exit_code') is None else s['exit_code']} | {'' if s.get('seconds') is None else s['seconds']} | {message} | `{s['log']}` |"
        )
    lines.append("")
    key_lines = []
    for sid in m.get("stage_order", list(m["stages"])):
        s = m["stages"][sid]
        results = {
            k: v for k, v in (s.get("results") or {}).items() if k not in {"versions"}
        }
        if results:
            key_lines.append(f"- **{sid} {s['title']}**")
            for key, value in list(results.items())[:14]:
                key_lines.append(f"  - {key}: {_fmt(value)}")
    if key_lines:
        lines += ["## 关键指标与产物", "", *key_lines, ""]
    if m.get("script_error"):
        lines += [
            "## 验收脚本内部错误",
            "",
            "```",
            m["script_error"][-3000:],
            "```",
            "",
        ]
    first = m.get("first_failure")
    if first:
        s = m["stages"][first]
        lines += [
            "## 第一个失败阶段",
            "",
            f"- 阶段 {first}（{s['title']}）：{s['message']}",
            f"- 日志：`{s['log']}`",
        ]
        if s.get("commands"):
            last = s["commands"][-1]
            cmd = last["cmd"].splitlines()[0]
            cmd = (
                cmd
                if len(cmd) <= 300 and "\n" not in last["cmd"]
                else cmd[:300] + " …（完整命令见 manifest.json）"
            )
            lines.append(
                f"- 失败命令：`{cmd}`（退出码 {last['exit_code']}，超时 {last['timed_out']}，工作目录 `{last['cwd']}`）"
            )
        for item in s.get("diagnosis") or []:
            lines.append(f"- 可能原因：{item}")
        lines.append("")

    def listed(status: str, with_reason: bool) -> str:
        sids = [sid for sid in m["stages"] if m["stages"][sid]["status"] == status]
        if not sids:
            return "无"
        if not with_reason:
            return ", ".join(sids)
        return "；".join(
            "{}（{}）".format(sid, m["stages"][sid]["message"]) for sid in sids
        )

    lines += [
        "## 覆盖范围",
        "",
        "- 未执行（NOT_RUN）：" + listed(NOT_RUN, False),
        "- 被阻塞（BLOCKED）：" + listed(BLOCKED, True),
        "- 不适用或用户排除（NOT_APPLICABLE）：" + listed(NOT_APPLICABLE, True),
        "",
    ]
    conclusions = m.get("conclusions") or {}
    if conclusions:
        lines += ["## 结论（五项分别判断）", ""]
        for key, label in (
            ("install", "1. 环境安装成功"),
            ("engineering", "2. 工程与功能验收通过"),
            ("numerics", "3. 数值训练过程正常"),
            ("improvement", "4. 策略性能改善"),
            ("paper", "5. 论文结果复现"),
        ):
            lines.append(f"- **{label}**：{conclusions.get(key, '未判断')}")
        lines.append("")
    advice = m.get("advice") or []
    if advice:
        lines += ["## 下一步建议", "", *[f"- {a}" for a in advice], ""]
    repo_check = m.get("repo_unchanged")
    if repo_check is not None:
        lines += [
            "## 仓库卫生",
            "",
            f"- 运行前后 `git status --porcelain` 一致：{repo_check}",
            "",
        ]
    lines += [
        "## 执行参数（生效配置）",
        "",
        "```json",
        json.dumps(m.get("config", {}), indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Context: paths, environments, resources
# --------------------------------------------------------------------------

SCRUBBED_ENV = {
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "VIRTUAL_ENV",
    "PIP_REQUIRE_VIRTUALENV",
    "PIP_TARGET",
    "PIP_PREFIX",
    "UV_PROJECT_ENVIRONMENT",
}


def _is_conda_bin(path: str) -> bool:
    lowered = path.lower()
    return any(
        tag in lowered for tag in ("conda", "mamba", "miniforge")
    ) and lowered.rstrip("/").endswith(("/bin", "/condabin"))


def resolve_path(value: Any, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (base / path)


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class Context:
    """Everything a stage needs: config, paths, environments, persistence."""

    def __init__(
        self,
        args: argparse.Namespace,
        cfg: dict,
        workspace: Path,
        repo: Path,
        run_id: str,
        run_dir: Path,
        state: RunState,
    ):
        self.args = args
        self.cfg = cfg
        self.workspace = workspace
        self.repo = repo
        self.run_id = run_id
        self.run_dir = run_dir
        self.state = state
        self.simulate = getattr(args, "simulate", None)
        self.inside_job = bool(getattr(args, "worker", False))
        self.venv = resolve_path(
            str(cfg["paths"]["venv"]).replace("{run_id}", run_id), workspace
        )
        self.tools = resolve_path(cfg["paths"]["tools"], workspace)
        self.lafan1 = resolve_path(cfg["data"]["lafan1"], workspace)
        self.bones = resolve_path(cfg["data"]["bones_seed"], workspace)
        self.verbose = bool(getattr(args, "verbose", False))
        self.timeout_scale = float(
            os.environ.get("SUPERMIMIC_ACCEPTANCE_TIMEOUT_SCALE", "1")
        )

    # ---- paths
    @property
    def venv_python(self) -> Path:
        return self.venv / "bin" / "python"

    def log(self, sid: str) -> Path:
        return self.run_dir / self.state.stage(sid)["log"]

    def timeout(self, key: str) -> float:
        return float(self.cfg["timeouts_minutes"][key]) * 60.0 * self.timeout_scale

    # ---- environments
    def ray_tmpdir(self) -> str:
        # Ray creates socket paths about 70 characters below this directory and
        # Linux caps them at 107, so it must stay within 38 characters.
        name = f"smacc_{self.run_id[9:15]}{self.run_id[-4:]}"
        base = os.environ.get("TMPDIR", "").rstrip("/")
        if base and len(base) + len(name) + 1 <= 38 and Path(base).is_dir():
            return f"{base}/{name}"
        return f"/tmp/{name}"

    def clean_env(self) -> dict:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in SCRUBBED_ENV and not k.startswith("CONDA")
        }
        env.pop("RAY_ADDRESS", None)
        env["PATH"] = os.pathsep.join(
            p
            for p in env.get("PATH", "").split(os.pathsep)
            if p and not _is_conda_bin(p)
        )
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["RAY_TMPDIR"] = self.ray_tmpdir()
        env[MARKER_ENV] = self.run_id
        for key, value in (self.cfg["cluster"]["env"] or {}).items():
            env[str(key)] = str(value)
        return env

    def tools_bin(self) -> Path:
        """Holds only a ``uv`` link, so the tools venv's python never shadows anything."""
        return self.tools / "uvbin"

    def install_env(self) -> dict:
        env = self.clean_env()
        env["PATH"] = f"{self.tools_bin()}{os.pathsep}{env['PATH']}"
        cache = self.cache_dir()
        env["UV_CACHE_DIR"] = str(cache)
        if self.cfg["install"]["cache"] == "isolated":
            env["UV_PYTHON_INSTALL_DIR"] = str(cache / "python")
        for key, value in (self.cfg["install"]["env"] or {}).items():
            env[str(key)] = str(value)
        return env

    def cache_dir(self) -> Path:
        if self.cfg["install"]["cache"] == "shared":
            shared = resolve_path(self.cfg["install"]["cache_dir"], self.workspace)
            if shared is None:
                raise AcceptanceError("install.cache=shared needs install.cache_dir")
            return shared
        return self.workspace / ".supermimic_cache" / self.run_id

    def venv_env(self, gpus: int = 1) -> dict:
        env = self.clean_env()
        env["PATH"] = (
            f"{self.venv / 'bin'}{os.pathsep}{self.tools_bin()}{os.pathsep}{env['PATH']}"
        )
        env["VIRTUAL_ENV"] = str(self.venv)
        env["PYTHONPATH"] = str(self.repo)
        env["REPO_PATH"] = str(self.repo)
        env["EMBODIED_PATH"] = str(self.repo / "examples" / "embodiment")
        env["EVALUATIONS_PATH"] = str(self.repo / "evaluations")
        # env.setdefault("MUJOCO_GL", "egl")
        # env.setdefault("PYOPENGL_PLATFORM", "egl")
        
        # SuperMimic uses MuJoCo Warp for physics, not OpenGL rendering.
        # Do not force EGL on headless cluster nodes without EGL support.
        env["MUJOCO_GL"] = "disabled"
        env.pop("PYOPENGL_PLATFORM", None)

        env["HYDRA_FULL_ERROR"] = "1"
        env["RLINF_TIMEOUT"] = str(int(self.cfg["cluster"]["rlinf_timeout_minutes"]))
        if gpus > 1 and self.cfg["cluster"]["nccl_debug"]:
            env.setdefault("NCCL_DEBUG", str(self.cfg["cluster"]["nccl_debug"]))
        return env

    # ---- commands
    def run(
        self,
        sid: str,
        cmd: list[str],
        *,
        timeout_key: str,
        env: dict | None = None,
        cwd: Path | None = None,
        watch: Callable[[str], bool] | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        cwd = cwd or self.repo
        env = env if env is not None else self.venv_env()
        result = run_command(
            cmd,
            cwd=cwd,
            env=env,
            log=self.log(sid),
            timeout_s=timeout_s if timeout_s is not None else self.timeout(timeout_key),
            watch=watch,
            echo=self.verbose,
        )
        self.state.add_command(sid, cmd, cwd, result)
        return result

    def py(
        self,
        sid: str,
        script: str,
        *args: str,
        timeout_key: str = "config",
        gpus: int = 1,
    ) -> CommandResult:
        """Run a probe script (written under the run dir) with the new venv's Python."""
        path = (
            self.run_dir
            / "scripts"
            / f"{sid}_{hashlib.sha1(script.encode()).hexdigest()[:8]}.py"
        )
        if not path.exists():
            atomic_write(path, script)
        return self.run(
            sid,
            [str(self.venv_python), str(path), *args],
            timeout_key=timeout_key,
            env=self.venv_env(gpus),
        )

    def note(self, text: str) -> None:
        print(f"[supermimic-acceptance] {text}", flush=True)


# --------------------------------------------------------------------------
# Resources: GPUs, Slurm, placement
# --------------------------------------------------------------------------


def detect_gpus(env: dict | None = None) -> list[dict]:
    """GPUs this process may use (nvidia-smi filtered by CUDA_VISIBLE_DEVICES)."""
    env = env or os.environ
    smi = shutil.which("nvidia-smi", path=env.get("PATH"))
    if not smi:
        return []
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": parts[0],
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_gb": round(float(parts[3]) / 1024, 1),
                    "driver": parts[4],
                }
            )
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        wanted = [
            v.strip() for v in visible.split(",") if v.strip() and v.strip() != "-1"
        ]
        gpus = [
            g
            for g in gpus
            if g["index"] in wanted
            or g["uuid"] in wanted
            or any(g["uuid"].startswith(w) for w in wanted)
        ]
    return gpus


def slurm_info(env: dict | None = None) -> dict:
    env = env or os.environ
    path = env.get("PATH")
    return {
        "in_allocation": bool(env.get("SLURM_JOB_ID")),
        "job_id": env.get("SLURM_JOB_ID"),
        "allocation_nodes": int(
            env.get("SLURM_JOB_NUM_NODES", env.get("SLURM_NNODES", "0")) or 0
        ),
        "sbatch": shutil.which("sbatch", path=path),
        "squeue": shutil.which("squeue", path=path),
        "sacct": shutil.which("sacct", path=path),
        "scancel": shutil.which("scancel", path=path),
        "srun": shutil.which("srun", path=path),
    }


def stage_gpus(cfg: dict, sid: str) -> tuple[int, int]:
    """(GPUs per node, nodes) a stage needs."""
    spec = STAGE_BY_ID[sid]
    if spec.resource in {"none", "cpu"}:
        return 0, 1
    if spec.resource == "gpu1":
        return 1, 1
    if spec.resource == "gpu2":
        return 2, 1
    if spec.resource == "multinode":
        return int(cfg["multinode"]["gpus_per_node"] or 0), int(
            cfg["multinode"]["nodes"] or 0
        )
    if sid == "13b":
        return int(cfg["bones"]["gpus"]), 1
    if spec.resource == "learning":
        return int(cfg["learning_signal"]["gpus"]), 1
    if spec.resource == "long":
        lt = cfg["long_training"]
        gpus = [int(lt["motion"]["gpus"] or 0)]
        if lt["velocity"]["enabled"]:
            gpus.append(int(lt["velocity"]["gpus"] or 0))
        return max(gpus), 1
    return 0, 1


def job_key(sid: str) -> str:
    resource = STAGE_BY_ID[sid].resource
    return {
        "cpu": "cpu",
        "gpu1": "gpu1",
        "gpu2": "gpu2",
        "learning": "learning",
        "long": "long",
        "multinode": "multinode",
    }[resource]


def missing_slurm_keys(cfg: dict, key: str) -> list[str]:
    missing = [] if cfg["slurm"]["partition"] else ["slurm.partition"]
    job = cfg["slurm"]["jobs"][key]
    missing += [
        f"slurm.jobs.{key}.{k}" for k in ("cpus", "mem", "time") if not job.get(k)
    ]
    return missing


def decide_placement(
    ctx: Context, sid: str, gpus_visible: list[dict], slurm: dict
) -> tuple[str, str]:
    """Where a stage runs: ("local" | "slurm" | "blocked", reason)."""
    spec = STAGE_BY_ID[sid]
    if spec.resource == "none" or ctx.inside_job:
        return "local", "当前节点" if spec.resource == "none" else "当前 Slurm 作业内"
    if ctx.simulate and ctx.cfg["scheduler"]["mode"] != "slurm":
        return "local", "模拟"
    if spec.resource == "cpu":
        if ctx.cfg["scheduler"]["cpu_stages"] == "local":
            return "local", "当前节点（scheduler.cpu_stages=local）"
        if not slurm["sbatch"]:
            return "blocked", "scheduler.cpu_stages=slurm 但找不到 sbatch"
        missing = missing_slurm_keys(ctx.cfg, "cpu")
        return (
            ("blocked", "缺少配置：" + ", ".join(missing))
            if missing
            else ("slurm", "提交 CPU 作业")
        )
    need, nodes = stage_gpus(ctx.cfg, sid)
    mode = ctx.cfg["scheduler"]["mode"]
    if spec.resource == "multinode":
        if nodes < 2 or need < 1:
            return (
                "blocked",
                "多节点需要 multinode.nodes >= 2 与 multinode.gpus_per_node >= 1",
            )
        if slurm["in_allocation"]:
            if slurm["allocation_nodes"] >= nodes and slurm["srun"]:
                return (
                    "local",
                    f"当前 Slurm Allocation 内（{slurm['allocation_nodes']} 节点）",
                )
            return (
                "blocked",
                f"当前 Allocation 只有 {slurm['allocation_nodes']} 个节点，需要 {nodes}",
            )
        if mode != "local" and slurm["sbatch"]:
            missing = missing_slurm_keys(ctx.cfg, "multinode")
            return (
                ("blocked", "缺少配置：" + ", ".join(missing))
                if missing
                else ("slurm", "提交多节点作业")
            )
        return "blocked", "多节点需要 Slurm（未找到 sbatch，或 scheduler.mode=local）"
    if need < 1:
        return "blocked", f"阶段 {sid} 的 GPU 数量未配置"
    if mode == "local" or slurm["in_allocation"]:
        if len(gpus_visible) >= need:
            return "local", f"当前节点可见 {len(gpus_visible)} 个 GPU"
        return "blocked", f"需要 {need} 个 GPU，当前节点可见 {len(gpus_visible)} 个"
    if slurm["sbatch"]:
        missing = missing_slurm_keys(ctx.cfg, job_key(sid))
        if missing:
            return "blocked", "需要通过 Slurm 申请 GPU，但缺少配置：" + ", ".join(
                missing
            )
        return "slurm", f"提交 Slurm 作业（{need} GPU）"
    if len(gpus_visible) >= need:
        return "local", f"无 Slurm，当前节点可见 {len(gpus_visible)} 个 GPU"
    return (
        "blocked",
        f"需要 {need} 个 GPU：当前节点只有 {len(gpus_visible)} 个，且没有 Slurm",
    )


def feature_exclusion(cfg: dict, sid: str) -> str | None:
    """Reason a stage is turned off by the YAML, or None."""
    if sid in {"08", "09", "10b"} and not cfg["multi_gpu"]["enabled"]:
        return "multi_gpu.enabled=false"
    if sid in {"10", "10b"} and not cfg["paper_scale"]["enabled"]:
        return "paper_scale.enabled=false"
    if sid == "10b" and not cfg["paper_scale"]["two_gpu_recipe"]:
        return "paper_scale.two_gpu_recipe=false"
    if sid == "11" and not cfg["multinode"]["enabled"]:
        return "multinode.enabled=false"
    if sid == "12" and not cfg["learning_signal"]["enabled"]:
        return "learning_signal.enabled=false"
    if sid == "13" and not cfg["long_training"]["enabled"]:
        return "long_training.enabled=false"
    if sid == "13b" and not cfg["bones"]["enabled"]:
        return "bones.enabled=false"
    return None


# --------------------------------------------------------------------------
# Command builders (shared by the stages and by the stage-02 compose check)
# --------------------------------------------------------------------------


def placement_override(gpus: int) -> list[str]:
    return [] if gpus <= 1 else [f"cluster.component_placement.supermimic=0-{gpus - 1}"]


def data_override(ctx: Context, config: str) -> list[str]:
    path = ctx.bones if config.startswith("supermimic_bones") else ctx.lafan1
    return [f"env.motion_path={path}"] if path else []


def train_overrides(ctx: Context, sid: str) -> list[dict]:
    """Every training command a stage will run: label, config, overrides, GPUs."""
    cfg = ctx.cfg
    clips = f"env.num_clips={cfg['smoke']['num_clips']}"
    mg = cfg["multi_gpu"]
    common2 = [
        *mg["overrides"],
        f"algorithm.total_updates={mg['total_updates']}",
        "algorithm.seed=0",
        "algorithm.param_check_interval=1",
        "runner.log_per_rank_metrics=true",
    ]
    half = max(1, int(mg["total_updates"]) // 2)
    paper = int(cfg["paper_scale"]["train_updates"])
    ranks_checked = [
        "algorithm.param_check_interval=1",
        "runner.log_per_rank_metrics=true",
    ]
    items: dict[str, list[dict]] = {
        "04": [
            {
                "label": "smoke",
                "config": "supermimic_smoke",
                "gpus": 1,
                "overrides": [
                    clips,
                    "runner.save_interval=3",
                    "runner.val_check_interval=3",
                    "runner.logger.logger_backends=[tensorboard]",
                ],
            }
        ],
        "05": [
            {
                "label": "kill",
                "config": "supermimic_smoke",
                "gpus": 1,
                "overrides": [
                    clips,
                    "runner.save_interval=1",
                    "algorithm.total_updates=1000",
                ],
            },
            # The resume target (last saved step + 2) is set when stage 05 runs.
            {
                "label": "resume",
                "config": "supermimic_smoke",
                "gpus": 1,
                "overrides": [
                    clips,
                    "runner.save_interval=1",
                    "algorithm.total_updates=4",
                    "runner.resume_dir=auto",
                ],
            },
        ],
        "06": [
            {
                "label": "only_eval",
                "config": "supermimic_smoke",
                "gpus": 1,
                "overrides": [clips, "runner.only_eval=true"],
            },
            {
                "label": "velocity_train",
                "config": "supermimic_velocity",
                "gpus": 1,
                "overrides": [clips, *cfg["smoke"]["velocity_overrides"]],
            },
        ],
        "08": [
            {
                "label": "w1",
                "config": "supermimic_lafan1",
                "gpus": 1,
                "overrides": [*common2, f"runner.save_interval={mg['total_updates']}"],
            },
            {
                "label": "w2",
                "config": "supermimic_lafan1",
                "gpus": 2,
                "overrides": [*common2, f"runner.save_interval={mg['total_updates']}"],
            },
            {
                "label": "w2_velocity",
                "config": "supermimic_velocity",
                "gpus": 2,
                "overrides": [*common2, f"runner.save_interval={mg['total_updates']}"],
            },
        ],
        "09": [
            {
                "label": "kill",
                "config": "supermimic_lafan1",
                "gpus": 2,
                "overrides": [
                    *[
                        o
                        for o in common2
                        if not o.startswith("algorithm.total_updates")
                    ],
                    "algorithm.total_updates=100000",
                    f"runner.save_interval={half}",
                ],
            },
            # The resume target (last saved step + total_updates // 2) is set when stage 09 runs.
            {
                "label": "resume",
                "config": "supermimic_lafan1",
                "gpus": 2,
                "overrides": [
                    *common2,
                    f"runner.save_interval={half}",
                    "runner.resume_dir=auto",
                ],
            },
            {
                "label": "w2_to_w1",
                "config": "supermimic_lafan1",
                "gpus": 1,
                "overrides": [
                    *mg["overrides"],
                    f"algorithm.total_updates={int(mg['total_updates']) + 10}",
                    "algorithm.seed=0",
                    "runner.save_interval=-1",
                ],
            },
            {
                "label": "rank_fault",
                "config": "supermimic_lafan1",
                "gpus": 2,
                "overrides": [
                    *mg["overrides"],
                    "algorithm.total_updates=100000",
                    "algorithm.seed=0",
                ],
            },
        ],
        "10": [
            {
                "label": "paper_train",
                "config": "supermimic_lafan1",
                "gpus": 1,
                "overrides": [
                    f"algorithm.total_updates={cfg['paper_scale']['train_updates']}",
                    f"runner.save_interval={cfg['paper_scale']['train_updates']}",
                ],
            },
        ],
        # Without BONES data the LAFAN1 run uses the scaling of supermimic_bones_seed_2gpu.yaml.
        "10b": [
            (
                {
                    "label": "paper_2gpu",
                    "config": "supermimic_bones_seed_2gpu",
                    "gpus": 2,
                    "overrides": [
                        f"algorithm.total_updates={paper}",
                        f"runner.save_interval={paper}",
                        *ranks_checked,
                    ],
                }
                if cfg["bones"]["enabled"] and ctx.bones
                else {
                    "label": "paper_2gpu",
                    "config": "supermimic_lafan1",
                    "gpus": 2,
                    "overrides": [
                        "env.num_envs=8192",
                        "algorithm.wm_window_batch=4096",
                        "algorithm.imagine_batch=8192",
                        "algorithm.lr=2.8284271e-4",
                        f"algorithm.total_updates={paper}",
                        f"runner.save_interval={paper}",
                        *ranks_checked,
                    ],
                }
            )
        ],
        "11": [
            {
                "label": "multinode",
                "config": "supermimic_lafan1",
                "gpus": int(cfg["multinode"]["gpus_per_node"] or 1)
                * max(int(cfg["multinode"]["nodes"] or 1), 1),
                "overrides": [
                    *cfg["multinode"]["overrides"],
                    f"algorithm.total_updates={cfg['multinode']['total_updates']}",
                    "algorithm.seed=0",
                    *ranks_checked,
                    f"runner.save_interval={cfg['multinode']['total_updates']}",
                ],
            },
        ],
    }
    ls = cfg["learning_signal"]
    items["12"] = [
        {
            "label": f"learning_{name}",
            "config": ls["config"],
            "gpus": int(ls["gpus"]),
            "overrides": [
                f"algorithm.total_updates={ls['updates']}",
                f"runner.val_check_interval={ls['val_interval']}",
                f"runner.save_interval={ls['val_interval']}",
                f"algorithm.seed={ls['seed']}",
                "runner.logger.logger_backends=[tensorboard]",
                *extra,
            ],
        }
        for name, extra in ls["variants"].items()
    ]
    lt = cfg["long_training"]
    long_items = []
    for task in ("motion", "velocity"):
        spec = lt[task]
        if task == "velocity" and not spec.get("enabled", True):
            continue
        overrides = ["runner.logger.logger_backends=[tensorboard]"]
        for key, hydra_key in (
            ("total_updates", "algorithm.total_updates"),
            ("save_interval", "runner.save_interval"),
            ("val_interval", "runner.val_check_interval"),
        ):
            if spec.get(key):
                overrides.append(f"{hydra_key}={spec[key]}")
        long_items.append(
            {
                "label": f"long_{task}",
                "config": spec["config"],
                "gpus": int(spec["gpus"] or 1),
                "overrides": overrides,
            }
        )
    items["13"] = long_items
    items["13b"] = [
        {
            "label": "bones",
            "config": "supermimic_bones_seed",
            "gpus": int(cfg["bones"]["gpus"]),
            "overrides": [
                f"algorithm.total_updates={cfg['bones']['train_updates']}",
                f"runner.save_interval={cfg['bones']['train_updates']}",
            ],
        }
    ]
    return items.get(sid, [])


EVAL_FOR = {
    "supermimic_smoke": "supermimic_lafan1_eval",
    "supermimic_lafan1": "supermimic_lafan1_eval",
    "supermimic_lafan1_anchor": "supermimic_lafan1_anchor_eval",
    "supermimic_lafan1_goal730": "supermimic_lafan1_goal730_eval",
    "supermimic_lafan1_obs99": "supermimic_lafan1_obs99_eval",
    "supermimic_velocity": "supermimic_velocity_eval",
    "supermimic_bones_seed": "supermimic_bones_seed_eval",
}


def merge_overrides(overrides: list[str]) -> list[str]:
    """Keep one Hydra override per key, the last one given."""
    kept, seen = [], set()
    for item in reversed(overrides):
        key = item.split("=", 1)[0].lstrip("+~")
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept[::-1]


def training_command(ctx: Context, item: dict, log_dir: Path) -> list[str]:
    """``run_supermimic.sh`` with the data path, placement and log dir set."""
    return [
        "bash",
        "examples/embodiment/run_supermimic.sh",
        item["config"],
        *merge_overrides(
            [
                *data_override(ctx, item["config"]),
                *placement_override(item["gpus"]),
                *item["overrides"],
                f"runner.logger.log_path={log_dir}",
            ]
        ),
    ]


def eval_command(
    ctx: Context, eval_config: str, checkpoint: Path, out_dir: Path, extra: list[str]
) -> list[str]:
    """What ``evaluations/run_eval.sh`` runs for SuperMimic, minus its repo-local log dir."""
    data = ctx.bones if "bones" in eval_config else ctx.lafan1
    return [
        str(ctx.venv_python),
        "evaluations/eval_supermimic.py",
        "--config-path",
        f"{ctx.repo / 'evaluations' / 'supermimic'}/",
        "--config-name",
        eval_config,
        f"runner.logger.log_path={out_dir / 'log'}",
        f"eval.checkpoint={checkpoint}",
        f"eval.output_dir={out_dir}",
        *([f"env.motion_path={data}"] if data else []),
        *extra,
    ]


def checkpoint_steps(log_dir: Path) -> tuple[list[int], list[str]]:
    """Complete global steps (with meta) and leftover ``.tmp`` dirs under a run."""
    steps, partial = [], []
    for path in log_dir.glob("*/global_step_*"):
        name = path.name
        if name.endswith(".tmp"):
            partial.append(str(path))
        elif (path / "checkpoint_meta.json").is_file():
            try:
                steps.append(int(name.split("_")[-1]))
            except ValueError:
                continue
    return sorted(steps), partial


def latest_checkpoint(log_dir: Path) -> Path | None:
    steps, _ = checkpoint_steps(log_dir)
    if not steps:
        return None
    return next(log_dir.glob(f"*/global_step_{steps[-1]}"))


def read_strict_json(path: Path) -> Any:
    def reject(constant: str) -> NoReturn:
        raise ValueError(f"{path} contains non-standard JSON constant {constant}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def csv_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


# --------------------------------------------------------------------------
# Probe scripts, run with the new venv's Python
# --------------------------------------------------------------------------

REQUIRED_REPO_FILES = [
    "requirements/install.sh",
    "requirements/embodied/envs/supermimic.txt",
    "pyproject.toml",
    "examples/embodiment/run_supermimic.sh",
    "examples/embodiment/slurm_supermimic.sh",
    "examples/embodiment/train_supermimic.py",
    "examples/embodiment/config/supermimic_lafan1.yaml",
    "examples/embodiment/config/supermimic_smoke.yaml",
    "examples/embodiment/config/supermimic_velocity.yaml",
    "evaluations/eval_supermimic.py",
    "evaluations/supermimic/supermimic_lafan1_eval.yaml",
    "toolkits/supermimic/profile_sim.py",
    "rlinf/workers/supermimic/trainer.py",
    "rlinf/workers/supermimic/checkpoint.py",
    "rlinf/workers/supermimic/validate.py",
]

KEY_DISTRIBUTIONS = [
    "torch",
    "warp-lang",
    "mujoco",
    "mujoco-warp",
    "mjlab",
    "hydra-core",
    "omegaconf",
    "ray",
    "numpy",
    "pytest",
]

KEY_MODULES = [
    "torch",
    "warp",
    "mujoco",
    "mujoco_warp",
    "mjlab",
    "hydra",
    "omegaconf",
    "ray",
    "rlinf",
    "rlinf.config",
    "rlinf.algorithms.supermimic",
    "rlinf.envs.sim.g1_mimic.motion",
    "rlinf.envs.sim.g1_mimic.env",
    "rlinf.workers.supermimic.checkpoint",
    "rlinf.workers.supermimic.distributed",
    "rlinf.workers.supermimic.trainer",
    "rlinf.workers.supermimic.worker",
    "rlinf.workers.supermimic.evaluator",
    "rlinf.workers.supermimic.validate",
    "rlinf.runners.supermimic_runner",
    "toolkits.supermimic.profile_sim",
]

PROBE_ENV = r"""
import importlib, json, site, sys, sysconfig
from importlib import metadata
dists, modules = sys.argv[1].split(","), sys.argv[2].split(",")
out = {
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "python": sys.version.split()[0],
    "purelib": sysconfig.get_paths()["purelib"],
    "user_site_enabled": bool(site.ENABLE_USER_SITE),
    "sys_path": [p for p in sys.path if p],
    "versions": {},
    "imports": {},
}
for name in dists:
    try:
        out["versions"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        out["versions"][name] = None
for name in modules:
    try:
        module = importlib.import_module(name)
        out["imports"][name] = {"ok": True, "file": getattr(module, "__file__", None)}
    except BaseException as error:
        out["imports"][name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
try:
    import torch
    out["torch_cuda_build"] = torch.version.cuda
except BaseException as error:
    out["torch_cuda_build"] = f"unavailable: {error}"
print("PROBE_JSON " + json.dumps(out))
"""

PROBE_GPU = r"""
import json
out = {}
import torch
out["torch"] = torch.__version__
out["torch_cuda_build"] = torch.version.cuda
out["cuda_available"] = torch.cuda.is_available()
out["device_count"] = torch.cuda.device_count()
if out["cuda_available"]:
    out["devices"] = [torch.cuda.get_device_name(i) for i in range(out["device_count"])]
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    out["torch_kernel_ok"] = float((x * 2).sum().item()) == float(1023 * 1024)
import warp as wp
wp.init()
out["warp"] = wp.config.version
out["warp_cuda_devices"] = int(wp.get_cuda_device_count())
import mujoco, mujoco_warp
out["mujoco"] = mujoco.__version__
out["mujoco_warp_file"] = mujoco_warp.__file__
print("PROBE_JSON " + json.dumps(out))
"""

PROBE_DATA = r"""
import json, sys
from rlinf.envs.sim.g1_mimic.motion import load_motion_library
lib = load_motion_library(sys.argv[1], None, "cpu")
names = list(getattr(lib, "clip_names", []) or [])
out = {"num_clips": int(lib.num_clips), "first_clips": names[:3]}
print("PROBE_JSON " + json.dumps(out))
"""

PROBE_COMPOSE = r"""
import json, sys
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rlinf.workers.supermimic.validate import validate_supermimic_cfg, validate_supermimic_eval_cfg
results = []
for item in json.load(open(sys.argv[1], encoding="utf-8")):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=item["config_dir"], version_base="1.1"):
            cfg = compose(config_name=item["config"], overrides=item["overrides"])
        cfg = validate_supermimic_cfg(cfg, world_size=item["world"])
        if item["kind"] == "eval":
            validate_supermimic_eval_cfg(cfg.eval)
        results.append({"id": item["id"], "ok": True})
    except BaseException as error:
        results.append({"id": item["id"], "ok": False, "error": f"{type(error).__name__}: {error}"})
print("PROBE_JSON " + json.dumps(results))
"""

PROBE_CKPT = r"""
import json, sys
from rlinf.workers.supermimic.checkpoint import verify_checkpoint
meta = verify_checkpoint(sys.argv[1])
print("PROBE_JSON " + json.dumps({"world_size": meta.get("world_size"), "files": sorted(meta.get("files", {})), "global_update": meta.get("global_update")}))
"""


def probe_json(result: CommandResult) -> Any:
    for line in reversed(result.output.splitlines()):
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON ") :])
    return None


def fail_from(result: CommandResult, message: str, **results) -> StageResult:
    """A FAIL carrying the exit code and the likely causes found in the output."""
    note = "超时" if result.timed_out else f"退出码 {result.code}"
    return StageResult(
        FAIL,
        f"{message}（{note}）",
        exit_code=result.code,
        results=results,
        diagnosis=diagnose("\n".join(result.output.splitlines()[-400:])),
    )


# --------------------------------------------------------------------------
# Git (read only)
# --------------------------------------------------------------------------


def git_read(repo: Path, *args: str) -> str | None:
    """Run a read-only git command; ``safe.directory`` is passed on the command line only."""
    git = shutil.which("git")
    if not git:
        return None
    try:
        out = subprocess.run(
            [git, "-c", f"safe.directory={repo}", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def git_snapshot(repo: Path) -> dict:
    status = git_read(repo, "status", "--porcelain", "--untracked-files=all")
    ignored = git_read(
        repo, "status", "--porcelain", "--ignored", "--untracked-files=all"
    )
    return {
        "commit": (git_read(repo, "rev-parse", "HEAD") or "").strip() or None,
        "branch": (git_read(repo, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
        or None,
        "describe": (git_read(repo, "log", "-1", "--format=%h %s") or "").strip()
        or None,
        "status": status,
        "dirty_files": None
        if status is None
        else len([s for s in status.splitlines() if s.strip()]),
        "ignored": None
        if ignored is None
        else sorted(s[3:] for s in ignored.splitlines() if s.startswith("!! ")),
    }


# --------------------------------------------------------------------------
# Stage 00: preflight
# --------------------------------------------------------------------------


def free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return round(shutil.disk_usage(probe).free / 1024**3, 1)


def reachable(host: str, port: int = 443, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def stage_preflight(ctx: Context) -> StageResult:
    cfg, results, blocked, failed = ctx.cfg, {}, [], []
    problems = check_config_types(cfg)
    if problems:
        return StageResult(
            FAIL,
            "配置值不合法：" + "；".join(problems),
            results={"config_problems": problems},
        )
    missing = [p for p in REQUIRED_REPO_FILES if not (ctx.repo / p).is_file()]
    if missing:
        return StageResult(
            FAIL,
            f"{ctx.repo} 不是包含 SuperMimic 的 RLinf 仓库，缺少：{', '.join(missing)}",
        )
    git = ctx.state.manifest.get("git", {})
    results["git_commit"] = git.get("commit")
    results["git_dirty_files"] = git.get("dirty_files")
    if git.get("commit") is None:
        results["git_warning"] = "无法读取 Git 版本（不是 Git 仓库或 git 不可用）"
    if ctx.lafan1 is None:
        blocked.append(
            "data.lafan1 未设置：请填写 LAFAN1 G1 片段目录（prepare_lafan1.py 的输出）"
        )
    elif not ctx.lafan1.is_dir():
        blocked.append(f"data.lafan1={ctx.lafan1} 不存在或不是目录")
    else:
        clips = len(list(ctx.lafan1.glob("clip_*.pt")))
        results["lafan1_clip_files"] = clips
        results["lafan1_manifest"] = (ctx.lafan1 / "manifest.json").is_file()
        if clips == 0:
            blocked.append(f"data.lafan1={ctx.lafan1} 下没有 clip_*.pt")
    if cfg["bones"]["enabled"]:
        if ctx.bones is None or not ctx.bones.is_dir():
            results["bones_warning"] = (
                f"bones.enabled=true 但 data.bones_seed={ctx.bones} 不可用，13b 将 BLOCKED"
            )
    for name, path in (
        ("lafan1", ctx.lafan1),
        ("bones_seed", ctx.bones),
        ("venv", ctx.venv),
    ):
        if path is not None and is_inside(path, ctx.repo):
            failed.append(f"{name} 路径 {path} 位于 RLinf 仓库内部；请放到仓库外")
    for name, path in (
        ("repo", ctx.repo),
        ("lafan1", ctx.lafan1),
        ("bones_seed", ctx.bones),
        ("venv", ctx.venv),
        ("results", ctx.run_dir),
    ):
        if path is not None and re.search(r"[\s'\"=,\[\]{}$`\\]", str(path)):
            failed.append(
                f"{name} 路径 {path} 含空白或 Hydra/Shell 特殊字符，请换一个路径"
            )
    if cfg["install"]["mode"] == "fresh":
        if ctx.venv.exists():
            blocked.append(
                f"install.mode=fresh 要求全新虚拟环境，但 {ctx.venv} 已存在；"
                "脚本不会删除它。请自行处理，或把 paths.venv 改成新路径（可用 {run_id} 占位）"
            )
    elif not ctx.venv_python.exists():
        blocked.append(f"install.mode=reuse 但 {ctx.venv_python} 不存在")
    need = float(cfg["install"]["min_free_gb"])
    for label, path in (
        ("venv", ctx.venv),
        ("results", ctx.run_dir),
        ("cache", ctx.cache_dir()),
    ):
        gb = free_gb(path)
        results[f"free_gb_{label}"] = gb
        if cfg["install"]["mode"] == "fresh" and label != "results" and gb < need:
            blocked.append(
                f"{label} 所在磁盘剩余 {gb} GB，少于 install.min_free_gb={need}"
            )
    tmp_base = Path(ctx.ray_tmpdir()).parent
    results["free_gb_tmp"] = free_gb(tmp_base)
    results["free_gb_home"] = free_gb(Path.home())
    small = [
        f"{where}（{p}）剩余 {results[key]} GB"
        for where, p, key in (
            ("Ray 与测试临时目录", tmp_base, "free_gb_tmp"),
            ("家目录缓存", Path.home(), "free_gb_home"),
        )
        if results[key] < 5
    ]
    if small:
        results["disk_warning"] = (
            "；".join(small)
            + "。建议运行前 export TMPDIR=<大盘上的短路径> XDG_CACHE_HOME=<大盘目录>"
        )
    if cfg["install"]["mode"] == "fresh" and not cfg["install"]["use_mirror"]:
        net = {
            h: reachable(h) for h in ("pypi.org", "github.com", "download.pytorch.org")
        }
        results["network"] = net
        if not all(net.values()):
            results["network_warning"] = (
                "部分安装源不可达；如在国内网络请设 install.use_mirror=true"
            )
    gpus = detect_gpus(os.environ)
    slurm = slurm_info(os.environ)
    results["slurm"] = {
        k: v
        for k, v in slurm.items()
        if k in {"in_allocation", "job_id", "allocation_nodes"}
    }
    results["slurm"]["sbatch_found"] = bool(slurm["sbatch"])
    plan = {}
    for sid in ctx.state.manifest["selected"]:
        placement, reason = decide_placement(ctx, sid, gpus, slurm)
        plan[sid] = f"{placement}: {reason}"
    results["placement_plan"] = plan
    ctx.state.manifest["placement_plan"] = plan
    core = [
        sid
        for sid in ("03", "04", "05", "06")
        if sid in plan and plan[sid].startswith("blocked")
    ]
    if core:
        blocked.append(
            "核心 GPU 阶段无法调度："
            + "；".join(f"{sid} {plan[sid][9:]}" for sid in core)
        )
    if slurm["in_allocation"] is False and slurm["sbatch"] and gpus:
        results["login_node_note"] = (
            "当前节点可见 GPU 但不在 Slurm 作业内：GPU 阶段仍会提交作业，不在登录节点训练"
        )
    if failed:
        return StageResult(FAIL, "；".join(failed), results=results)
    if blocked:
        return StageResult(BLOCKED, "；".join(blocked), results=results)
    return StageResult(PASS, "配置、仓库、数据、磁盘与调度检查通过", results=results)


# --------------------------------------------------------------------------
# Stage 01: clean install
# --------------------------------------------------------------------------


def ensure_uv(ctx: Context, sid: str) -> tuple[str | None, str]:
    """``uv`` for install.sh, kept in a tools venv outside the target venv."""
    link = ctx.tools_bin() / "uv"
    if link.exists():
        return str(link), "复用工具目录中的 uv"
    if not ctx.cfg["install"]["bootstrap_uv"]:
        found = shutil.which("uv", path=ctx.clean_env()["PATH"])
        if found:
            return found, "使用 PATH 上的 uv（install.bootstrap_uv=false）"
        return None, "找不到 uv，且 install.bootstrap_uv=false"
    real = ctx.tools / "bin" / "uv"
    with FileLock(ctx.workspace / ".supermimic_acceptance_tools.lock"):
        if not real.exists():
            result = ctx.run(
                sid,
                [sys.executable, "-m", "venv", str(ctx.tools)],
                timeout_key="install",
                env=ctx.clean_env(),
                cwd=ctx.workspace,
            )
            if result.code != 0:
                return (
                    None,
                    f"无法用 {sys.executable} 创建工具 venv（缺少 venv/ensurepip？）；可设 install.bootstrap_uv=false 并自行把 uv 放入 PATH",
                )
            pip = [
                str(ctx.tools / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "uv",
            ]
            if ctx.cfg["install"]["pip_index_url"]:
                pip += ["--index-url", str(ctx.cfg["install"]["pip_index_url"])]
            result = ctx.run(
                sid, pip, timeout_key="install", env=ctx.clean_env(), cwd=ctx.workspace
            )
            if result.code != 0 or not real.exists():
                return None, "向工具 venv 安装 uv 失败（网络或索引问题），见日志"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(real)
    return str(link), "已在工具 venv 中安装 uv（只供 install.sh 使用，不进入目标环境）"


def user_install_config(env: dict) -> list[str]:
    """User- or system-level settings that can change what uv installs."""
    found = [
        k
        for k in (
            "UV_INDEX_URL",
            "UV_DEFAULT_INDEX",
            "UV_EXTRA_INDEX_URL",
            "UV_INDEX",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "UV_CONFIG_FILE",
        )
        if env.get(k)
    ]
    home = Path(env.get("HOME", "~")).expanduser()
    xdg = Path(env.get("XDG_CONFIG_HOME", home / ".config"))
    for path in (xdg / "uv" / "uv.toml", Path("/etc/uv/uv.toml")):
        if path.is_file():
            found.append(str(path))
    return found


class FileLock:
    """Exclusive lock so two runs never execute install.sh in one checkout at once."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "w")
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        self.handle.write(f"{os.getpid()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        import fcntl

        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def install_command(ctx: Context) -> list[str]:
    cmd = [
        "bash",
        "requirements/install.sh",
        "embodied",
        "--env",
        "supermimic",
        "--venv",
        str(ctx.venv),
    ]
    if ctx.cfg["install"]["python"]:
        cmd += ["--python", str(ctx.cfg["install"]["python"])]
    if ctx.cfg["install"]["use_mirror"]:
        cmd.append("--use-mirror")
    if ctx.cfg["install"]["no_root"]:
        cmd.append("--no-root")
    return cmd


def stage_install(ctx: Context) -> StageResult:
    sid = "01"
    if ctx.cfg["install"]["mode"] == "reuse":
        return StageResult(
            NOT_APPLICABLE, f"install.mode=reuse：复用 {ctx.venv}，未验证干净安装"
        )
    uv, how = ensure_uv(ctx, sid)
    if uv is None:
        return StageResult(BLOCKED, how)
    env = ctx.install_env()
    if Path(uv).parent != ctx.tools_bin():
        env["PATH"] = f"{Path(uv).parent}{os.pathsep}{env['PATH']}"
    uv_version = run_command(
        [uv, "--version"],
        cwd=ctx.workspace,
        env=env,
        log=ctx.log(sid),
        timeout_s=60,
        echo=False,
    )
    cmd = install_command(ctx)
    results = {
        "venv": str(ctx.venv),
        "install_command": shlex.join(cmd),
        "uv": uv,
        "uv_source": how,
        "uv_version": uv_version.output.strip().splitlines()[-1]
        if uv_version.code == 0
        else None,
        "cache_mode": ctx.cfg["install"]["cache"],
        "cache_dir": env["UV_CACHE_DIR"],
        "use_mirror": bool(ctx.cfg["install"]["use_mirror"]),
        "extra_env": sorted((ctx.cfg["install"]["env"] or {}).keys()),
        "user_level_install_config": user_install_config(env),
    }
    # install.sh patches pyproject.toml and writes uv.lock in the checkout, so
    # concurrent runs must not install at the same time.
    with FileLock(ctx.workspace / ".supermimic_acceptance_install.lock"):
        if ctx.venv.exists():
            return StageResult(
                BLOCKED, f"{ctx.venv} 已被其他运行创建，不是全新环境", results=results
            )
        result = ctx.run(sid, cmd, timeout_key="install", env=env)
    if result.code != 0 or not ctx.venv_python.exists():
        return fail_from(
            result,
            "requirements/install.sh 失败：按仓库正式流程无法装出 SuperMimic 环境",
            **results,
        )
    venv_env = ctx.venv_env()
    check = ctx.run(
        sid,
        [uv, "pip", "check", "--python", str(ctx.venv_python)],
        timeout_key="config",
        env=venv_env,
    )
    results["pip_check_exit"] = check.code
    results["pip_check"] = check.tail[-20:]
    freeze = ctx.run(
        sid,
        [uv, "pip", "freeze", "--python", str(ctx.venv_python)],
        timeout_key="config",
        env=venv_env,
    )
    if freeze.code == 0:
        atomic_write(
            ctx.run_dir / "env" / "pip_freeze.txt", freeze.output.split("\n# end:")[0]
        )
        results["packages"] = len(
            [
                line
                for line in freeze.output.splitlines()
                if "==" in line or " @ " in line
            ]
        )
    cfg_file = ctx.venv / "pyvenv.cfg"
    if cfg_file.is_file():
        text = cfg_file.read_text(encoding="utf-8", errors="replace")
        atomic_write(ctx.run_dir / "env" / "pyvenv.cfg", text)
        results["system_site_packages"] = "include-system-site-packages = true" in text
    probe = ctx.py(sid, PROBE_ENV, ",".join(KEY_DISTRIBUTIONS), "torch")
    info = probe_json(probe) or {}
    results["versions"] = info.get("versions", {})
    results["python"] = info.get("python")
    results["torch_cuda_build"] = info.get("torch_cuda_build")
    atomic_write(
        ctx.run_dir / "env" / "install_info.json",
        json.dumps(results, indent=2, ensure_ascii=False),
    )
    if check.code != 0:
        return StageResult(
            FAIL,
            "依赖冲突：`uv pip check` 报告不兼容的包，依赖声明需要修复后用全新环境重测",
            exit_code=check.code,
            results=results,
            diagnosis=[line for line in check.tail if line.strip()][-10:],
        )
    if results.get("system_site_packages"):
        return StageResult(
            FAIL, "新 venv 开启了 system site-packages，不是隔离环境", results=results
        )
    missing = [d for d in KEY_DISTRIBUTIONS if not results["versions"].get(d)]
    if missing:
        return StageResult(
            FAIL,
            "正式安装流程没有装上：" + ", ".join(missing) + "（依赖声明不完整）",
            results=results,
        )
    return StageResult(
        PASS,
        f"install.sh 成功；uv pip check 通过；{results.get('packages', '?')} 个包",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Stage 02: imports, isolation, Hydra, data
# --------------------------------------------------------------------------


def isolation_problems(
    info: dict, venv: Path, repo: Path, probe_dir: Path
) -> list[str]:
    """Imports and sys.path entries that come from outside the new venv."""
    problems = []
    venv_r = str(venv.resolve())
    if not str(Path(info["prefix"]).resolve()).startswith(venv_r):
        problems.append(f"sys.prefix={info['prefix']} 不是新 venv")
    if info.get("user_site_enabled"):
        problems.append("用户 site-packages 已启用")
    allowed = [
        venv_r,
        str(repo.resolve()),
        str(Path(info["base_prefix"]).resolve()),
        str(probe_dir.resolve()),
    ]
    for entry in info.get("sys_path", []):
        # Path-hook entries (editable finders) are not directories; the
        # per-module file check below covers what they would load.
        if not entry.startswith("/"):
            continue
        resolved = str(Path(entry).resolve())
        if not any(resolved.startswith(a) for a in allowed):
            problems.append(f"sys.path 含外部目录 {entry}")
    for name, item in info.get("imports", {}).items():
        file = item.get("file")
        if not item.get("ok") or not file:
            continue
        resolved = str(Path(file).resolve())
        if name.split(".")[0] in {"rlinf", "toolkits"}:
            if not resolved.startswith(str(repo.resolve())):
                problems.append(f"{name} 来自 {file}，不是被测仓库")
        elif not resolved.startswith(venv_r):
            problems.append(f"{name} 来自 {file}，不在新 venv 内")
    return problems


def compose_plan(ctx: Context) -> list[dict]:
    """Every training and eval command this run will execute, as compose items."""
    train_dir = str(ctx.repo / "examples" / "embodiment" / "config")
    eval_dir = str(ctx.repo / "evaluations" / "supermimic")
    plan = []
    for sid in ctx.state.manifest["selected"]:
        if ctx.state.stage(sid)["status"] == NOT_APPLICABLE or config_requirements(
            ctx, sid
        ):
            continue
        for item in train_overrides(ctx, sid):
            overrides = [
                *data_override(ctx, item["config"]),
                *placement_override(item["gpus"]),
                *item["overrides"],
                "runner.logger.log_path=/tmp/compose_check",
            ]
            if sid == "11":
                nodes = int(ctx.cfg["multinode"]["nodes"] or 1)
                overrides = [
                    o
                    for o in overrides
                    if not o.startswith(
                        ("cluster.component_placement", "cluster.num_nodes")
                    )
                ]
                overrides += [
                    f"cluster.num_nodes={nodes}",
                    f"cluster.component_placement.supermimic=0-{item['gpus'] - 1}",
                ]
            overrides = merge_overrides(
                [o for o in overrides if o != "runner.resume_dir=auto"]
            )
            plan.append(
                {
                    "id": f"{sid}:{item['label']}",
                    "config_dir": train_dir,
                    "config": item["config"],
                    "overrides": overrides,
                    "world": item["gpus"],
                    "kind": "train",
                }
            )
            eval_config = EVAL_FOR.get(item["config"])
            if eval_config and sid in {"04", "06", "12", "13", "13b"}:
                extra = (
                    velocity_eval_overrides(ctx) if "velocity" in eval_config else []
                )
                data = ctx.bones if "bones" in eval_config else ctx.lafan1
                plan.append(
                    {
                        "id": f"{sid}:{item['label']}:eval",
                        "config_dir": eval_dir,
                        "config": eval_config,
                        "overrides": [
                            *([f"env.motion_path={data}"] if data else []),
                            "eval.checkpoint=/tmp/compose_check",
                            *extra,
                        ],
                        "world": 1,
                        "kind": "eval",
                    }
                )
    return plan


def velocity_eval_overrides(ctx: Context) -> list[str]:
    if ctx.state.manifest["mode"] == "smoke":
        return ["eval.velocity=smoke"]
    return ["eval.velocity=g2", f"eval.g2_trials={ctx.cfg['evaluation']['g2_trials']}"]


def stage_config(ctx: Context) -> StageResult:
    sid = "02"
    env_probe = ctx.py(
        sid, PROBE_ENV, ",".join(KEY_DISTRIBUTIONS), ",".join(KEY_MODULES)
    )
    info = probe_json(env_probe)
    if info is None:
        return fail_from(env_probe, "新环境的 Python 无法运行探针")
    atomic_write(ctx.run_dir / "env" / "python_probe.json", json.dumps(info, indent=2))
    results = {
        "python": info["python"],
        "executable": info["executable"],
        "versions": info["versions"],
    }
    bad = {k: v["error"] for k, v in info["imports"].items() if not v["ok"]}
    if bad:
        return StageResult(
            FAIL,
            "关键模块导入失败（依赖声明不完整或版本不兼容）："
            + "；".join(f"{k}: {v}" for k, v in bad.items()),
            results={**results, "import_errors": bad},
            diagnosis=diagnose("\n".join(bad.values())),
        )
    problems = isolation_problems(info, ctx.venv, ctx.repo, ctx.run_dir / "scripts")
    if problems:
        return StageResult(
            FAIL,
            "环境未隔离：" + "；".join(problems[:8]),
            results={**results, "isolation": problems},
        )
    results["isolation"] = "全部关键模块来自新 venv 或被测仓库；用户 site 关闭"
    data = ctx.py(sid, PROBE_DATA, str(ctx.lafan1))
    lib = probe_json(data)
    if lib is None:
        return fail_from(data, f"LAFAN1 片段无法加载（{ctx.lafan1}）", **results)
    results["lafan1_clips"] = lib["num_clips"]
    ctx.state.manifest["lafan1_clips"] = lib["num_clips"]
    if lib["num_clips"] < int(ctx.cfg["smoke"]["num_clips"]):
        return StageResult(
            BLOCKED,
            f"LAFAN1 只有 {lib['num_clips']} 条，smoke.num_clips 需要 {ctx.cfg['smoke']['num_clips']}",
            results=results,
        )
    if (
        ctx.cfg["bones"]["enabled"]
        and "13b" in ctx.state.manifest["selected"]
        and ctx.bones
        and ctx.bones.is_dir()
    ):
        bones = ctx.py(sid, PROBE_DATA, str(ctx.bones))
        b = probe_json(bones)
        results["bones_clips"] = b["num_clips"] if b else f"加载失败：{bones.tail[-1:]}"
    plan = compose_plan(ctx)
    plan_path = ctx.run_dir / "scripts" / "compose_plan.json"
    atomic_write(plan_path, json.dumps(plan, indent=2))
    composed = ctx.py(sid, PROBE_COMPOSE, str(plan_path))
    checked = probe_json(composed)
    if checked is None:
        return fail_from(composed, "Hydra 组合检查脚本失败", **results)
    errors = [f"{c['id']}: {c['error']}" for c in checked if not c["ok"]]
    results["hydra_commands_checked"] = len(checked)
    if errors:
        return StageResult(
            FAIL,
            "计划执行的命令有 Hydra 组合或校验错误：" + "；".join(errors[:5]),
            results={**results, "hydra_errors": errors},
        )
    return StageResult(
        PASS,
        f"{len(KEY_MODULES)} 个模块导入、隔离、数据（{lib['num_clips']} 条）与 {len(checked)} 条命令组合检查通过",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Stage 03: unit tests
# --------------------------------------------------------------------------

_SKIP_LINE = re.compile(r"^SKIPPED \[(\d+)\] ([^:]+):\d+: (.*)$")
_SUMMARY = re.compile(
    r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed|deselected)"
)
FORBIDDEN_SKIP = re.compile(
    r"mjlab|CUDA|Warp|Gloo|could not import|No module|needs bash", re.I
)


def stage_unit_tests(ctx: Context) -> StageResult:
    sid = "03"
    gpu = ctx.py(sid, PROBE_GPU)
    info = probe_json(gpu)
    if info is None:
        return fail_from(gpu, "GPU 探针失败：torch/Warp/MuJoCo 在 GPU 节点上无法初始化")
    atomic_write(
        ctx.run_dir / "metrics" / "03_gpu_probe.json", json.dumps(info, indent=2)
    )
    results = {
        "gpu_probe": {
            k: info.get(k)
            for k in (
                "torch",
                "torch_cuda_build",
                "device_count",
                "devices",
                "warp",
                "warp_cuda_devices",
                "mujoco",
            )
        }
    }
    if not info.get("cuda_available") or not info.get("torch_kernel_ok"):
        return StageResult(
            FAIL,
            "nvidia-smi 可见 GPU，但新环境的 torch 无法使用 CUDA（驱动与 CUDA 构建不匹配？）",
            results=results,
        )
    if not info.get("warp_cuda_devices"):
        return StageResult(FAIL, "Warp 看不到 CUDA 设备", results=results)
    files = sorted((ctx.repo / "tests" / "unit_tests").glob("test_supermimic*.py"))
    files.append(ctx.repo / "tests" / "unit_tests" / "test_cluster_config.py")
    env = ctx.venv_env()
    env["PYTHONPATH"] = f"{ctx.repo}{os.pathsep}{ctx.repo / 'tests' / 'unit_tests'}"
    per_file, failures, bad_skips = {}, [], []
    for path in files:
        rel = str(path.relative_to(ctx.repo))
        result = ctx.run(
            sid,
            [
                str(ctx.venv_python),
                "-m",
                "pytest",
                rel,
                "-rs",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            timeout_key="unit_test_file",
            env=env,
        )
        cleanup_run_processes(ctx.run_id)
        counts = {}
        for line in reversed(result.output.splitlines()):
            if re.search(r" in [\d.]+s", line) and _SUMMARY.search(line):
                counts = {word: int(number) for number, word in _SUMMARY.findall(line)}
                break
        skips = []
        for line in result.output.splitlines():
            match = _SKIP_LINE.match(line.strip())
            if match:
                skips.append(f"{match.group(1)}× {match.group(3)}")
                if FORBIDDEN_SKIP.search(match.group(3)):
                    bad_skips.append(f"{rel}: {match.group(3)}")
        per_file[rel] = {
            "exit": result.code,
            "seconds": round(result.seconds, 1),
            **counts,
            "skips": skips,
        }
        if result.code != 0 or result.timed_out:
            failures.append(
                f"{rel}（{'超时' if result.timed_out else f'退出码 {result.code}'}）"
            )
    atomic_write(
        ctx.run_dir / "metrics" / "03_unit_tests.json",
        json.dumps(per_file, indent=2, ensure_ascii=False),
    )
    results["files"] = len(files)
    results["passed"] = sum(v.get("passed", 0) for v in per_file.values())
    results["failed"] = sum(
        v.get("failed", 0) + v.get("error", 0) + v.get("errors", 0)
        for v in per_file.values()
    )
    results["skipped"] = sum(v.get("skipped", 0) for v in per_file.values())
    results["allowed_skips"] = sorted(
        {
            s
            for v in per_file.values()
            for s in v["skips"]
            if not FORBIDDEN_SKIP.search(s)
        }
    )
    if failures:
        return StageResult(
            FAIL, "单元测试失败：" + "；".join(failures), results=results, diagnosis=[]
        )
    if bad_skips:
        return StageResult(
            FAIL,
            "GPU 节点上有测试因依赖或 CUDA 不可用被跳过：" + "；".join(bad_skips[:6]),
            results=results,
        )
    return StageResult(
        PASS,
        f"{len(files)} 个文件：{results['passed']} passed，{results['skipped']} skipped（均为卡数等允许原因）",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------

COUNTERS = (
    "sim_vector_steps",
    "env_transitions",
    "wm_optimizer_steps",
    "actor_optimizer_steps",
    "critic_optimizer_steps",
)


def train(
    ctx: Context,
    sid: str,
    item: dict,
    *,
    timeout_key: str,
    extra: list[str] | None = None,
    log_dir: Path | None = None,
    watch: Callable[[str], bool] | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Run one training command; returns exit info, parsed metrics and checkpoint state."""
    log_dir = log_dir or (ctx.run_dir / "checkpoints" / f"{sid}_{item['label']}")
    cmd = training_command(
        ctx, {**item, "overrides": [*item["overrides"], *(extra or [])]}, log_dir
    )
    env = ctx.venv_env(item["gpus"])
    result = ctx.run(
        sid, cmd, timeout_key=timeout_key, env=env, watch=watch, timeout_s=timeout_s
    )
    orphans = marker_pids(ctx.run_id)
    cleanup_run_processes(ctx.run_id)
    parsed = parse_training_log(result.output)
    metrics_file = ctx.run_dir / "metrics" / f"{sid}_{item['label']}.jsonl"
    atomic_write(
        metrics_file,
        "".join(
            json.dumps({"kind": k, **row}, default=str) + "\n"
            for k in ("metrics", "evals")
            for row in parsed[k]
        ),
    )
    steps, partial = checkpoint_steps(log_dir)
    return {
        "result": result,
        "metrics": parsed["metrics"],
        "evals": parsed["evals"],
        "log_dir": log_dir,
        "steps": steps,
        "partial": partial,
        "orphans": orphans,
    }


def last_value(rows: list[dict], key: str) -> Any:
    for row in reversed(rows):
        if key in row:
            return row[key]
    return None


def numeric_values(rows: list[dict]) -> list:
    return [
        v
        for row in rows
        for v in row.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def check_training(run: dict, *, expect_updates: int | None = None) -> list[str]:
    """Problems with one finished training run (empty when it is healthy)."""
    problems = []
    rows = run["metrics"]
    if not rows:
        problems.append("日志中没有 `supermimic metrics` 行")
        return problems
    if not all_finite(numeric_values(rows)) or not all_finite(
        numeric_values(run["evals"])
    ):
        problems.append("指标中出现 NaN/Inf")
    for key in COUNTERS:
        series = [row[key] for row in rows if key in row]
        if not series:
            problems.append(f"缺少计数 {key}")
        elif any(b < a for a, b in zip(series, series[1:])):
            problems.append(f"计数 {key} 不单调")
    if (
        expect_updates is not None
        and last_value(rows, "train/global_update") != expect_updates
    ):
        problems.append(
            f"train/global_update 最终为 {last_value(rows, 'train/global_update')}，应为 {expect_updates}"
        )
    return problems


def verify_ckpt(ctx: Context, sid: str, path: Path) -> dict | None:
    return probe_json(ctx.py(sid, PROBE_CKPT, str(path)))


def summarize_run(run: dict) -> dict:
    rows = run["metrics"]
    out = {
        "exit": run["result"].code,
        "seconds": round(run["result"].seconds, 1),
        "logged_updates": len(rows),
    }
    for key in (
        *COUNTERS,
        "train/global_update",
        "train/nonfinite_skips",
        "train/world_size",
        "memory/peak_allocated",
        "time/comm_s",
    ):
        value = last_value(rows, key)
        if value is not None:
            out[key] = value
    if run["steps"]:
        out["checkpoints"] = run["steps"]
    return out


# --------------------------------------------------------------------------
# Stages 04-06: smoke, checkpoint, evaluation
# --------------------------------------------------------------------------


def stage_smoke(ctx: Context) -> StageResult:
    sid = "04"
    item = train_overrides(ctx, sid)[0]
    run = train(ctx, sid, item, timeout_key="smoke")
    results = summarize_run(run)
    if run["result"].code != 0:
        return fail_from(run["result"], "单 GPU Smoke 训练失败", **results)
    expect = 3
    problems = check_training(run, expect_updates=expect)
    ckpt = latest_checkpoint(run["log_dir"])
    if ckpt is None:
        problems.append("训练结束但没有完整检查点")
    else:
        meta = verify_ckpt(ctx, sid, ckpt)
        results["checkpoint"] = str(ckpt)
        if meta is None:
            problems.append(f"检查点 {ckpt} 校验失败")
    tb = list(run["log_dir"].rglob("events.out.tfevents*"))
    results["tensorboard_files"] = len(tb)
    if not tb:
        problems.append("没有 TensorBoard 事件文件")
    if run["orphans"]:
        results["leftover_after_exit"] = len(run["orphans"])
    ctx.state.manifest["smoke_checkpoint_root"] = str(
        run["log_dir"] / "supermimic_smoke"
    )
    if problems:
        return StageResult(
            FAIL,
            "Smoke 训练结果异常：" + "；".join(problems),
            run["result"].code,
            results,
        )
    return StageResult(
        PASS,
        f"真实 MuJoCo Warp 训练 {expect} 次更新，计数单调、指标有限、检查点完整",
        0,
        results,
    )


def stage_checkpoint(ctx: Context) -> StageResult:
    sid = "05"
    kill_item, resume_item = train_overrides(ctx, sid)
    log_dir = ctx.run_dir / "checkpoints" / "05_kill_resume"
    kill_item = {
        **kill_item,
        "overrides": [
            o
            for o in kill_item["overrides"]
            if not o.startswith("algorithm.total_updates")
        ]
        + ["algorithm.total_updates=1000"],
    }

    def two_saved(_output: str) -> bool:
        return len(checkpoint_steps(log_dir)[0]) >= 2

    killed = train(
        ctx, sid, kill_item, timeout_key="checkpoint", log_dir=log_dir, watch=two_saved
    )
    results = {"kill_run": summarize_run(killed)}
    if not killed["result"].killed_by_watch:
        return fail_from(
            killed["result"], "强杀测试没有在保存两个检查点后触发", **results
        )
    steps = killed["steps"]
    last = steps[-1]
    results["killed_after_steps"] = steps
    results["partial_dirs_after_kill"] = len(killed["partial"])
    target = last + 2
    resume_item = {
        **resume_item,
        "overrides": [
            o
            for o in resume_item["overrides"]
            if not o.startswith("algorithm.total_updates")
        ]
        + [f"algorithm.total_updates={target}"],
    }
    resumed = train(ctx, sid, resume_item, timeout_key="checkpoint", log_dir=log_dir)
    results["resume_run"] = summarize_run(resumed)
    if resumed["result"].code != 0:
        return fail_from(resumed["result"], "从强杀后的检查点恢复失败", **results)
    problems = []
    match = re.search(
        r"Resuming SuperMimic .*?global_step_(\d+)", resumed["result"].output
    )
    results["resumed_from"] = int(match.group(1)) if match else None
    if results["resumed_from"] != last:
        problems.append(f"应从 global_step_{last} 恢复，实际 {results['resumed_from']}")
    first = next(
        (
            row.get("train/global_update")
            for row in resumed["metrics"]
            if row.get("train/global_update", 0) > last
        ),
        None,
    )
    if first != last + 1:
        problems.append(f"恢复后第一次更新应为 {last + 1}，实际 {first}")
    problems += check_training(resumed, expect_updates=target)
    final = next(log_dir.glob(f"*/global_step_{target}"), None)
    if final is None or verify_ckpt(ctx, sid, final) is None:
        problems.append(f"恢复后没有完整的 global_step_{target}")
    if problems:
        return StageResult(
            FAIL,
            "检查点恢复异常：" + "；".join(problems),
            resumed["result"].code,
            results,
        )
    return StageResult(
        PASS,
        f"保存 {steps} 后 SIGKILL，恢复自 {last} 并训练到 {target}，检查点校验通过",
        0,
        results,
    )


def run_eval(
    ctx: Context,
    sid: str,
    label: str,
    eval_config: str,
    checkpoint: Path,
    extra: list[str],
) -> tuple[CommandResult, dict, list[str]]:
    out_dir = ctx.run_dir / "evaluations" / f"{sid}_{label}"
    result = ctx.run(
        sid,
        eval_command(ctx, eval_config, checkpoint, out_dir, extra),
        timeout_key="evaluation",
    )
    cleanup_run_processes(ctx.run_id)
    problems, info = [], {"output_dir": str(out_dir)}
    if result.code != 0:
        return result, info, [f"评测命令失败（退出码 {result.code}）"]
    try:
        summary = read_strict_json(out_dir / "summary.json")
    except (OSError, ValueError) as error:
        return result, info, [f"summary.json 不可读：{error}"]
    metrics = summary.get("metrics", {})
    meta = summary.get("meta", {})
    info.update(
        num_clips=meta.get("num_clips"),
        global_step=meta.get("global_step"),
        metrics=dict(list(metrics.items())[:12]),
    )
    if not all_finite([v for v in metrics.values() if isinstance(v, (int, float))]):
        problems.append("评测指标出现非有限值")
    if (out_dir / "per_clip.csv").is_file():
        rows = csv_rows(out_dir / "per_clip.csv")
        info["per_clip_rows"] = rows
        if rows != meta.get("num_clips"):
            problems.append(f"per_clip.csv {rows} 行，片段数 {meta.get('num_clips')}")
    for name in ("velocity_smoke.csv", "velocity_g2.csv"):
        if (out_dir / name).is_file():
            info[name] = csv_rows(out_dir / name)
            if info[name] == 0:
                problems.append(f"{name} 为空")
    return result, info, problems


def stage_evaluation(ctx: Context) -> StageResult:
    sid = "06"
    root = Path(
        ctx.state.manifest.get("smoke_checkpoint_root")
        or ctx.run_dir / "checkpoints" / "04_smoke" / "supermimic_smoke"
    )
    results, problems = {}, []
    first, info1, p1 = run_eval(ctx, sid, "motion", "supermimic_lafan1_eval", root, [])
    results["motion"] = info1
    if first.code != 0:
        return fail_from(first, "动作跟踪独立评测失败", **results)
    problems += p1
    second, info2, p2 = run_eval(
        ctx, sid, "motion_repeat", "supermimic_lafan1_eval", root, []
    )
    problems += p2
    # if second.code == 0 and info1.get("metrics") != info2.get("metrics"):
    #     problems.append("同一检查点、同一种子两次评测结果不同")

    # GPU evaluation is numerically reproducible within tolerance,
    # but bitwise equality is not guaranteed.
    metrics1 = info1.get("metrics") or {}
    metrics2 = info2.get("metrics") or {}

    atol = 5e-5
    rtol = 2e-4

    count_keys = {"eval/num_clips", "eval/steps"}

    repeat_diffs = {}
    repeat_ok = bool(metrics1) and metrics1.keys() == metrics2.keys()

    for key in metrics1.keys() & metrics2.keys():
        a, b = metrics1[key], metrics2[key]

        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            repeat_ok = False
            continue

        if not math.isfinite(a) or not math.isfinite(b):
            repeat_ok = False
            continue

        abs_diff = abs(a - b)
        repeat_diffs[key] = abs_diff

        if key == "eval/num_clips":
            close = a == b
        else:
            close = math.isclose(a, b, rel_tol=rtol, abs_tol=atol)

        if not close:
            repeat_ok = False

    results["motion_repeat_within_tolerance"] = repeat_ok
    results["motion_repeat_abs_diffs"] = repeat_diffs
    results["motion_repeat_atol"] = atol
    results["motion_repeat_rtol"] = rtol

    if second.code == 0 and not repeat_ok:
        problems.append("同一检查点、同一种子两次评测超出数值容差")
    
    results["motion_repeat_identical"] = info1.get("metrics") == info2.get("metrics")
    only_item, vel_item = train_overrides(ctx, sid)
    latest = latest_checkpoint(root.parent)
    only = train(
        ctx,
        sid,
        only_item,
        timeout_key="evaluation",
        extra=[f"runner.resume_dir={latest}"],
    )
    results["only_eval"] = {
        "exit": only["result"].code,
        "eval_lines": len(only["evals"]),
    }
    if only["result"].code != 0:
        return fail_from(
            only["result"], "训练入口 runner.only_eval=true 失败", **results
        )
    if not only["evals"] or not all_finite(numeric_values(only["evals"])):
        problems.append("only_eval 没有输出有限的评测指标")
    vel = train(ctx, sid, vel_item, timeout_key="smoke")
    results["velocity_train"] = summarize_run(vel)
    if vel["result"].code != 0:
        return fail_from(vel["result"], "Velocity Smoke 训练失败", **results)
    problems += [f"velocity: {p}" for p in check_training(vel)]
    vroot = vel["log_dir"] / "supermimic_velocity"
    vres, vinfo, vp = run_eval(
        ctx,
        sid,
        "velocity",
        "supermimic_velocity_eval",
        vroot,
        velocity_eval_overrides(ctx),
    )
    results["velocity_eval"] = vinfo
    if vres.code != 0:
        return fail_from(vres, "Velocity 独立评测失败", **results)
    problems += vp
    if problems:
        return StageResult(FAIL, "评测异常：" + "；".join(problems), 0, results)
    return StageResult(
        PASS,
        f"Motion 评测 {info1.get('per_clip_rows')} 条片段、两次一致；only_eval 与 Velocity 评测通过",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Stages 07 and 10: profiling and paper scale
# --------------------------------------------------------------------------


def profile(
    ctx: Context, sid: str, num_envs: list[int], estimate: list[int], timeout_key: str
) -> tuple[CommandResult, dict | None]:
    out = ctx.run_dir / "metrics" / f"{sid}_profile.json"
    p = ctx.cfg["profiling"]
    cmd = [
        str(ctx.venv_python),
        "toolkits/supermimic/profile_sim.py",
        "--config",
        "supermimic_lafan1",
        "--motion-path",
        str(ctx.lafan1),
        "--num-envs",
        *[str(n) for n in num_envs],
        "--steps",
        str(p["steps"]),
        "--warmup",
        str(p["warmup"]),
        "--update-iters",
        str(p["update_iters"]),
        "--estimate-envs",
        *[str(n) for n in estimate],
        "--timeout",
        str(int(ctx.timeout(timeout_key))),
        "--out",
        str(out),
    ]
    result = ctx.run(sid, cmd, timeout_key=timeout_key)
    cleanup_run_processes(ctx.run_id)
    try:
        return result, read_strict_json(out)
    except (OSError, ValueError):
        return result, None


def profile_rows(report: dict) -> dict:
    keep = (
        "vector_steps_per_s",
        "env_transitions_per_s",
        "sim_share",
        "update_s_steady",
        "torch_peak_allocated",
        "build_s",
    )
    return {
        str(r["num_envs"]): (
            {k: r[k] for k in keep if k in r}
            if "error" not in r
            else {"error": r["error"][:200]}
        )
        for r in report.get("measured", [])
    }


def stage_profiling(ctx: Context) -> StageResult:
    sid = "07"
    sizes = [int(n) for n in ctx.cfg["profiling"]["num_envs"]]
    result, report = profile(
        ctx, sid, sizes, [int(ctx.cfg["paper_scale"]["profile_num_envs"])], "profiling"
    )
    if report is None:
        return fail_from(result, "Profiling 没有产出报告")
    results = {
        "measured": profile_rows(report),
        "estimated(仅外推，非测量)": report.get("estimated", []),
    }
    errors = [k for k, v in results["measured"].items() if "error" in v]
    if errors:
        return StageResult(
            FAIL,
            f"num_envs={','.join(errors)} 的 Profiling 失败（见 metrics/07_profile.json）",
            result.code,
            results,
            diagnose(json.dumps(report)),
        )
    return StageResult(PASS, f"num_envs {sizes} 的吞吐与显存已实测", 0, results)


def stage_paper_scale(ctx: Context) -> StageResult:
    sid = "10"
    gpus = detect_gpus(ctx.venv_env())
    need = float(ctx.cfg["paper_scale"]["min_gpu_memory_gb"])
    results = {"gpu": gpus[0] if gpus else None}
    if not gpus or gpus[0]["memory_gb"] < need:
        have = gpus[0]["memory_gb"] if gpus else 0
        return StageResult(
            BLOCKED,
            f"论文规模需要单卡显存 >= {need} GB（paper_scale.min_gpu_memory_gb），当前 {have} GB",
            results=results,
        )
    n = int(ctx.cfg["paper_scale"]["profile_num_envs"])
    result, report = profile(ctx, sid, [n], [], "paper_scale")
    if report is None:
        return fail_from(result, f"num_envs={n} 的 Profiling 没有产出报告", **results)
    results["profile"] = profile_rows(report)
    if any("error" in v for v in results["profile"].values()):
        return StageResult(
            FAIL,
            f"论文规模 num_envs={n} 无法运行（见 metrics/10_profile.json）",
            result.code,
            results,
            diagnose(json.dumps(report)),
        )
    item = train_overrides(ctx, sid)[0]
    run = train(ctx, sid, item, timeout_key="paper_scale")
    results["train"] = summarize_run(run)
    if run["result"].code != 0:
        return fail_from(
            run["result"], "论文配置 supermimic_lafan1 短训练失败", **results
        )
    problems = check_training(
        run, expect_updates=int(ctx.cfg["paper_scale"]["train_updates"])
    )
    if problems:
        return StageResult(FAIL, "论文规模训练异常：" + "；".join(problems), 0, results)
    return StageResult(
        PASS,
        f"论文规模 {n} 环境可运行，{ctx.cfg['paper_scale']['train_updates']} 次更新完成",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Stages 08, 09, 10b: two GPUs
# --------------------------------------------------------------------------


def check_two_rank(run: dict, world: int) -> list[str]:
    problems = []
    rows = run["metrics"]
    divergence = [
        row["train/param_divergence"] for row in rows if "train/param_divergence" in row
    ]
    if not divergence:
        problems.append("没有记录 train/param_divergence")
    elif any(d != 0.0 for d in divergence):
        problems.append(f"train/param_divergence 不为 0：最大 {max(divergence)}")
    if last_value(rows, "train/world_size") not in (world, float(world)):
        problems.append(
            f"train/world_size={last_value(rows, 'train/world_size')}，应为 {world}"
        )
    out = run["result"].output
    if "NCCL INFO" not in out:
        problems.append("日志中没有 NCCL INFO 行，无法证明使用了 NCCL")
    last = rows[-1] if rows else {}
    for key in COUNTERS:
        r0, r1 = last.get(f"rank_0/{key}"), last.get(f"rank_1/{key}")
        if r0 is not None and r1 is not None and r0 != r1:
            problems.append(f"rank_0/{key}={r0} 与 rank_1/{key}={r1} 不同")
    return problems


def stage_nccl(ctx: Context) -> StageResult:
    sid = "08"
    w1_item, w2_item, vel = train_overrides(ctx, sid)
    updates = int(ctx.cfg["multi_gpu"]["total_updates"])
    w1 = train(ctx, sid, w1_item, timeout_key="two_gpu")
    results = {"w1": summarize_run(w1)}
    if w1["result"].code != 0:
        return fail_from(w1["result"], "单卡对照运行失败", **results)
    w2 = train(ctx, sid, w2_item, timeout_key="two_gpu")
    results["w2"] = summarize_run(w2)
    if w2["result"].code != 0:
        return fail_from(w2["result"], "双 GPU NCCL 训练失败", **results)
    problems = check_training(w1, expect_updates=updates) + check_training(
        w2, expect_updates=updates
    )
    problems += check_two_rank(w2, 2)
    for key in COUNTERS:
        a, b = last_value(w1["metrics"], key), last_value(w2["metrics"], key)
        if a != b:
            problems.append(f"W1 与 W2 的 {key} 不同（{a} vs {b}）")
    ckpt = latest_checkpoint(w2["log_dir"])
    meta = verify_ckpt(ctx, sid, ckpt) if ckpt else None
    results["w2_checkpoint_meta"] = meta
    if (
        not meta
        or meta.get("world_size") != 2
        or not {"rank_0.pt", "rank_1.pt"} <= set(meta.get("files", []))
    ):
        problems.append("双卡检查点缺少 world_size=2 或 rank_0.pt/rank_1.pt")
    nccl_lines = [
        line for line in w2["result"].output.splitlines() if "NCCL INFO" in line
    ]
    atomic_write(
        ctx.run_dir / "metrics" / "08_nccl_info.txt", "\n".join(nccl_lines[:400]) + "\n"
    )
    results["nccl_info_lines"] = len(nccl_lines)
    comm = [row.get("time/comm_s") for row in w2["metrics"] if row.get("time/comm_s")]
    if comm:
        results["time/comm_s_mean"] = sum(comm) / len(comm)
    v2 = train(ctx, sid, vel, timeout_key="two_gpu")
    results["w2_velocity"] = summarize_run(v2)
    if v2["result"].code != 0:
        return fail_from(v2["result"], "双 GPU Velocity 训练失败", **results)
    problems += [
        f"velocity: {p}"
        for p in check_training(v2, expect_updates=updates) + check_two_rank(v2, 2)
    ]
    if problems:
        return StageResult(
            FAIL, "双 GPU 一致性检查失败：" + "；".join(problems[:6]), 0, results
        )
    return StageResult(
        PASS,
        "真实 NCCL：param_divergence 恒为 0，W1/W2 计数逐项相等，双 rank 检查点完整",
        0,
        results,
    )


def _rank_pid(run_id: str, output: str, rank: int) -> int | None:
    """PID of one of our Ray worker processes with ``RANK=<rank>``."""
    wanted = f"RANK={rank}".encode()
    for pid in marker_pids(run_id):
        try:
            with open(f"/proc/{pid}/environ", "rb") as handle:
                env = handle.read().split(b"\0")
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read()
        except OSError:
            continue
        if wanted in env and cmdline.startswith(b"ray::"):
            return pid
    match = re.search(rf"SuperMimicGroup\(rank={rank}\)\s+pid=(\d+)", output)
    return int(match.group(1)) if match else None


def stage_multigpu_fault(ctx: Context) -> StageResult:
    sid = "09"
    kill_item, resume_item, w1_item, fault_item = train_overrides(ctx, sid)
    log_dir = ctx.run_dir / "checkpoints" / "09_kill_resume"
    kill_item = {
        **kill_item,
        "overrides": [
            o
            for o in kill_item["overrides"]
            if not o.startswith("algorithm.total_updates")
        ]
        + ["algorithm.total_updates=100000"],
    }
    killed = train(
        ctx,
        sid,
        kill_item,
        timeout_key="multi_gpu",
        log_dir=log_dir,
        watch=lambda _o: bool(checkpoint_steps(log_dir)[0]),
    )
    results = {"kill_run": summarize_run(killed)}
    if not killed["result"].killed_by_watch:
        return fail_from(
            killed["result"], "双卡强杀测试没有在第一个检查点后触发", **results
        )
    last = killed["steps"][-1]
    half = max(1, int(ctx.cfg["multi_gpu"]["total_updates"]) // 2)
    target = last + half
    resume_item = {
        **resume_item,
        "overrides": [
            o
            for o in resume_item["overrides"]
            if not o.startswith("algorithm.total_updates")
        ]
        + [f"algorithm.total_updates={target}"],
    }
    resumed = train(ctx, sid, resume_item, timeout_key="multi_gpu", log_dir=log_dir)
    results["resume_run"] = summarize_run(resumed)
    if resumed["result"].code != 0:
        return fail_from(resumed["result"], "双卡恢复失败", **results)
    problems = check_training(resumed, expect_updates=target) + check_two_rank(
        resumed, 2
    )
    ckpt = latest_checkpoint(log_dir)
    w1_item = {
        **w1_item,
        "overrides": [
            o
            for o in w1_item["overrides"]
            if not o.startswith("algorithm.total_updates")
        ]
        + [f"algorithm.total_updates={target + 10}"],
    }
    w1 = train(
        ctx, sid, w1_item, timeout_key="multi_gpu", extra=[f"runner.resume_dir={ckpt}"]
    )
    results["w2_to_w1"] = summarize_run(w1)
    if w1["result"].code != 0:
        return fail_from(w1["result"], "双卡检查点换到单卡恢复失败", **results)
    problems += [f"w2→w1: {p}" for p in check_training(w1, expect_updates=target + 10)]
    fault_timeout = float(ctx.cfg["multi_gpu"]["fault_timeout_minutes"]) * 60.0
    killed_at: dict = {}

    def kill_rank_one(output: str) -> bool:
        if killed_at or "supermimic metrics" not in output:
            return False
        pid = _rank_pid(ctx.run_id, output, 1)
        if pid and pid in marker_pids(ctx.run_id):
            os.kill(pid, signal.SIGKILL)
            killed_at.update(pid=pid, t=time.time())
        return False

    fault = train(
        ctx,
        sid,
        fault_item,
        timeout_key="multi_gpu",
        watch=kill_rank_one,
        timeout_s=ctx.timeout("multi_gpu"),
    )
    detect = round(time.time() - killed_at["t"], 1) if killed_at else None
    results["rank_fault"] = {
        "killed_pid": killed_at.get("pid"),
        "exit": fault["result"].code,
        "timed_out": fault["result"].timed_out,
        "seconds_to_exit_after_kill": detect,
        "orphans_after_exit": len(fault["orphans"]),
    }
    if not killed_at:
        problems.append("没能从日志中找到 rank 1 的 PID，未执行单 rank 故障注入")
    elif fault["result"].code == 0 or fault["result"].timed_out:
        problems.append("杀掉 rank 1 后作业没有以非零退出码结束")
    elif detect is not None and detect > fault_timeout:
        problems.append(
            f"故障后 {detect} s 才退出，超过 multi_gpu.fault_timeout_minutes"
        )
    if problems:
        return StageResult(
            FAIL, "多 GPU 恢复/故障检查失败：" + "；".join(problems[:6]), 0, results
        )
    return StageResult(
        PASS,
        f"双卡从 {last} 恢复到 {target}；W2→W1 继续训练；rank 1 被杀后 {detect} s 非零退出",
        0,
        results,
    )


def stage_paper_2gpu(ctx: Context) -> StageResult:
    sid = "10b"
    gpus = detect_gpus(ctx.venv_env())
    need = float(ctx.cfg["paper_scale"]["min_gpu_memory_gb"])
    if len(gpus) < 2 or min(g["memory_gb"] for g in gpus[:2]) < need:
        return StageResult(
            BLOCKED,
            f"论文双卡配方需要 2 张显存 >= {need} GB 的 GPU",
            results={"gpus": gpus},
        )
    item = train_overrides(ctx, sid)[0]
    if item["config"].startswith("supermimic_bones") and not (
        ctx.bones and ctx.bones.is_dir()
    ):
        return StageResult(BLOCKED, "BONES 双卡配方需要 data.bones_seed")
    run = train(ctx, sid, item, timeout_key="paper_scale")
    results = {"config": item["config"], "train": summarize_run(run)}
    if run["result"].code != 0:
        return fail_from(run["result"], "论文双卡配方训练失败", **results)
    problems = check_training(
        run, expect_updates=int(ctx.cfg["paper_scale"]["train_updates"])
    ) + check_two_rank(run, 2)
    if problems:
        return StageResult(
            FAIL, "论文双卡配方异常：" + "；".join(problems[:6]), 0, results
        )
    return StageResult(
        PASS,
        f"{item['config']} 双卡 {ctx.cfg['paper_scale']['train_updates']} 次更新完成",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Slurm
# --------------------------------------------------------------------------


def sbatch_header(
    ctx: Context, key: str, name: str, nodes: int, gpus_per_node: int, output: Path
) -> list[str]:
    s = ctx.cfg["slurm"]
    job = s["jobs"][key]
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={s['partition']}",
        f"#SBATCH --nodes={nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={job['cpus']}",
        f"#SBATCH --mem={job['mem']}",
        f"#SBATCH --time={job['time']}",
        f"#SBATCH --output={output}",
    ]
    if gpus_per_node:
        if s["gpu_request"] == "gres":
            lines.append(f"#SBATCH --gres=gpu:{gpus_per_node}")
        else:
            lines.append(f"#SBATCH --gpus-per-node={gpus_per_node}")
    if s["account"]:
        lines.append(f"#SBATCH --account={s['account']}")
    if s["qos"]:
        lines.append(f"#SBATCH --qos={s['qos']}")
    lines += [f"#SBATCH {arg}" for arg in s["extra_sbatch_args"] or []]
    return lines


def slurm_tool(name: str, env: dict) -> str | None:
    return shutil.which(name, path=env.get("PATH"))


def sbatch_submit(
    ctx: Context, script: Path, env: dict, log: Path
) -> tuple[str | None, str]:
    sbatch = slurm_tool("sbatch", env)
    if not sbatch:
        return None, "找不到 sbatch"
    result = run_command(
        [sbatch, "--parsable", str(script)],
        cwd=ctx.workspace,
        env=env,
        log=log,
        timeout_s=300,
        echo=False,
    )
    if result.code != 0:
        return None, "sbatch 提交失败：" + " | ".join(
            line for line in result.tail[-5:] if line.strip()
        )
    job_id = result.output.strip().splitlines()[-1].split(";")[0].strip()
    if not job_id.isdigit():
        return None, f"sbatch 返回无法识别的作业号：{result.output.strip()[-200:]}"
    _SLURM_JOBS.add(job_id)
    return job_id, "ok"


def slurm_state(job_id: str, env: dict) -> tuple[str | None, str]:
    """Live state from squeue, or (None, '') when the job has left the queue."""
    squeue = slurm_tool("squeue", env)
    if not squeue:
        return None, ""
    try:
        out = subprocess.run(
            [squeue, "-h", "-j", job_id, "-o", "%T|%r"],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN", "squeue 超时"
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if not line:
        return None, ""
    state, _, reason = line.partition("|")
    return state.strip(), reason.strip()


def slurm_final(job_id: str, env: dict) -> dict:
    sacct = slurm_tool("sacct", env)
    if not sacct:
        return {"state": "UNKNOWN", "exit": None, "note": "没有 sacct"}
    for _ in range(6):
        try:
            out = subprocess.run(
                [
                    sacct,
                    "-X",
                    "-n",
                    "-P",
                    "-j",
                    job_id,
                    "-o",
                    "State,ExitCode,Elapsed,NodeList",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired):
            time.sleep(5)
            continue
        line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        if line:
            parts = (line.split("|") + ["", "", "", ""])[:4]
            return {
                "state": parts[0].split()[0],
                "exit": parts[1],
                "elapsed": parts[2],
                "nodes": parts[3],
            }
        time.sleep(5)
    return {"state": "UNKNOWN", "exit": None, "note": "sacct 没有记录"}


def scancel_own(env: dict, job_ids: Iterable[str] | None = None) -> list[str]:
    """Cancel jobs this run submitted, and nothing else."""
    scancel = slurm_tool("scancel", env)
    cancelled = []
    for job_id in sorted(job_ids if job_ids is not None else _SLURM_JOBS):
        if job_id in _SLURM_JOBS and scancel:
            subprocess.run([scancel, job_id], capture_output=True, timeout=120, env=env)
            cancelled.append(job_id)
    return cancelled


def slurm_wait(ctx: Context, job_id: str, env: dict, run_timeout_s: float) -> dict:
    """Track a job until it ends, cancelling it on queue or run timeout."""
    poll = float(ctx.cfg["scheduler"]["poll_seconds"])
    max_queue = float(ctx.cfg["scheduler"]["max_queue_wait_minutes"]) * 60.0
    submitted = time.time()
    started = None
    last_state = None
    while True:
        state, reason = slurm_state(job_id, env)
        if state is None:
            break
        if state != last_state:
            ctx.note(f"Slurm 作业 {job_id}：{state} {reason}")
            last_state = state
        if state in {"PENDING", "CONFIGURING"}:
            if time.time() - submitted > max_queue:
                scancel_own(env, [job_id])
                return {
                    "state": "QUEUE_TIMEOUT",
                    "reason": reason,
                    "queued_s": round(time.time() - submitted),
                }
        elif started is None:
            started = time.time()
        if started and time.time() - started > run_timeout_s:
            scancel_own(env, [job_id])
            return {"state": "RUN_TIMEOUT", "ran_s": round(time.time() - started)}
        time.sleep(poll)
    final = slurm_final(job_id, env)
    final["queued_s"] = round((started or time.time()) - submitted)
    final["ran_s"] = round(time.time() - started) if started else None
    return final


def record_job(ctx: Context, job: dict) -> None:
    ctx.state.manifest["slurm_jobs"].append(job)
    ctx.state.save()


def run_slurm_batch(ctx: Context, batch: list[str]) -> None:
    """Run consecutive stages with one job spec inside a single Slurm job."""
    key = job_key(batch[0])
    gpus = max(stage_gpus(ctx.cfg, sid)[0] for sid in batch)
    name = f"smacc-{'-'.join(batch)}"
    out = ctx.run_dir / "logs" / f"slurm_{batch[0]}_%j.out"
    script = ctx.run_dir / "slurm" / f"job_{'_'.join(batch)}.sbatch"
    python3 = os.environ.get("SUPERMIMIC_ACCEPTANCE_PYTHON", sys.executable)
    body = [
        *sbatch_header(ctx, key, name, 1, gpus, out),
        "set -u",
        f"cd {shlex.quote(str(ctx.workspace))}",
        'echo "[smacc] job ${SLURM_JOB_ID:-?} on $(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
        shlex.join(
            [
                python3,
                str(Path(__file__).resolve()),
                "--worker",
                "--run-dir",
                str(ctx.run_dir),
                "--stages",
                ",".join(batch),
            ]
        ),
    ]
    atomic_write(script, "\n".join(body) + "\n")
    for sid in batch:
        ctx.state.set(sid, PENDING, "已提交 Slurm，排队中", placement=f"slurm:{name}")
    env = ctx.clean_env()
    job_id, why = sbatch_submit(ctx, script, env, ctx.log(batch[0]))
    if job_id is None:
        for sid in batch:
            ctx.state.set(sid, BLOCKED, why, finished_at=now_iso())
        return
    ctx.note(f"阶段 {','.join(batch)} 已提交为 Slurm 作业 {job_id}")
    timeout = sum(ctx.timeout(TIMEOUT_KEY[sid]) for sid in batch) + 600
    job = {
        "job_id": job_id,
        "stages": batch,
        "script": str(script),
        "submitted_at": now_iso(),
    }
    try:
        final = slurm_wait(ctx, job_id, env, timeout)
    finally:
        ctx.state.reload()
    job.update(final)
    record_job(ctx, job)
    out_file = ctx.run_dir / "logs" / f"slurm_{batch[0]}_{job_id}.out"
    tail = (
        out_file.read_text(encoding="utf-8", errors="replace")[-20000:]
        if out_file.is_file()
        else ""
    )
    for index, sid in enumerate(batch):
        record = ctx.state.stage(sid)
        if record["status"] in FINAL_STATES:
            continue
        if final["state"] == "QUEUE_TIMEOUT":
            ctx.state.set(
                sid,
                BLOCKED,
                f"Slurm 作业 {job_id} 排队超过 scheduler.max_queue_wait_minutes，已 scancel",
                finished_at=now_iso(),
            )
            continue
        if (
            record["status"] == RUNNING
            or index == 0
            or all(ctx.state.stage(b)["status"] == PASS for b in batch[:index])
        ):
            ctx.state.finish(
                sid,
                StageResult(
                    FAIL,
                    f"Slurm 作业 {job_id} 以 {final['state']}（{final.get('exit')}）结束，阶段未正常收尾"
                    "（可能节点故障、OOM、作业超时或 worker 未启动）",
                    diagnosis=diagnose(tail)
                    or [line for line in tail.splitlines() if line.strip()][-5:],
                ),
            )
        else:
            ctx.state.set(
                sid,
                NOT_RUN,
                f"Slurm 作业 {job_id} 在此阶段前结束",
                finished_at=now_iso(),
            )


TIMEOUT_KEY = {
    "00": "config",
    "01": "install",
    "02": "config",
    "03": "unit_test_file",
    "04": "smoke",
    "05": "checkpoint",
    "06": "evaluation",
    "07": "profiling",
    "08": "two_gpu",
    "09": "multi_gpu",
    "10": "paper_scale",
    "10b": "paper_scale",
    "11": "multinode",
    "12": "learning_signal",
    "13": "long_training",
    "13b": "learning_signal",
    "14": "config",
}


# --------------------------------------------------------------------------
# Stage 11: multi-node
# --------------------------------------------------------------------------


def multinode_job_script(
    ctx: Context, item: dict, log_dir: Path, out: Path | None
) -> list[str]:
    mn = ctx.cfg["multinode"]
    nodes, gpn = int(mn["nodes"]), int(mn["gpus_per_node"])
    port = (
        int(mn["ray_port"])
        if mn["ray_port"]
        else 20000 + int(ctx.run_id[-4:], 16) % 10000
    )
    probe_paths = [
        str(ctx.venv_python),
        str(ctx.repo / "examples" / "embodiment" / "train_supermimic.py"),
        str(ctx.lafan1),
        str(ctx.run_dir),
    ]
    probe = (
        "for p in " + " ".join(shlex.quote(p) for p in probe_paths) + "; do "
        'test -e "$p" || { echo "SMACC_MISSING $p on $(hostname)"; exit 3; }; done; '
        f'touch {shlex.quote(str(ctx.run_dir / "slurm"))}/fs_probe_$(hostname)_$SLURM_PROCID && echo "SMACC_FS_OK $(hostname)"'
    )
    overrides = [
        o
        for o in [*data_override(ctx, item["config"]), *item["overrides"]]
        if not o.startswith(("cluster.num_nodes", "cluster.component_placement"))
    ]
    header = (
        sbatch_header(ctx, "multinode", "smacc-multinode", nodes, gpn, out)
        if out
        else ["#!/bin/bash"]
    )
    return [
        *header,
        "set -u",
        f"cd {shlex.quote(str(ctx.repo))}",
        f"export GPUS_PER_NODE={gpn} RAY_PORT={port} REPO_PATH={shlex.quote(str(ctx.repo))}",
        f"srun --nodes={nodes} --ntasks={nodes} --ntasks-per-node=1 bash -c {shlex.quote(probe)} || {{ echo SMACC_FS_PROBE_FAILED; exit 3; }}",
        shlex.join(
            [
                "bash",
                "examples/embodiment/slurm_supermimic.sh",
                item["config"],
                *overrides,
                f"runner.logger.log_path={log_dir}",
            ]
        ),
        "code=$?",
        'echo "SMACC_EXIT_CODE=$code"',
        "exit $code",
    ]


def stage_multinode(ctx: Context) -> StageResult:
    sid = "11"
    mn = ctx.cfg["multinode"]
    nodes, gpn = int(mn["nodes"]), int(mn["gpus_per_node"])
    world = nodes * gpn
    item = train_overrides(ctx, sid)[0]
    log_dir = ctx.run_dir / "checkpoints" / "11_multinode"
    env = ctx.venv_env(world)
    slurm = slurm_info(env)
    results = {"nodes": nodes, "gpus_per_node": gpn, "world_size": world}
    if slurm["in_allocation"]:
        script = ctx.run_dir / "slurm" / "multinode_in_allocation.sh"
        atomic_write(
            script, "\n".join(multinode_job_script(ctx, item, log_dir, None)) + "\n"
        )
        result = ctx.run(sid, ["bash", str(script)], timeout_key="multinode", env=env)
        output = result.output
        code = result.code
    else:
        out = ctx.run_dir / "logs" / "11_multinode_slurm_%j.out"
        script = ctx.run_dir / "slurm" / "multinode.sbatch"
        atomic_write(
            script, "\n".join(multinode_job_script(ctx, item, log_dir, out)) + "\n"
        )
        job_id, why = sbatch_submit(ctx, script, env, ctx.log(sid))
        if job_id is None:
            return StageResult(BLOCKED, why, results=results)
        final = slurm_wait(ctx, job_id, env, ctx.timeout("multinode"))
        record_job(
            ctx, {"job_id": job_id, "stages": [sid], "script": str(script), **final}
        )
        results["slurm"] = final
        if final["state"] == "QUEUE_TIMEOUT":
            return StageResult(
                BLOCKED, f"多节点作业 {job_id} 排队超时，已 scancel", results=results
            )
        out_file = ctx.run_dir / "logs" / f"11_multinode_slurm_{job_id}.out"
        output = (
            out_file.read_text(encoding="utf-8", errors="replace")
            if out_file.is_file()
            else ""
        )
        with open(ctx.log(sid), "a", encoding="utf-8") as handle:
            handle.write(f"\n# Slurm output {out_file}\n{output}")
        match = re.search(r"SMACC_EXIT_CODE=(\d+)", output)
        code = int(match.group(1)) if match else None
    cleanup_run_processes(ctx.run_id)
    if "SMACC_FS_PROBE_FAILED" in output:
        missing = re.findall(r"SMACC_MISSING (.+)", output)
        return StageResult(
            BLOCKED, "计算节点看不到共享目录：" + "；".join(missing[:4]), code, results
        )
    results["fs_probe_nodes"] = len(set(re.findall(r"SMACC_FS_OK (\S+)", output)))
    parsed = parse_training_log(output)
    run = {
        "result": CommandResult(code=code, seconds=0.0, output=output),
        "metrics": parsed["metrics"],
        "evals": parsed["evals"],
        "steps": checkpoint_steps(log_dir)[0],
    }
    results["train"] = summarize_run(run)
    if code != 0:
        return StageResult(
            FAIL,
            f"多节点训练失败（退出码 {code}）",
            code,
            results,
            diagnose(output[-20000:]),
        )
    problems = check_training(
        run, expect_updates=int(mn["total_updates"])
    ) + check_two_rank(run, world)
    ckpt = latest_checkpoint(log_dir)
    meta = verify_ckpt(ctx, sid, ckpt) if ckpt else None
    results["checkpoint_meta"] = meta
    ranks = {f"rank_{r}.pt" for r in range(world)}
    if (
        not meta
        or meta.get("world_size") != world
        or not ranks <= set(meta.get("files", []))
    ):
        problems.append(
            f"共享存储上的检查点缺少 world_size={world} 或 {world} 个 rank 文件"
        )
    if problems:
        return StageResult(
            FAIL, "多节点检查失败：" + "；".join(problems[:6]), code, results
        )
    return StageResult(
        PASS,
        f"{nodes} 节点 × {gpn} GPU：param_divergence 恒为 0，检查点含 {world} 个 rank 文件",
        0,
        results,
    )


# --------------------------------------------------------------------------
# Stages 12, 13, 13b: learning signal, long training, BONES
# --------------------------------------------------------------------------

EVAL_KEYS = (
    "eval/normalized_return",
    "eval/fail_rate",
    "eval/episode_length_s",
    "eval/body_tracking",
)
TRAIN_KEYS = (
    "wm/one_step_loss",
    "wm/k_step_loss",
    "critic/loss",
    "actor/objective",
    "actor/grad_norm",
    "wm/grad_norm",
    "sim/reward_mean",
    "train/nonfinite_skips",
)


def curve_summary(run: dict) -> dict:
    """First and last values of the logged training and eval metrics (no thresholds)."""
    out = {}
    for key in TRAIN_KEYS:
        series = [row[key] for row in run["metrics"] if key in row]
        if series:
            out[key] = {"first": series[0], "last": series[-1], "n": len(series)}
    for key in EVAL_KEYS:
        series = [row[key] for row in run["evals"] if key in row]
        if series:
            out[key] = {
                "first": series[0],
                "last": series[-1],
                "n": len(series),
                "delta": series[-1] - series[0],
            }
    return out


def train_and_eval(
    ctx: Context,
    sid: str,
    item: dict,
    timeout_key: str,
    eval_extra: list[str] | None = None,
) -> tuple[dict, list[str], dict]:
    run = train(ctx, sid, item, timeout_key=timeout_key)
    info = {"train": summarize_run(run), "curves": curve_summary(run)}
    if run["result"].code != 0:
        return info, [f"{item['label']} 训练失败（退出码 {run['result'].code}）"], run
    problems = check_training(
        run,
        expect_updates=next(
            (
                int(o.split("=")[1])
                for o in item["overrides"]
                if o.startswith("algorithm.total_updates=")
            ),
            None,
        ),
    )
    eval_config = EVAL_FOR.get(item["config"])
    if eval_config:
        root = run["log_dir"] / item["config"]
        roots = (
            [p for p in run["log_dir"].iterdir() if p.is_dir()]
            if run["log_dir"].is_dir()
            else []
        )
        root = root if root.is_dir() else (roots[0] if roots else root)
        extra = (
            eval_extra
            if eval_extra is not None
            else (velocity_eval_overrides(ctx) if "velocity" in eval_config else [])
        )
        result, einfo, eproblems = run_eval(
            ctx, sid, item["label"], eval_config, root, extra
        )
        info["final_eval"] = einfo
        problems += [f"{item['label']} 评测：{p}" for p in eproblems]
    return info, problems, run


def stage_learning(ctx: Context) -> StageResult:
    sid = "12"
    results, problems = {}, []
    for item in train_overrides(ctx, sid):
        info, p, run = train_and_eval(ctx, sid, item, "learning_signal")
        results[item["label"]] = info
        if run["result"].code != 0:
            return fail_from(
                run["result"], f"学习信号运行 {item['label']} 失败", **results
            )
        problems += p
    if problems:
        return StageResult(
            FAIL, "学习信号阶段数值或评测异常：" + "；".join(problems[:6]), 0, results
        )
    return StageResult(
        PASS,
        f"{len(results)} 个变体完成 {ctx.cfg['learning_signal']['updates']} 次更新，曲线已记录（不设改善阈值）",
        0,
        results,
    )


def stage_long(ctx: Context) -> StageResult:
    sid = "13"
    results, problems = {}, []
    for item in train_overrides(ctx, sid):
        info, p, run = train_and_eval(ctx, sid, item, "long_training")
        results[item["label"]] = info
        if run["result"].code != 0:
            return fail_from(run["result"], f"长训练 {item['label']} 失败", **results)
        problems += p
    if problems:
        return StageResult(
            FAIL, "长训练结果异常：" + "；".join(problems[:6]), 0, results
        )
    return StageResult(
        PASS, "长训练与评测完成，结果见 metrics/ 与 evaluations/", 0, results
    )


def stage_bones(ctx: Context) -> StageResult:
    sid = "13b"
    item = train_overrides(ctx, sid)[0]
    info, problems, run = train_and_eval(ctx, sid, item, "learning_signal")
    if run["result"].code != 0:
        return fail_from(run["result"], "BONES-SEED 训练失败", **info)
    if problems:
        return StageResult(
            FAIL, "BONES-SEED 结果异常：" + "；".join(problems[:6]), 0, info
        )
    return StageResult(
        PASS,
        f"BONES-SEED {ctx.cfg['bones']['train_updates']} 次更新与评测完成",
        0,
        info,
    )


def config_requirements(ctx: Context, sid: str) -> list[str]:
    """Configuration a stage needs before it can be scheduled."""
    cfg, missing = ctx.cfg, []
    if sid == "11":
        missing += [
            f"multinode.{k}"
            for k in ("nodes", "gpus_per_node")
            if not cfg["multinode"][k]
        ]
    if sid == "13":
        lt = cfg["long_training"]
        for task in ("motion", "velocity"):
            if task == "velocity" and not lt["velocity"]["enabled"]:
                continue
            missing += [
                f"long_training.{task}.{k}"
                for k in ("gpus", "total_updates")
                if not lt[task][k]
            ]
    if sid == "13b" and not (ctx.bones and ctx.bones.is_dir()):
        missing.append("data.bones_seed（目录不存在或未设置）")
    return missing


# --------------------------------------------------------------------------
# Stage 14: summary and conclusions
# --------------------------------------------------------------------------


def conclusions(m: dict) -> tuple[dict, list[str]]:
    st = {sid: s["status"] for sid, s in m["stages"].items()}
    msg = {sid: s["message"] for sid, s in m["stages"].items()}
    out, advice = {}, []
    if st["01"] == PASS:
        out["install"] = (
            "是：全新 venv 仅通过 requirements/install.sh 安装，uv pip check 通过，关键模块来自新环境"
        )
    elif st["01"] == NOT_APPLICABLE:
        out["install"] = "未验证：" + (msg["01"] or "未执行安装阶段")
    else:
        out["install"] = f"否（{st['01']}）：{msg['01']}"
    engineering = [
        sid
        for sid in ("02", "03", "04", "05", "06", "07", "08", "09", "10", "10b", "11")
        if sid in m["selected"]
    ]
    failed = [sid for sid in engineering if st[sid] == FAIL]
    blocked = [sid for sid in engineering if st[sid] in {BLOCKED, NOT_RUN}]
    passed = [sid for sid in engineering if st[sid] == PASS]
    if failed:
        out["engineering"] = f"否：阶段 {', '.join(failed)} 失败"
    elif blocked:
        out["engineering"] = (
            f"部分：通过 {', '.join(passed) or '无'}；未完成 {', '.join(blocked)}"
        )
    elif passed:
        out["engineering"] = f"是（{m['mode']} 模式）：通过 {', '.join(passed)}"
    else:
        out["engineering"] = "未运行"
    learning = m["stages"]["12"]
    if st["12"] == PASS:
        out["numerics"] = "正常：学习信号阶段全部指标有限、计数单调（见 12 的 curves）"
    elif st["04"] == PASS:
        out["numerics"] = (
            "仅 Smoke 级别：几次更新内指标有限、计数单调；未做长于 Smoke 的数值诊断"
            if st["12"] != FAIL
            else f"异常：{msg['12']}"
        )
    else:
        out["numerics"] = "未判断：训练阶段没有通过"
    if st["12"] == PASS:
        deltas = []
        for name, info in learning["results"].items():
            curves = info.get("curves", {}) if isinstance(info, dict) else {}
            for key in ("eval/normalized_return", "eval/fail_rate"):
                if key in curves:
                    deltas.append(
                        f"{name} {key}: {curves[key]['first']:.4g} → {curves[key]['last']:.4g}"
                    )
        out["improvement"] = (
            "观测到的变化：" + "；".join(deltas) if deltas else "没有训练内评测记录"
        ) + "。单种子、短预算，不能据此断言学习有效"
    else:
        out["improvement"] = "未判断：没有完成学习信号阶段"
    if st["13"] == PASS:
        out["paper"] = (
            "长训练已完成，结果见 13 阶段；与论文数值的对比需要人工逐项核对，脚本不自动判定复现"
        )
    else:
        out["paper"] = "未复现：本次没有完成论文预算的长训练，不能声称复现论文结果"
    first = m.get("first_failure")
    if first:
        s = m["stages"][first]
        advice.append(f"先查看 `{s['log']}`，处理阶段 {first} 的失败：{s['message']}")
        if first == "01" or any("依赖" in d for d in s.get("diagnosis", [])):
            advice.append(
                "依赖问题需修复 requirements 后，用一个新的 paths.venv 重新运行全部验收；不要在旧环境中补装"
            )
    for sid in m["stage_order"]:
        if st[sid] == BLOCKED:
            advice.append(f"阶段 {sid} 被阻塞：{msg[sid]}")
    if m.get("simulation"):
        out = {k: f"（模拟，不代表真实验收）{v}" for k, v in out.items()}
    return out, advice


def stage_summary(ctx: Context) -> StageResult:
    m = ctx.state.manifest
    after = git_snapshot(ctx.repo)
    before = m.get("git", {})
    unchanged = after.get("status") == before.get("status")
    new_ignored = sorted(
        set(after.get("ignored") or []) - set(before.get("ignored") or [])
    )
    m["repo_unchanged"] = unchanged
    m["repo_new_ignored"] = new_ignored
    m["conclusions"], m["advice"] = conclusions(m)
    results = {"repo_unchanged": unchanged, "new_ignored_files": new_ignored[:20]}
    if not unchanged:
        return StageResult(
            FAIL,
            "运行期间 RLinf 仓库的 git status 发生变化（验收不应修改仓库）",
            results=results,
        )
    return StageResult(
        PASS,
        "汇总完成"
        + (
            f"；仓库新增被忽略文件 {len(new_ignored)} 个（见结果）"
            if new_ignored
            else ""
        ),
        0,
        results,
    )


STAGE_FUNCS: dict[str, Callable[[Context], StageResult]] = {
    "00": stage_preflight,
    "01": stage_install,
    "02": stage_config,
    "03": stage_unit_tests,
    "04": stage_smoke,
    "05": stage_checkpoint,
    "06": stage_evaluation,
    "07": stage_profiling,
    "08": stage_nccl,
    "09": stage_multigpu_fault,
    "10": stage_paper_scale,
    "10b": stage_paper_2gpu,
    "11": stage_multinode,
    "12": stage_learning,
    "13": stage_long,
    "13b": stage_bones,
    "14": stage_summary,
}


# --------------------------------------------------------------------------
# Simulation (self-test of this script only)
# --------------------------------------------------------------------------

SIM_METRICS = (
    "[INFO RLinf] supermimic metrics {'train/global_update': %d.0, 'sim_vector_steps': %d.0, "
    "'env_transitions': %d.0, 'wm_optimizer_steps': %d.0, 'actor_optimizer_steps': %d.0, "
    "'critic_optimizer_steps': %d.0, 'train/nonfinite_skips': 0.0, 'wm/one_step_loss': 0.5}"
)


def simulated_stage(ctx: Context, sid: str) -> StageResult:
    """Mock stage: a real subprocess with scripted output, exit code or hang."""
    kind, _, target = (ctx.simulate or "success").partition("-at:")
    hit = target == sid
    if hit and kind == "block":
        return StageResult(BLOCKED, f"模拟：阶段 {sid} 缺少资源")
    lines = [
        f"print('SIMULATED stage {sid}: this is not a SuperMimic run', flush=True)"
    ]
    for step in (1, 2):
        row = SIM_METRICS % (step, 10 * step, 80 * step, 10 * step, step, 20 * step)
        lines.append(f"print({row!r}, flush=True)")
    if hit and kind == "fail":
        lines.append(
            "import sys; print('Traceback (most recent call last):'); print('RuntimeError: CUDA error: simulated failure', flush=True); sys.exit(7)"
        )
    if hit and kind in {"timeout", "hang"}:
        lines.append("import time; time.sleep(3600)")
    timeout_s = 3.0 if hit and kind == "timeout" else None
    result = ctx.run(
        sid,
        [sys.executable, "-c", "\n".join(lines)],
        timeout_key=TIMEOUT_KEY[sid],
        env=ctx.clean_env(),
        timeout_s=timeout_s,
    )
    parsed = parse_training_log(result.output)
    results = {"simulated": True, "metric_lines_parsed": len(parsed["metrics"])}
    if result.code != 0:
        return fail_from(result, f"模拟阶段 {sid} 失败", **results)
    return StageResult(PASS, f"模拟阶段 {sid} 通过（Mock，不代表真实验收）", 0, results)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def precheck(ctx: Context, sid: str, batch: tuple = ()) -> tuple[str, str] | None:
    """Status a stage gets without running, or None when it may run now."""
    m = ctx.state.manifest
    if m.get("aborted"):
        return NOT_RUN, f"运行被 {m['aborted']} 中止"
    if m.get("first_failure"):
        return NOT_RUN, f"阶段 {m['first_failure']} 失败后停止"
    for dep in STAGE_BY_ID[sid].deps:
        status = ctx.state.stage(dep)["status"]
        if status in {PASS, NOT_APPLICABLE} or dep in batch:
            continue
        if status == BLOCKED:
            return BLOCKED, f"依赖阶段 {dep} 被阻塞"
        return NOT_RUN, f"依赖阶段 {dep} 状态为 {status}"
    if not ctx.simulate:
        missing = config_requirements(ctx, sid)
        if missing:
            return BLOCKED, "缺少配置：" + ", ".join(missing)
    return None


def run_stage(ctx: Context, sid: str, placement: str) -> StageResult:
    spec = STAGE_BY_ID[sid]
    ctx.state.start(sid, placement)
    with open(ctx.log(sid), "a", encoding="utf-8") as handle:
        handle.write(
            f"# stage {sid} {spec.name} start {now_iso()} placement={placement} host={socket.gethostname()}\n"
        )
    ctx.note(f"阶段 {sid} {spec.title} 开始（{placement}）")
    func = (
        STAGE_FUNCS[sid]
        if (not ctx.simulate or sid == "14")
        else (lambda c: simulated_stage(c, sid))
    )
    try:
        result = func(ctx)
    except Interrupted as stop:
        result = StageResult(
            FAIL,
            f"收到 {signal.Signals(stop.signum).name}，运行中止",
            diagnosis=["用户或调度器中断"],
        )
        ctx.state.manifest["aborted"] = signal.Signals(stop.signum).name
        ctx.state.finish(sid, result)
        raise
    except Exception as error:
        text = traceback.format_exc()
        with open(ctx.log(sid), "a", encoding="utf-8") as handle:
            handle.write(f"\n# acceptance script error\n{text}\n")
        result = StageResult(
            FAIL,
            f"验收脚本在阶段 {sid} 内部出错：{type(error).__name__}: {error}",
            diagnosis=[text.splitlines()[-1]],
        )
    finally:
        cleanup_run_processes(ctx.run_id)
    ctx.state.finish(sid, result)
    ctx.note(f"阶段 {sid} {result.status}：{result.message}")
    return result


def orchestrate(ctx: Context) -> None:
    order = ctx.state.manifest["stage_order"]
    gpus = detect_gpus(os.environ)
    slurm = slurm_info(os.environ)
    for index, sid in enumerate(order):
        if sid == "14":
            continue
        if ctx.state.stage(sid)["status"] in FINAL_STATES:
            continue
        pre = precheck(ctx, sid)
        if pre:
            ctx.state.set(sid, *pre)
            continue
        placement, reason = decide_placement(ctx, sid, gpus, slurm)
        if placement == "blocked":
            ctx.state.set(sid, BLOCKED, reason)
            continue
        if placement == "local" or STAGE_BY_ID[sid].resource == "multinode":
            run_stage(ctx, sid, f"{placement}: {reason}")
            continue
        batch = [sid]
        key = (job_key(sid), stage_gpus(ctx.cfg, sid)[0])
        for nxt in order[index + 1 :]:
            if nxt == "14" or ctx.state.stage(nxt)["status"] in FINAL_STATES:
                break
            if (
                STAGE_BY_ID[nxt].resource in {"none", "multinode"}
                or (job_key(nxt), stage_gpus(ctx.cfg, nxt)[0]) != key
            ):
                break
            if (
                precheck(ctx, nxt, tuple(batch))
                or decide_placement(ctx, nxt, gpus, slurm)[0] != "slurm"
            ):
                break
            batch.append(nxt)
        run_slurm_batch(ctx, batch)


def worker_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    state = RunState.load(run_dir)
    m = state.manifest
    cfg = m["config"]
    ctx = Context(
        args, cfg, Path(m["workspace"]), Path(m["repo"]), m["run_id"], run_dir, state
    )
    ctx.simulate = m.get("simulation")
    ctx.inside_job = True
    install_signal_handlers()
    stages = [s for s in args.stages.split(",") if s]
    try:
        for sid in stages:
            state.reload()
            if state.stage(sid)["status"] in FINAL_STATES:
                continue
            pre = precheck(ctx, sid, tuple(stages))
            if pre:
                state.set(sid, *pre)
                continue
            run_stage(
                ctx,
                sid,
                f"slurm:{os.environ.get('SLURM_JOB_ID', '?')}@{socket.gethostname()}",
            )
    except Interrupted:
        for sid in stages:
            if state.stage(sid)["status"] == PENDING:
                state.set(sid, NOT_RUN, "Slurm 作业被中断")
        return 128 + 15
    finally:
        cleanup_run_processes(ctx.run_id)
    return 0


def install_signal_handlers() -> None:
    def handler(signum: int, _frame: object) -> NoReturn:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise Interrupted(signum)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def final_exit_code(m: dict) -> int:
    if m.get("aborted"):
        return 128 + int(getattr(signal, m["aborted"], signal.SIGTERM))
    if m.get("script_error"):
        return 1
    statuses = [m["stages"][sid]["status"] for sid in m["stage_order"]]
    if FAIL in statuses:
        return 1
    if BLOCKED in statuses or NOT_RUN in statuses:
        return 2
    return 0


def mem_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        return None
    return None


def selected_stages(mode: str, only: str | None) -> list[str]:
    chosen = [s.sid for s in STAGES if mode in s.modes]
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        unknown = wanted - set(STAGE_BY_ID)
        if unknown:
            raise AcceptanceError(
                f"--stages 含未知阶段 {sorted(unknown)}；可选 {[s.sid for s in STAGES]}"
            )
        chosen = [sid for sid in chosen if sid in wanted or sid in {"00", "14"}]
    return chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="supermimic_acceptance.py",
        description="SuperMimic 集群一键验收（放在 RLinf 仓库旁边运行，只依赖 Python 3 标准库）。",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
        help="smoke：安装到评测的快速验收；full：完整矩阵（默认）",
    )
    parser.add_argument(
        "--config", help="YAML 配置（默认：脚本同目录的 supermimic_acceptance.yaml）"
    )
    parser.add_argument(
        "--repo", help="RLinf 仓库路径（默认：配置 paths.repo，或脚本同目录的 RLinf/）"
    )
    parser.add_argument("--results", help="结果根目录（覆盖 paths.results）")
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印阶段、命令、资源与配置，不执行"
    )
    parser.add_argument(
        "--stages", help="只运行这些阶段（逗号分隔，如 03,04）；00 与 14 总会运行"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="把子进程输出同时打印到终端"
    )
    parser.add_argument("--simulate", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.simulate and not re.fullmatch(
        r"success|(fail|timeout|hang|block)-at:\w+", args.simulate
    ):
        parser.error(
            "--simulate 取值：success、fail-at:<阶段>、timeout-at:<阶段>、hang-at:<阶段>、block-at:<阶段>"
        )
    if args.worker and not (args.run_dir and args.stages):
        parser.error("--worker 需要 --run-dir 与 --stages")
    return args


def dry_run(ctx: Context) -> int:
    m = ctx.state.manifest
    gpus = detect_gpus(os.environ)
    slurm = slurm_info(os.environ)
    print(
        f"# SuperMimic 验收 dry-run（不执行任何命令，不创建文件）\n仓库：{ctx.repo}\n结果目录：{ctx.run_dir.parent}\nvenv：{ctx.venv}\n模式：{m['mode']}"
    )
    print(
        f"Git：{m['git'].get('commit')}（{m['git'].get('branch')}），未提交改动 {m['git'].get('dirty_files')} 个"
    )
    print(
        f"本节点可见 GPU：{len(gpus)}；Slurm：allocation={slurm['in_allocation']} sbatch={'有' if slurm['sbatch'] else '无'}"
    )
    problems = check_config_types(ctx.cfg)
    if ctx.lafan1 is None:
        problems.append("data.lafan1 未设置（必填：LAFAN1 G1 片段目录）")
    elif not ctx.lafan1.is_dir():
        problems.append(f"data.lafan1={ctx.lafan1} 不存在")
    if problems:
        print("配置问题（正式运行时会在 00 阶段报出）：\n  " + "\n  ".join(problems))
    print("venv 路径中的 {run_id} 会替换为本次运行 ID；以下命令中的 <run_id> 同理。")
    print("\n## 阶段计划")
    for sid in m["stage_order"]:
        spec = STAGE_BY_ID[sid]
        status = ctx.state.stage(sid)["status"]
        if status == NOT_APPLICABLE:
            print(
                f"- {sid} {spec.title}: NOT_APPLICABLE（{ctx.state.stage(sid)['message']}）"
            )
            continue
        missing = config_requirements(ctx, sid)
        placement, reason = decide_placement(ctx, sid, gpus, slurm)
        where = (
            f"BLOCKED（缺少配置：{', '.join(missing)}）"
            if missing
            else f"{placement}（{reason}）"
        )
        print(f"- {sid} {spec.title}: {where}")
        if sid == "01" and ctx.cfg["install"]["mode"] == "fresh":
            print(f"    $ {shlex.join(install_command(ctx))}   # cwd={ctx.repo}")
        for item in train_overrides(ctx, sid) if not missing else []:
            log_dir = ctx.run_dir / "checkpoints" / f"{sid}_{item['label']}"
            print(
                f"    $ {shlex.join(training_command(ctx, item, log_dir))}   # GPUs={item['gpus']}"
            )
        if placement == "slurm" and not missing:
            header = sbatch_header(
                ctx,
                job_key(sid),
                f"smacc-{sid}",
                1 if sid != "11" else int(ctx.cfg["multinode"]["nodes"]),
                stage_gpus(ctx.cfg, sid)[0],
                ctx.run_dir / "logs" / f"slurm_{sid}_%j.out",
            )
            print(
                "    sbatch: "
                + " ".join(h[8:] for h in header if h.startswith("#SBATCH"))
            )
    print("\n## 生效配置\n" + json.dumps(ctx.cfg, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)
    script_dir = Path(__file__).resolve().parent
    try:
        cfg_path = (
            Path(args.config).resolve()
            if args.config
            else script_dir / "supermimic_acceptance.yaml"
        )
        cfg = load_config(cfg_path if (args.config or cfg_path.is_file()) else None)
        problems = check_config_types(cfg)
        if problems:
            raise AcceptanceError("配置错误：\n  " + "\n  ".join(problems))
        repo = resolve_path(args.repo or cfg["paths"]["repo"] or "RLinf", script_dir)
        if not repo.is_dir():
            raise AcceptanceError(
                f"找不到 RLinf 仓库 {repo}；请把脚本放在 RLinf/ 旁边，或用 --repo / paths.repo 指定"
            )
        repo = repo.resolve()
        if is_inside(script_dir, repo):
            raise AcceptanceError(
                f"验收脚本位于仓库 {repo} 内部；请放到仓库外（例如它的父目录）"
            )
        results_root = resolve_path(args.results or cfg["paths"]["results"], script_dir)
        if is_inside(results_root, repo):
            raise AcceptanceError(
                f"结果目录 {results_root} 位于仓库内部；请改 paths.results"
            )
        mode = args.mode
        selected = selected_stages(mode, args.stages)
        if (
            args.stages
            and "01" not in selected
            and cfg["install"]["mode"] == "fresh"
            and not args.simulate
        ):
            raise AcceptanceError(
                "--stages 不含 01 时需要 install.mode=reuse（并指定已有的 paths.venv）"
            )
    except AcceptanceError as error:
        print(f"[supermimic-acceptance] {error}", file=sys.stderr)
        return 2
    git = git_snapshot(repo)
    header = {
        "mode": mode,
        "selected": selected,
        "simulation": args.simulate,
        "config": cfg,
        "config_file": str(cfg_path) if cfg_path.is_file() else None,
        "repo": str(repo),
        "workspace": str(script_dir),
        "command_line": shlex.join([sys.executable, *sys.argv]),
        "git": git,
        "environment": {
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "mem_gb": mem_gb(),
            "system_python": platform.python_version(),
            "gpus": [
                f"{g['name']} {g['memory_gb']} GB" for g in detect_gpus(os.environ)
            ],
            "driver": next((g["driver"] for g in detect_gpus(os.environ)), None),
        },
        "scheduler": {
            "mode": cfg["scheduler"]["mode"],
            **{
                k: v
                for k, v in slurm_info(os.environ).items()
                if k in {"in_allocation", "job_id"}
            },
        },
    }
    if args.dry_run:
        fake_dir = results_root / "<run_id>"
        state = RunState(
            fake_dir,
            {**header, "stages": {}, "stage_order": EXEC_ORDER, "run_id": "dryrun"},
        )
        state.manifest["stages"] = {
            s.sid: {"status": PENDING, "message": "", "log": ""} for s in STAGES
        }
        ctx = Context(args, cfg, script_dir, repo, "dryrun_0000", fake_dir, state)
        mark_exclusions(ctx, save=False)
        return dry_run(ctx)
    try:
        run_id, run_dir = make_run_dir(results_root, git.get("commit") or "")
    except OSError as error:
        print(
            f"[supermimic-acceptance] 结果目录 {results_root} 不可写：{error}",
            file=sys.stderr,
        )
        return 2
    header["run_dir"] = str(run_dir)
    state = RunState.create(run_dir, run_id, header)
    ctx = Context(args, cfg, script_dir, repo, run_id, run_dir, state)
    state.manifest["venv"] = str(ctx.venv)
    mark_exclusions(ctx, save=True)
    ctx.note(f"运行 {run_id}，结果目录 {run_dir}")
    install_signal_handlers()
    env = ctx.clean_env()
    try:
        orchestrate(ctx)
    except Interrupted as stop:
        name = signal.Signals(stop.signum).name
        state.manifest["aborted"] = name
        ctx.note(f"收到 {name}：取消本次 Slurm 作业、清理本次进程并写入报告")
        cancelled = scancel_own(env)
        if cancelled:
            for _ in range(12):
                if all(slurm_state(j, env)[0] is None for j in cancelled):
                    break
                time.sleep(5)
            state.reload()
            state.manifest["aborted"] = name
        for sid in state.manifest["stage_order"]:
            if state.stage(sid)["status"] == RUNNING:
                state.finish(sid, StageResult(FAIL, f"收到 {name}，运行中止"))
    except Exception:
        text = traceback.format_exc()
        state.manifest["script_error"] = text
        print(f"[supermimic-acceptance] 验收脚本内部错误：\n{text}", file=sys.stderr)
    finally:
        cleanup_run_processes(run_id)
        stop = state.manifest.get("aborted")
        if stop:
            reason = f"运行被 {stop} 中止，未执行"
        elif state.manifest.get("script_error"):
            reason = "验收脚本内部错误，未执行"
        else:
            reason = "未执行"
        for sid in state.manifest["stage_order"]:
            if sid != "14" and state.stage(sid)["status"] in {PENDING, RUNNING}:
                state.set(sid, NOT_RUN, reason)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        run_stage(ctx, "14", "local")
        state.manifest["finished_at"] = now_iso()
        state.manifest["exit_code"] = final_exit_code(state.manifest)
        state.save()
    print(
        f"\n[supermimic-acceptance] 报告：{run_dir / 'summary.md'}\n[supermimic-acceptance] 退出码：{state.manifest['exit_code']}"
    )
    for key, value in (state.manifest.get("conclusions") or {}).items():
        print(f"  - {key}: {value}")
    return state.manifest["exit_code"]


def mark_exclusions(ctx: Context, save: bool) -> None:
    m = ctx.state.manifest
    for spec in STAGES:
        record = m["stages"][spec.sid]
        if spec.sid not in m["selected"]:
            reason = (
                "不在 smoke 模式内"
                if m["mode"] not in spec.modes
                else "未被 --stages 选中"
            )
            record.update(status=NOT_APPLICABLE, message=reason)
            continue
        excluded = None if ctx.simulate else feature_exclusion(ctx.cfg, spec.sid)
        if excluded:
            record.update(status=NOT_APPLICABLE, message=f"配置关闭（{excluded}）")
    if save:
        ctx.state.save()


if __name__ == "__main__":
    sys.exit(main())
