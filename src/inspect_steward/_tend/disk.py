"""Trace logs filling the disk: the reading, the safe-to-remove decision, and the reclaim.

Inspect and scout each write a `trace-<pid>.log` per process into a shared data directory, gzip it when the process exits, and keep only the ten newest *files* (`rotate_trace_files`). What they never bound is the *size* of one: a long-running scan's trace log grows for as long as the scan runs, and a process that crashed left an uncompressed `.log` that no later run rotates away until ten newer ones exist. So a handful of files can hold tens of gigabytes while the count stays under the cap, and inspect's own housekeeping will not reclaim them. Steward is the thing already running on the box every ten minutes, so it is where a near-full disk gets noticed and the trace logs that are safe to remove get removed.

**Reactive, not proactive.** Nothing is removed while the disk has room. A reading is taken each turn, and only when free space is below the configured mark (`directives.disk_low`, on by default) does anything get deleted — and then only enough to cross back over it.

**Safe means the owner has exited.** A `.gz` is always safe: compression happens only at process exit, so the process is gone. A `trace-<pid>.log` is safe only when that pid is not a live process and is not our own. An unreadable or reused pid fails safe — the file is kept — and a file whose name is not `trace-<pid>.log[.gz]` is never touched, which is what leaves an `INSPECT_TRACE_FILE` at an operator-chosen path alone. **This is not the eval-log invariant's concern**: a trace log is an ephemeral diagnostic, not a result, and `_evalset.archive`'s *never delete a result* is about `logs/`, which nothing here reaches.

**The decision is pure.** `plan_cleanup` takes a free-space figure, a file listing, and a liveness predicate and returns what to remove — so `status` previews exactly what `tend` will delete, and the whole decision is testable without a full disk. The reading, the listing, and the unlink are the IO around it, and each never raises for the reason `_workspace.log` does not: a full disk is the condition this guards against, and failing the turn that would relieve it is the one outcome to avoid.
"""

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from platformdirs import user_data_path

from .._util.size import format_bytes
from .._worker import HostDisk
from .._workspace import OBSERVATION, JournalEvent

DEFAULT_RECLAIM_MARGIN = 512 * 1024**2
"""Bytes to reclaim past the mark, so a turn that frees space does not leave the disk one byte over the line and the next turn right back below it.

Half a gibibyte is a few turns of headroom at a typical trace-log growth rate — enough that the item clears and stays cleared rather than flapping on and off as the next log grows.
"""

TRACE_RE = re.compile(r"^trace-(\d+)\.log(\.gz)?$")
"""The name inspect and scout give a trace log: `trace-<pid>.log`, gzipped to `trace-<pid>.log.gz` on exit. The pid is the only thing in it, and it is what the safety check turns on."""


def _data_dir() -> Path:
    """Inspect's data directory, resolved the way inspect resolves it — `platformdirs` directly, so no directory is created as a side effect of reading one. Inspect's own `inspect_trace_dir()` calls `mkdir`, which a `status` must not do."""
    return user_data_path("inspect_ai")


def trace_dirs() -> list[Path]:
    """The directories inspect and scout write trace logs into.

    Both resolve under inspect's data directory — `traces` for inspect, `scout_traces` for scout — and share a mount in every deployment that does not symlink one elsewhere.
    """
    base = _data_dir()
    return [base / "traces", base / "scout_traces"]


def disk_path() -> Path | None:
    """A path on the trace logs' filesystem whose free space can be read.

    The trace directories themselves may not exist yet on a box that has not run inspect, so this walks up to the nearest ancestor that does — free space is the mount's, and any existing path on it answers for the same disk. `None` only where not even the home directory exists, which is not a real machine.
    """
    base = _data_dir()
    for candidate in [base, *base.parents]:
        if candidate.exists():
            return candidate
    return None


def parse_trace_pid(name: str) -> tuple[int | None, bool]:
    """The pid a trace log's name carries, and whether it is compressed.

    Args:
        name: A file name.

    Returns:
        `(pid, compressed)` for a `trace-<pid>.log[.gz]`, or `(None, False)` for a name that is not a trace log at all.
    """
    match = TRACE_RE.match(name)
    if match is None:
        return None, False
    return int(match.group(1)), match.group(2) is not None


@dataclass(frozen=True)
class TraceFile:
    """One trace log on disk, with the two facts the decision needs."""

    path: Path
    size: int
    pid: int
    compressed: bool


def read_trace_files(dirs: Sequence[Path]) -> list[TraceFile]:
    """Every trace log across the directories, with its size and pid.

    Never raises: a directory that does not exist is skipped, and a file that vanished or will not `stat` between the listing and the read is dropped — the same posture `_workspace.log.truncate_log` takes, because this runs to relieve a full disk and must not fail on one. Names that are not trace logs are omitted, so nothing outside inspect's own convention is ever a candidate.

    Args:
        dirs: The directories to read (`trace_dirs`).

    Returns:
        The trace logs found, in no particular order.
    """
    files: list[TraceFile] = []
    for directory in dirs:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            pid, compressed = parse_trace_pid(entry.name)
            if pid is None:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            files.append(
                TraceFile(path=entry, size=size, pid=pid, compressed=compressed)
            )
    return files


def safe_to_remove(file: TraceFile, *, alive: Callable[[int], bool]) -> bool:
    """Whether a trace log can be removed without touching a live process's own.

    A compressed file is always safe — inspect gzips a trace log only as the process exits, so a `.gz` is a dead process's by construction. An uncompressed one is safe only when its pid is neither alive nor our own: the owner has gone and left the file behind.

    Args:
        file: The candidate.
        alive: Whether a pid is a live process. Fails safe at the call site — an unreadable pid reads as alive, so the file is kept.

    Returns:
        Whether removing it is safe.
    """
    if file.compressed:
        return True
    return file.pid != os.getpid() and not alive(file.pid)


@dataclass(frozen=True)
class Removal:
    """One trace log the turn plans to remove, and why it is safe to."""

    path: Path
    size: int
    pid: int
    reason: Literal["dead_pid", "compressed"]


def plan_cleanup(
    *,
    free: int,
    low_mark: int,
    files: Sequence[TraceFile],
    alive: Callable[[int], bool],
    margin: int = DEFAULT_RECLAIM_MARGIN,
) -> list[Removal]:
    """Which trace logs to remove to get free space back over the mark.

    Empty while the disk has room — nothing is removed to tidy up, only to relieve pressure. Below the mark, the safe files are taken largest-first and only until free space would cross `low_mark + margin`, so the most space is reclaimed with the fewest deletions and a turn stops the moment it has enough rather than clearing the directory.

    Pure: the free figure, the file listing, and the liveness predicate are all arguments, which is what lets `status` compute the same plan `tend` carries out and lets the decision be tested without a real full disk.

    Args:
        free: Bytes free on the trace logs' filesystem now.
        low_mark: The floor below which reclaiming happens.
        files: The trace logs on disk (`read_trace_files`).
        alive: Whether a pid is a live process.
        margin: Bytes to reclaim past the mark, for hysteresis.

    Returns:
        The files to remove, largest-first, or empty where free space is already at or above the mark.
    """
    if free >= low_mark:
        return []
    target = low_mark + margin
    candidates = sorted(
        (file for file in files if safe_to_remove(file, alive=alive)),
        key=lambda file: file.size,
        reverse=True,
    )
    removals: list[Removal] = []
    reclaimed = 0
    for file in candidates:
        if free + reclaimed >= target:
            break
        removals.append(
            Removal(
                path=file.path,
                size=file.size,
                pid=file.pid,
                reason="compressed" if file.compressed else "dead_pid",
            )
        )
        reclaimed += file.size
    return removals


@dataclass(frozen=True)
class DiskReport:
    """The trace logs' filesystem this turn, and what Steward would reclaim from it.

    One object the terminal, `status.md`, the item, and the journal all read, so that no two surfaces disagree about whether the disk is short — the discipline `_tend.memory.MemoryReport` keeps for the same reason.
    """

    host: HostDisk
    """This turn's reading."""

    low_mark: int
    """The floor free space is judged against, in bytes."""

    removable: tuple[Removal, ...]
    """What a tend would remove, computed on the shared path so a `status` previews exactly it. Empty while the disk has room."""

    since: str | None = None
    """When a tend last recorded the disk as not short, or `None` where none has.

    What makes a shortage an *episode*: the item's id carries it, so an acknowledgment covers this shortage and not the next one after a recovery. `None` on a report assembled by hand, which claims nothing about earlier turns.
    """

    @property
    def low(self) -> bool:
        """Whether free space is under the mark."""
        return self.host.free < self.low_mark

    @property
    def tier(self) -> Literal["low"] | None:
        """The one condition this reports, or `None` where the disk has room."""
        return "low" if self.low else None

    @property
    def reclaimable(self) -> int:
        """Bytes the planned removals would free."""
        return sum(removal.size for removal in self.removable)

    @property
    def figures(self) -> str:
        """The reading as one line, which every rendering puts its own label in front of."""
        host = self.host
        side = "below" if self.low else "above"
        return (
            f"free {format_bytes(host.free)} of {format_bytes(host.total)} "
            f"({side} the {format_bytes(self.low_mark)} mark)"
        )

    @property
    def lines(self) -> list[str]:
        """The figures, then what can be reclaimed where anything is — one source for both renderings."""
        out = [self.figures]
        if self.removable:
            count = len(self.removable)
            out.append(
                f"{count} dead-process trace log{'s' if count != 1 else ''} "
                f"reclaimable ({format_bytes(self.reclaimable)})"
            )
        return out


def disk_report(
    host: HostDisk,
    files: Sequence[TraceFile],
    *,
    alive: Callable[[int], bool],
    low_mark: int,
    margin: int = DEFAULT_RECLAIM_MARGIN,
    since: str | None = None,
) -> DiskReport:
    """This turn's report: the reading, with the reclaim plan it implies.

    Args:
        host: The filesystem reading.
        files: The trace logs on disk (`read_trace_files`).
        alive: Whether a pid is a live process.
        low_mark: The floor free space is judged against.
        margin: Bytes to reclaim past the mark, for hysteresis.
        since: When a tend last recorded the disk as not short (`read_disk_since`).

    Returns:
        The report. The plan is `plan_cleanup`'s, so what the item previews is what a tend removes.
    """
    removable = plan_cleanup(
        free=host.free,
        low_mark=low_mark,
        files=files,
        alive=alive,
        margin=margin,
    )
    return DiskReport(
        host=host, low_mark=low_mark, removable=tuple(removable), since=since
    )


def disk_payload(report: DiskReport | None) -> dict[str, Any] | None:
    """The `disk` an observation records, or `None` while reclaiming is switched off.

    The tier rides beside the figures so that `read_disk_since` can tell a turn that was not short from one whose item was merely acknowledged — the shape `_tend.memory.memory_payload` keeps.

    Args:
        report: This turn's report.

    Returns:
        The payload for the observation's `disk` key.
    """
    if report is None:
        return None
    return {
        "total": report.host.total,
        "free": report.host.free,
        "mark": report.low_mark,
        "tier": report.tier,
    }


def read_disk_since(events: Sequence[JournalEvent]) -> str | None:
    """When a tend last recorded the disk as not short, as the journal's timestamp.

    The boundary of the current shortage, read off the recorded **tier** rather than off the item list — an acknowledged item leaves the list while the disk stays short, and reading that as a recovery would mint a new episode the turn after somebody accepted the old one. A turn that recorded no reading (reclaiming switched off) is not a shortage either. The structural twin of `_tend.memory.read_memory_since`.

    Args:
        events: Events in file order, as `read_journal` returns them.

    Returns:
        The newest such observation's timestamp, or `None` where every recorded observation was short, or none exist.
    """
    for event in reversed(events):
        if event.type != OBSERVATION:
            continue
        recorded = event.payload.get("disk")
        if not isinstance(recorded, dict):
            return event.ts
        if cast(dict[str, Any], recorded).get("tier") is None:
            return event.ts
    return None


def remove_trace_file(path: Path) -> bool:
    """Remove one trace log, returning whether it was there to remove.

    Swallows `OSError` and returns `False` — the `_workspace.log` discipline — so a file that vanished, or one the process cannot unlink, costs the turn nothing and is not journalled as reclaimed. Only a real unlink returns `True`.

    Args:
        path: The file to remove.

    Returns:
        Whether a file was actually unlinked.
    """
    try:
        path.unlink()
        return True
    except OSError:
        return False


__all__ = [
    "DEFAULT_RECLAIM_MARGIN",
    "DiskReport",
    "Removal",
    "TraceFile",
    "disk_path",
    "disk_payload",
    "disk_report",
    "parse_trace_pid",
    "plan_cleanup",
    "read_disk_since",
    "read_trace_files",
    "remove_trace_file",
    "safe_to_remove",
    "trace_dirs",
]
