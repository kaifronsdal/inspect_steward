"""Trace logs filling the disk: the safe-to-remove decision, the listing, and the episode.

The decision is pure and driven from synthetic files and a synthetic free-space
figure, because the one input a test cannot manufacture is a real full disk. What
is checked is the shape of the answer: nothing is removed while there is room, and
below the mark the safe files go largest-first only until there is enough — and a
file whose process is still alive, or whose name is not a trace log's, is never a
candidate however hard the disk is squeezed.
"""

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
from inspect_steward._tend.disk import (
    DEFAULT_RECLAIM_MARGIN,
    DiskReport,
    TraceFile,
    disk_payload,
    disk_report,
    parse_trace_pid,
    plan_cleanup,
    read_disk_since,
    read_trace_files,
    remove_trace_file,
    safe_to_remove,
)
from inspect_steward._worker import HostDisk
from inspect_steward._workspace import OBSERVATION, read_journal

GIB = 1024**3
TOTAL = 500 * GIB


def trace(pid: int, size: int, *, compressed: bool = False) -> TraceFile:
    name = f"trace-{pid}.log" + (".gz" if compressed else "")
    return TraceFile(
        path=Path("/traces") / name, size=size, pid=pid, compressed=compressed
    )


def host(free: int) -> HostDisk:
    return HostDisk(total=TOTAL, free=free, path="/traces")


def living(*pids: int) -> Callable[[int], bool]:
    """A liveness predicate: these pids are alive, the rest are gone."""
    alive = set(pids)

    def predicate(pid: int) -> bool:
        return pid in alive

    return predicate


# --- what is safe to remove -------------------------------------------------


@pytest.mark.parametrize(
    ("file", "alive", "safe"),
    [
        pytest.param(
            trace(1, GIB, compressed=True),
            living(1),
            True,
            id="a gz is safe even with a live pid",
        ),
        pytest.param(trace(2, GIB), living(), True, id="a dead pid's log is safe"),
        pytest.param(trace(3, GIB), living(3), False, id="a live pid's log is not"),
        pytest.param(
            trace(os.getpid(), GIB), living(), False, id="our own log is never safe"
        ),
    ],
)
def test_what_a_trace_log_being_safe_to_remove_means(
    file: TraceFile, alive: Callable[[int], bool], safe: bool
) -> None:
    assert safe_to_remove(file, alive=alive) is safe


@pytest.mark.parametrize(
    ("name", "pid", "compressed"),
    [
        pytest.param("trace-123.log", 123, False, id="a plain trace log"),
        pytest.param("trace-123.log.gz", 123, True, id="a compressed one"),
        pytest.param(
            "trace-abc.log", None, False, id="a non-numeric pid is not a trace log"
        ),
        pytest.param("foo.log", None, False, id="an unrelated name"),
        pytest.param("trace-123.txt", None, False, id="a trace with the wrong suffix"),
    ],
)
def test_the_pid_a_trace_logs_name_carries(
    name: str, pid: int | None, compressed: bool
) -> None:
    assert parse_trace_pid(name) == (pid, compressed)


# --- the decision -----------------------------------------------------------


def test_a_disk_with_room_removes_nothing() -> None:
    files = [trace(1, 10 * GIB), trace(2, 10 * GIB, compressed=True)]

    assert (
        plan_cleanup(free=50 * GIB, low_mark=2 * GIB, files=files, alive=living()) == []
    )


def test_below_the_mark_the_largest_safe_files_go_first() -> None:
    # free is 1 GiB under a 2 GiB mark; largest-first reclaims the 3 GiB log and
    # stops, rather than taking the two small ones or clearing the directory
    files = [trace(1, GIB), trace(2, 3 * GIB), trace(3, GIB)]

    plan = plan_cleanup(
        free=1 * GIB, low_mark=2 * GIB, files=files, alive=living(), margin=0
    )

    assert [removal.size for removal in plan] == [3 * GIB]


def test_it_stops_once_there_is_enough_rather_than_clearing_the_directory() -> None:
    # five removable gibibytes, but a 1 GiB shortfall with no margin needs one
    files = [trace(index, GIB) for index in range(1, 6)]

    plan = plan_cleanup(
        free=1 * GIB, low_mark=2 * GIB, files=files, alive=living(), margin=0
    )

    assert len(plan) == 1


def test_the_margin_is_reclaimed_past_the_mark() -> None:
    # the same shortfall, now with a gibibyte of margin, takes a second file
    files = [trace(index, GIB) for index in range(1, 6)]

    with_margin = plan_cleanup(
        free=1 * GIB, low_mark=2 * GIB, files=files, alive=living(), margin=GIB
    )
    without = plan_cleanup(
        free=1 * GIB, low_mark=2 * GIB, files=files, alive=living(), margin=0
    )

    # target is mark (2) + margin (1) = 3, so free (1) + reclaimed must reach 3:
    # two gibibytes, where a zero margin stopped at one
    assert sum(removal.size for removal in with_margin) == 2 * GIB
    assert sum(removal.size for removal in without) == 1 * GIB


def test_a_live_pids_log_is_never_removed_however_hard_the_disk_is_squeezed() -> None:
    # the whole safety story in one case: a near-full disk, a huge live-pid log
    # that would fix it in one deletion, and it is still left alone
    files = [
        trace(999, 100 * GIB),  # live, and exactly what would help
        trace(2, GIB),  # dead, small
        trace(3, GIB, compressed=True),  # a gz
    ]

    plan = plan_cleanup(free=0, low_mark=200 * GIB, files=files, alive=living(999))

    pids = {removal.pid for removal in plan}
    assert 999 not in pids
    assert pids == {2, 3}


def test_a_compressed_log_is_taken_even_when_its_pid_is_live() -> None:
    files = [trace(1, GIB, compressed=True)]

    plan = plan_cleanup(free=0, low_mark=2 * GIB, files=files, alive=living(1))

    assert [removal.reason for removal in plan] == ["compressed"]


def test_all_live_leaves_nothing_to_remove_but_the_disk_is_still_short() -> None:
    files = [trace(1, 10 * GIB), trace(2, 10 * GIB)]

    report = disk_report(host(1 * GIB), files, alive=living(1, 2), low_mark=2 * GIB)

    assert report.removable == ()
    assert (
        report.tier == "low"
    )  # the item still fires; there is just nothing to reclaim


# --- the listing on disk ----------------------------------------------------


def test_the_listing_reads_sizes_and_pids_and_skips_everything_else(
    tmp_path: Path,
) -> None:
    (tmp_path / "trace-100.log").write_bytes(b"x" * 10)
    (tmp_path / "trace-200.log.gz").write_bytes(b"y" * 20)
    (tmp_path / "notes.txt").write_bytes(b"z" * 30)  # not a trace log
    (tmp_path / "trace-abc.log").write_bytes(b"w" * 40)  # not a numeric pid

    files = {file.pid: file for file in read_trace_files([tmp_path])}

    assert set(files) == {100, 200}
    assert files[100].size == 10 and files[100].compressed is False
    assert files[200].size == 20 and files[200].compressed is True


def test_a_directory_that_does_not_exist_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "trace-1.log").write_bytes(b"x")

    files = read_trace_files([tmp_path / "gone", tmp_path])

    assert [file.pid for file in files] == [1]


def test_removing_a_file_that_is_gone_is_not_reported_as_reclaimed(
    tmp_path: Path,
) -> None:
    present = tmp_path / "trace-1.log"
    present.write_bytes(b"x")

    assert remove_trace_file(present) is True
    assert not present.exists()
    assert remove_trace_file(tmp_path / "trace-2.log") is False


# --- the report an operator reads -------------------------------------------


def test_the_figures_name_free_total_and_the_mark() -> None:
    low = disk_report(host(1 * GIB), [], alive=living(), low_mark=2 * GIB)
    assert "below the 2.0 GiB mark" in low.figures
    assert low.low is True and low.tier == "low"

    fine = disk_report(host(400 * GIB), [], alive=living(), low_mark=2 * GIB)
    assert "above the 2.0 GiB mark" in fine.figures
    assert fine.low is False and fine.tier is None


def test_the_reclaimable_line_appears_only_when_something_can_go() -> None:
    files = [trace(1, 3 * GIB), trace(2, GIB)]
    report = disk_report(host(0), files, alive=living(), low_mark=2 * GIB)

    assert report.reclaimable == report.removable[0].size
    assert any("reclaimable" in line for line in report.lines)

    healthy = disk_report(host(400 * GIB), files, alive=living(), low_mark=2 * GIB)
    assert healthy.lines == [healthy.figures]


# --- the episode the journal holds ------------------------------------------


NOW = 1_800_000_000.0
"""2027-01-15T08:00:00Z, the clock the timestamps below are stamped from."""


def at(seconds_ago: float) -> str:
    return (
        datetime.fromtimestamp(NOW - seconds_ago, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def observe(journal: Path, ts: str, disk: dict[str, object] | None) -> None:
    with journal.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"ts": ts, "type": OBSERVATION, "disk": disk}) + "\n")


def reading(free: int, tier: str | None) -> dict[str, object]:
    return {"total": TOTAL, "free": free, "mark": 2 * GIB, "tier": tier}


def test_the_episode_boundary_is_the_last_turn_that_was_not_short(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal.jsonl"
    observe(journal, at(2400), reading(400 * GIB, None))
    # an acknowledged shortage leaves the item list and not the tier, so these
    # turns are still short and do not move the boundary
    observe(journal, at(1800), reading(1 * GIB, "low"))
    observe(journal, at(600), reading(1 * GIB, "low"))

    assert read_disk_since(read_journal(journal).events) == at(2400)


def test_a_turn_with_reclaiming_off_is_not_short(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    observe(journal, at(1200), reading(1 * GIB, "low"))
    observe(journal, at(600), None)

    assert read_disk_since(read_journal(journal).events) == at(600)


def test_a_workspace_short_since_its_first_tend_has_no_boundary(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    observe(journal, at(600), reading(1 * GIB, "low"))

    assert read_disk_since(read_journal(journal).events) is None
    assert read_disk_since([]) is None


def test_the_payload_carries_the_tier_and_reads_back_round_trip() -> None:
    low = disk_payload(disk_report(host(1 * GIB), [], alive=living(), low_mark=2 * GIB))
    assert low is not None and low["tier"] == "low" and low["free"] == 1 * GIB

    fine = disk_payload(
        disk_report(host(400 * GIB), [], alive=living(), low_mark=2 * GIB)
    )
    assert fine is not None and fine["tier"] is None
    assert disk_payload(None) is None


def test_the_default_margin_is_a_sensible_headroom() -> None:
    # not a behaviour, a guard: the margin is positive, so a turn that frees
    # space leaves the disk over the line rather than one byte above it
    assert DEFAULT_RECLAIM_MARGIN > 0


def test_a_report_assembled_by_hand_claims_nothing_about_earlier_turns() -> None:
    report = DiskReport(host=host(1 * GIB), low_mark=2 * GIB, removable=())
    assert report.since is None
