import json
import os

import pytest

from enhancer.jobs import (
    JobState,
    SettingsMismatch,
    SourceMismatch,
    settings_hash,
    source_identity,
)


@pytest.fixture
def a_source(tmp_path):
    """A real file on disk, so size and modification time exist."""
    path = tmp_path / "in.mp4"
    path.write_bytes(b"first film" * 100)
    return path


def test_settings_hash_is_stable_for_same_input():
    a = settings_hash({"model": "span", "scale": 2, "tile": 512})
    b = settings_hash({"tile": 512, "scale": 2, "model": "span"})
    assert a == b, "key order must not affect the hash"


def test_settings_hash_changes_when_a_value_changes():
    a = settings_hash({"model": "span", "scale": 2})
    b = settings_hash({"model": "span", "scale": 4})
    assert a != b


def test_new_job_has_no_completed_segments(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    assert job.completed_segments == []
    assert job.next_segment_index == 0


def test_marking_a_segment_complete_advances_the_ledger(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    job.mark_complete(0)
    assert job.completed_segments == [0]
    assert job.next_segment_index == 1


def test_journal_survives_a_round_trip(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    job.mark_complete(0)
    reloaded = JobState.load(tmp_path / "job", settings={"scale": 2}, source="in.mp4")
    assert reloaded.completed_segments == [0]
    assert reloaded.total_frames == 1000
    assert reloaded.segment_frames == 500


def test_resume_refuses_when_settings_changed(tmp_path):
    JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    with pytest.raises(SettingsMismatch):
        JobState.load(tmp_path / "job", settings={"scale": 4}, source="in.mp4")


def test_segment_count_rounds_up(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1001, segment_frames=500)
    assert job.segment_count == 3


def test_start_frame_for_segment(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1000, segment_frames=500)
    assert job.start_frame_for(0) == 0
    assert job.start_frame_for(1) == 500


def test_frames_in_final_segment_is_the_remainder(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1100, segment_frames=500)
    assert job.frames_in_segment(0) == 500
    assert job.frames_in_segment(2) == 100


def test_is_complete_only_when_every_segment_is_done(tmp_path):
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1000, segment_frames=500)
    job.mark_complete(0)
    assert not job.is_complete
    job.mark_complete(1)
    assert job.is_complete


def test_out_of_order_completion_is_recorded(tmp_path):
    """Segments may finish out of order; next_segment_index is the first gap."""
    job = JobState.create(tmp_path / "job", source="in.mp4", settings={}, total_frames=1500, segment_frames=500)
    job.mark_complete(1)
    assert job.next_segment_index == 0


# --- source identity --------------------------------------------------------


def test_resume_refuses_a_job_belonging_to_a_different_source(tmp_path, a_source):
    """The job directory defaults to <output>.job, so two renders to one output
    name share a journal. Matching on settings alone handed a *finished* job
    straight back for an unrelated film: nothing was rendered, the earlier
    film's segments were re-concatenated, and the run reported success."""
    other = tmp_path / "other.mp4"
    other.write_bytes(b"second film" * 100)

    JobState.create(tmp_path / "job", source=a_source, settings={"scale": 2},
                    total_frames=1000, segment_frames=500)
    with pytest.raises(SourceMismatch, match="different source"):
        JobState.load(tmp_path / "job", settings={"scale": 2}, source=other)


def test_a_source_mismatch_is_refused_like_a_settings_mismatch(tmp_path, a_source):
    """Callers catch SettingsMismatch; both refusals must arrive there."""
    other = tmp_path / "other.mp4"
    other.write_bytes(b"second film" * 100)
    JobState.create(tmp_path / "job", source=a_source, settings={},
                    total_frames=10, segment_frames=5)
    with pytest.raises(SettingsMismatch):
        JobState.load(tmp_path / "job", settings={}, source=other)


def test_resume_refuses_when_the_source_was_replaced_in_place(tmp_path, a_source):
    """Render a film, replace the file with a better rip, render again to the
    same name. The path is unchanged, so a path comparison alone lets the stale
    job through and delivers the old rip."""
    JobState.create(tmp_path / "job", source=a_source, settings={},
                    total_frames=10, segment_frames=5)

    a_source.write_bytes(b"a better rip, longer" * 100)
    os.utime(a_source, (0, 0))   # a different file by every cheap measure

    with pytest.raises(SourceMismatch, match="changed since"):
        JobState.load(tmp_path / "job", settings={}, source=a_source)


def test_resume_accepts_the_same_untouched_source(tmp_path, a_source):
    """The guard must not cost anyone a legitimate resume."""
    JobState.create(tmp_path / "job", source=a_source, settings={},
                    total_frames=10, segment_frames=5)
    reloaded = JobState.load(tmp_path / "job", settings={}, source=a_source)
    assert reloaded.total_frames == 10


def test_source_identity_ignores_case_on_a_folding_filesystem(tmp_path, a_source):
    """Otherwise IN.MP4 and in.mp4 would look like two different films."""
    shouty = a_source.with_name(a_source.name.upper())
    if os.path.normcase("A") != os.path.normcase("a"):
        pytest.skip("case-sensitive filesystem")
    assert source_identity(shouty) == source_identity(a_source)


def test_source_identity_uses_more_than_the_path(tmp_path, a_source):
    before = source_identity(a_source)
    a_source.write_bytes(b"different content entirely")
    os.utime(a_source, (0, 0))
    assert source_identity(a_source) != before


def test_source_identity_survives_a_missing_file(tmp_path):
    """An unreadable source is the render's problem, not the journal's."""
    assert "path" in source_identity(tmp_path / "gone.mp4")


def test_an_old_journal_without_a_fingerprint_still_checks_the_path(tmp_path, a_source):
    """Journals written before source identity was recorded must not become a
    way back into the bug."""
    JobState.create(tmp_path / "job", source=a_source, settings={},
                    total_frames=10, segment_frames=5)
    journal = tmp_path / "job" / "job.json"
    data = json.loads(journal.read_text())
    del data["source_fingerprint"]
    journal.write_text(json.dumps(data))

    other = tmp_path / "other.mp4"
    other.write_bytes(b"second film")
    with pytest.raises(SourceMismatch):
        JobState.load(tmp_path / "job", settings={}, source=other)
    # ...and still resumes for the source it names.
    assert JobState.load(tmp_path / "job", settings={}, source=a_source).total_frames == 10
