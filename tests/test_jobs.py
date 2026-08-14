import json

import pytest

from enhancer.jobs import JobState, SettingsMismatch, settings_hash


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
    reloaded = JobState.load(tmp_path / "job", settings={"scale": 2})
    assert reloaded.completed_segments == [0]
    assert reloaded.total_frames == 1000
    assert reloaded.segment_frames == 500


def test_resume_refuses_when_settings_changed(tmp_path):
    JobState.create(tmp_path / "job", source="in.mp4", settings={"scale": 2}, total_frames=1000, segment_frames=500)
    with pytest.raises(SettingsMismatch):
        JobState.load(tmp_path / "job", settings={"scale": 4})


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
