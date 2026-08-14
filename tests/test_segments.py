import numpy as np
import pytest

from enhancer.jobs import JobState
from enhancer.segments import (
    assemble,
    completed_segment_paths,
    segment_path,
    write_segment,
)
from enhancer.video_io import Decoder, SourceProfile


def test_segment_path_is_zero_padded_and_sortable(tmp_path):
    assert segment_path(tmp_path, 7).name == "seg_00007.mkv"
    names = [segment_path(tmp_path, i).name for i in (2, 10, 1)]
    assert sorted(names) == ["seg_00001.mkv", "seg_00002.mkv", "seg_00010.mkv"]


def test_write_segment_produces_a_playable_file(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(profile).frames())[:10]
    out = segment_path(tmp_path, 0)
    write_segment(out, iter(frames), width=320, height=240, fps=25.0, source=profile)
    assert out.exists()
    assert len(list(Decoder(SourceProfile.probe(out)).frames())) == 10


def test_write_segment_leaves_no_part_file_on_success(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    frames = list(Decoder(profile).frames())[:5]
    out = segment_path(tmp_path, 0)
    write_segment(out, iter(frames), width=320, height=240, fps=25.0, source=profile)
    assert not out.with_suffix(".part.mkv").exists()
    assert list(tmp_path.glob("*.part*")) == []


def test_write_segment_discards_partial_output_on_failure(tmp_path, synthetic_clip):
    """An interrupted segment must not leave a file that looks complete."""
    profile = SourceProfile.probe(synthetic_clip)

    def exploding_frames():
        yield from list(Decoder(profile).frames())[:3]
        raise RuntimeError("simulated interruption")

    out = segment_path(tmp_path, 0)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_segment(out, exploding_frames(), width=320, height=240, fps=25.0, source=profile)
    assert not out.exists(), "a failed segment must not be left behind as complete"


def test_completed_segment_paths_are_in_index_order(tmp_path):
    for i in (2, 0, 1):
        segment_path(tmp_path, i).write_bytes(b"x")
    found = completed_segment_paths(tmp_path, count=3)
    assert [p.name for p in found] == ["seg_00000.mkv", "seg_00001.mkv", "seg_00002.mkv"]


def test_completed_segment_paths_raises_on_a_gap(tmp_path):
    segment_path(tmp_path, 0).write_bytes(b"x")
    segment_path(tmp_path, 2).write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        completed_segment_paths(tmp_path, count=3)


def test_assemble_concatenates_segments_and_preserves_frame_count(tmp_path, synthetic_clip):
    profile = SourceProfile.probe(synthetic_clip)
    all_frames = list(Decoder(profile).frames())
    write_segment(segment_path(tmp_path, 0), iter(all_frames[:25]),
                  width=320, height=240, fps=25.0, source=profile)
    write_segment(segment_path(tmp_path, 1), iter(all_frames[25:50]),
                  width=320, height=240, fps=25.0, source=profile)

    final = tmp_path / "final.mkv"
    assemble(tmp_path, count=2, output=final, source=profile)
    assert final.exists()
    assert len(list(Decoder(SourceProfile.probe(final)).frames())) == 50
