import pytest

from enhancer.timing import OutputFrame, output_frame_count, plan_output_frames


def test_same_fps_is_a_pure_passthrough():
    plan = plan_output_frames(src_fps=24, dst_fps=24, src_count=10)
    assert len(plan) == 10
    assert all(f.t == 0.0 for f in plan)
    assert [f.left for f in plan] == list(range(10))


def test_doubling_produces_alternating_real_and_synthetic_frames():
    plan = plan_output_frames(src_fps=24, dst_fps=48, src_count=4)
    assert [f.t for f in plan[:4]] == [0.0, 0.5, 0.0, 0.5]


def test_doubling_brackets_are_correct():
    plan = plan_output_frames(src_fps=24, dst_fps=48, src_count=4)
    assert (plan[1].left, plan[1].right) == (0, 1)
    assert (plan[2].left, plan[2].right) == (1, 1)


def test_non_integer_ratio_24_to_60():
    """2.5x cannot be expressed as 'insert N frames'."""
    plan = plan_output_frames(src_fps=24, dst_fps=60, src_count=4)
    assert len(plan) == 10
    expected = [0.0, 0.4, 0.8, 0.2, 0.6, 0.0, 0.4, 0.8, 0.2, 0.6]
    assert [round(f.t, 6) for f in plan] == expected


def test_ntsc_rates_do_not_drift():
    plan = plan_output_frames(src_fps=30000 / 1001, dst_fps=60000 / 1001, src_count=100)
    assert len(plan) == 200
    assert all(round(f.t, 6) in (0.0, 0.5) for f in plan)


def test_every_frame_references_valid_source_indices():
    plan = plan_output_frames(src_fps=25, dst_fps=60, src_count=50)
    for f in plan:
        assert 0 <= f.left < 50
        assert 0 <= f.right < 50
        assert f.right >= f.left


def test_t_is_always_within_unit_interval():
    plan = plan_output_frames(src_fps=23.976, dst_fps=59.94, src_count=30)
    assert all(0.0 <= f.t < 1.0 for f in plan)


def test_exact_source_frames_are_marked_as_copies():
    plan = plan_output_frames(src_fps=24, dst_fps=48, src_count=3)
    assert plan[0].is_copy
    assert not plan[1].is_copy


def test_last_source_frame_never_brackets_past_the_end():
    plan = plan_output_frames(src_fps=24, dst_fps=60, src_count=3)
    assert plan[-1].right <= 2


def test_output_frame_count_matches_the_plan_length():
    for dst in (24, 30, 48, 50, 60, 120):
        assert output_frame_count(24, dst, 100) == len(
            plan_output_frames(24, dst, 100)
        )


def test_slowing_down_is_rejected():
    with pytest.raises(ValueError, match="target frame rate"):
        plan_output_frames(src_fps=60, dst_fps=30, src_count=10)


def test_nonpositive_rates_are_rejected():
    with pytest.raises(ValueError):
        plan_output_frames(src_fps=0, dst_fps=60, src_count=10)


def test_empty_source_produces_an_empty_plan():
    assert plan_output_frames(src_fps=24, dst_fps=48, src_count=0) == []
