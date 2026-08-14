import pytest
from enhancer.vram import plan_tiles


def test_single_tile_when_image_fits():
    tiles = plan_tiles(h=100, w=100, tile=256, overlap=16)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.y0, t.y1, t.x0, t.x1) == (0, 100, 0, 100)
    assert (t.py0, t.py1, t.px0, t.px1) == (0, 100, 0, 100)


def test_cores_partition_image_exactly():
    tiles = plan_tiles(h=300, w=500, tile=128, overlap=16)
    covered = set()
    for t in tiles:
        for y in range(t.y0, t.y1):
            for x in range(t.x0, t.x1):
                assert (y, x) not in covered, "cores must not overlap"
                covered.add((y, x))
    assert len(covered) == 300 * 500, "cores must cover every pixel"


def test_padding_extends_core_and_clamps_at_edges():
    tiles = plan_tiles(h=300, w=300, tile=128, overlap=16)
    for t in tiles:
        assert t.py0 == max(0, t.y0 - 16)
        assert t.py1 == min(300, t.y1 + 16)
        assert t.px0 == max(0, t.x0 - 16)
        assert t.px1 == min(300, t.x1 + 16)


def test_zero_overlap_means_padded_equals_core():
    tiles = plan_tiles(h=200, w=200, tile=64, overlap=0)
    for t in tiles:
        assert (t.py0, t.py1, t.px0, t.px1) == (t.y0, t.y1, t.x0, t.x1)


def test_rejects_nonpositive_tile():
    with pytest.raises(ValueError):
        plan_tiles(h=10, w=10, tile=0, overlap=0)


import torch
from enhancer.vram import run_tiled


def test_identity_roundtrip_is_exact_at_zero_overlap():
    img = torch.rand(1, 3, 200, 300)
    out = run_tiled(lambda t: t, img, tile=64, overlap=0, scale=1)
    assert torch.equal(out, img)


def test_identity_roundtrip_is_exact_with_overlap():
    """Exact-crop reassembly must be lossless at any overlap."""
    img = torch.rand(1, 3, 200, 300)
    out = run_tiled(lambda t: t, img, tile=64, overlap=16, scale=1)
    assert torch.equal(out, img)


def test_scale_2x_produces_correct_shape():
    img = torch.rand(1, 3, 100, 150)
    out = run_tiled(
        lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest"),
        img, tile=32, overlap=8, scale=2,
    )
    assert out.shape == (1, 3, 200, 300)


def test_nearest_upscale_matches_untiled_result():
    img = torch.rand(1, 3, 64, 64)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")
    tiled = run_tiled(fn, img, tile=16, overlap=8, scale=2)
    assert torch.allclose(tiled, fn(img))


def test_single_tile_path_matches_direct_call():
    img = torch.rand(1, 3, 32, 32)
    out = run_tiled(lambda t: t * 2, img, tile=256, overlap=16, scale=1)
    assert torch.equal(out, img * 2)


def test_reassembly_is_exact_for_non_square_non_divisible_at_scale_4():
    """Closes the main test gap: non-square, non-divisible size, scale > 1.

    Bit-exact, not allclose — nearest-neighbor upscaling has no floating-point
    accumulation, so any crop-offset bug shows up as a hard mismatch.
    """
    img = torch.rand(1, 3, 213, 377)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
    tiled = run_tiled(fn, img, tile=64, overlap=16, scale=4)
    assert torch.equal(tiled, fn(img))


def test_scale_mismatch_raises_instead_of_corrupting():
    """A 4x model driven with scale=2 must raise, not silently write scrambled pixels."""
    img = torch.rand(1, 3, 64, 64)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
    with pytest.raises(ValueError):
        run_tiled(fn, img, tile=16, overlap=4, scale=2)


def test_single_tile_path_also_validates_scale():
    """The single-tile fast path must agree with the multi-tile path on scale checks."""
    img = torch.rand(1, 3, 32, 32)
    fn = lambda t: torch.nn.functional.interpolate(t, scale_factor=4, mode="nearest")
    with pytest.raises(ValueError):
        run_tiled(fn, img, tile=256, overlap=16, scale=2)


def test_scale_mismatch_on_edge_tile_raises():
    """A model that pads its input to a stride multiple returns a different
    effective scale on small edge tiles than on full-size interior tiles.
    _check_scale must run on every tile, not just the first, or this
    silently returns a wrong-shape, wrong-pixel result with no error.
    """
    def padding_model(t):
        h, w = t.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        t2 = torch.nn.functional.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return torch.nn.functional.interpolate(t2, scale_factor=2, mode="nearest")

    with pytest.raises(ValueError):
        run_tiled(padding_model, torch.rand(1, 3, 200, 200), tile=64, overlap=0, scale=2)


def test_zero_dimension_image_raises_value_error():
    img = torch.rand(1, 3, 0, 64)
    with pytest.raises(ValueError):
        run_tiled(lambda t: t, img, tile=32, overlap=0, scale=1)


def test_nonpositive_scale_raises_value_error():
    img = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError):
        run_tiled(lambda t: t, img, tile=32, overlap=0, scale=0)


from enhancer.vram import TileFloorReached, TileRunner, is_oom_error


def _oom_for_first(n_failures):
    """Return an infer fn that raises CUDA OOM for the first n calls."""
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        if calls["n"] <= n_failures:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return t

    fn.calls = calls
    return fn


def test_steps_down_ladder_on_oom_and_succeeds():
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    out = runner.run(_oom_for_first(1), img)
    assert torch.equal(out, img)
    assert runner.tile == 192, "the next rung down from 256 is 192, not 128"


def test_steps_down_repeatedly_until_success():
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(3), img)
    assert runner.tile == 32


def test_raises_tile_floor_reached_when_floor_still_ooms():
    runner = TileRunner(tile=64, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    with pytest.raises(TileFloorReached):
        runner.run(_oom_for_first(99), img)


def test_recovers_tile_size_after_sustained_success():
    """Recovery uses the probe interval, which doubles after each OOM.

    With recover_after=3, one OOM doubles the interval to 6, so only 3 more
    successes (4 total, one short of 6) is not enough to trigger a step-up;
    it takes 5 more (bringing the running total to 6).
    """
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32, recover_after=3)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(1), img)
    assert runner.tile == 192
    for _ in range(5):
        runner.run(lambda t: t, img)
    assert runner.tile == 256, "tile should step back up after sustained success"


def test_recovery_never_exceeds_starting_tile():
    runner = TileRunner(tile=128, overlap=0, scale=1, min_tile=32, recover_after=1)
    img = torch.rand(1, 3, 64, 64)
    for _ in range(10):
        runner.run(lambda t: t, img)
    assert runner.tile == 128


def test_step_up_reaches_max_tile_above_ladder_top():
    """max_tile above the top ladder rung must not strand recovery below it.

    _step_up only ever returns a ladder rung; when max_tile (2048) is above
    the top rung (1024), stepping up from 1024 must still land on max_tile,
    not get stuck returning 1024 unchanged forever.
    """
    runner = TileRunner(tile=2048, overlap=0, scale=1, min_tile=32, recover_after=1)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(1), img)
    assert runner.tile == 1024, "the top ladder rung below 2048"

    for _ in range(20):
        runner.run(lambda t: t, img)

    assert runner.tile == 2048, "should climb all the way back to max_tile, not stick at 1024"


def test_probe_interval_doubles_once_per_frame_not_per_retry():
    """One frame's descent through several ladder rungs is one probe-interval
    doubling, not one per retry. Otherwise a single transient spike that
    forces several step-downs inflates the recovery interval by 2**n instead
    of 2**1, making the runner recover orders of magnitude slower than
    recover_after implies.
    """
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        if calls["n"] <= 4:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return t

    runner = TileRunner(tile=1024, overlap=0, scale=1, min_tile=32, recover_after=5)
    img = torch.rand(1, 3, 64, 64)
    runner.run(fn, img)

    assert calls["n"] == 5, "sanity: 4 OOM retries then a success, all within one run() call"
    assert runner._probe_interval == 10, "doubled exactly once, not once per retry (2**4=80)"


def test_is_oom_error_accepts_plain_runtime_error_variants():
    assert is_oom_error(RuntimeError("CUDA error: out of memory"))
    assert is_oom_error(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"))
    assert is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert is_oom_error(MemoryError("cannot allocate"))
    assert is_oom_error(torch.cuda.OutOfMemoryError("CUDA out of memory"))


def test_is_oom_error_rejects_unrelated_runtime_error():
    """Retrying an illegal memory access forever would mask a real bug."""
    assert not is_oom_error(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    )
    assert not is_oom_error(ValueError("nope"))


def test_runner_recovers_from_plain_runtime_error_oom():
    calls = {"n": 0}

    def fn(t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: out of memory")
        return t

    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    out = runner.run(fn, img)
    assert torch.equal(out, img)
    assert calls["n"] == 2


def test_runner_propagates_non_oom_errors():
    def fn(t):
        raise ValueError("boom")

    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    with pytest.raises(ValueError):
        runner.run(fn, img)


def test_probe_interval_backs_off_so_oom_does_not_repeat_forever():
    """OOM events must grow logarithmically with frame count, not linearly.

    A fixed recover_after would re-probe (and re-OOM against) a tile size that
    genuinely doesn't fit every `recover_after` frames, forever. The doubling
    probe interval converges instead.
    """
    threshold = 256
    recover_after = 5
    n_frames = 500
    runner = TileRunner(tile=1024, overlap=0, scale=1, min_tile=128, recover_after=recover_after)
    oom_count = {"n": 0}

    def fn(patch):
        if runner.tile > threshold:
            oom_count["n"] += 1
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return patch

    img = torch.rand(1, 3, 64, 64)  # smaller than any ladder rung: always single-tile
    for _ in range(n_frames):
        runner.run(fn, img)

    linear_bound = n_frames // recover_after
    assert oom_count["n"] < linear_bound, (
        f"{oom_count['n']} OOMs over {n_frames} frames is not far fewer than "
        f"the fixed-interval bound of {linear_bound}"
    )
    assert oom_count["n"] < 20, "expected logarithmic growth, not linear"


from enhancer.vram import HEADROOM_BYTES, TILE_LADDER, choose_tile, select_device


def test_headroom_default_is_one_gigabyte():
    assert HEADROOM_BYTES == 1024 ** 3


def test_ladder_is_descending():
    assert list(TILE_LADDER) == sorted(TILE_LADDER, reverse=True)


def test_picks_largest_tile_that_fits():
    # 3 GB usable after headroom, 64 bytes per output pixel.
    free = 4 * 1024 ** 3
    tile = choose_tile(free_bytes=free, bytes_per_output_pixel=64)
    budget = free - HEADROOM_BYTES
    assert tile * tile * 64 <= budget
    bigger = [t for t in TILE_LADDER if t > tile]
    for b in bigger:
        assert b * b * 64 > budget, "should have picked the largest fitting tile"


def test_falls_back_to_smallest_tile_when_nothing_fits():
    tile = choose_tile(free_bytes=HEADROOM_BYTES + 1, bytes_per_output_pixel=10 ** 9)
    assert tile == TILE_LADDER[-1]


def test_zero_free_memory_returns_smallest_tile():
    assert choose_tile(free_bytes=0, bytes_per_output_pixel=64) == TILE_LADDER[-1]


def test_choose_tile_accounts_for_overlap():
    """The padded (tile + 2*overlap) area is what's actually allocated."""
    bytes_per_output_pixel = 1
    budget = 100_000
    free = budget + HEADROOM_BYTES

    no_overlap = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, overlap=0)
    with_overlap = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, overlap=32)

    assert no_overlap == 256
    assert with_overlap < no_overlap

    padded = with_overlap + 2 * 32
    assert padded * padded * bytes_per_output_pixel <= budget


def test_choose_tile_accounts_for_scale():
    """A 4x model produces 16x the pixels, so it must pick a smaller tile."""
    free = HEADROOM_BYTES + 50 * 1024 * 1024
    bytes_per_output_pixel = 64

    t1 = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, scale=1)
    t4 = choose_tile(free_bytes=free, bytes_per_output_pixel=bytes_per_output_pixel, scale=4)

    assert t4 < t1


def test_select_device_returns_cpu_when_not_preferring_cuda():
    assert select_device(prefer_cuda=False) == torch.device("cpu")


from enhancer.vram import (
    bytes_per_output_pixel_for_dtype,
    physical_vram_ceiling,
)


def test_bytes_per_pixel_doubles_for_float32():
    fp16 = bytes_per_output_pixel_for_dtype(torch.float16)
    fp32 = bytes_per_output_pixel_for_dtype(torch.float32)
    assert fp32 == fp16 * 2


def test_bytes_per_pixel_is_positive():
    assert bytes_per_output_pixel_for_dtype(torch.float16) > 0


def test_physical_ceiling_subtracts_headroom():
    assert physical_vram_ceiling(total_bytes=6 * 1024 ** 3) == 6 * 1024 ** 3 - HEADROOM_BYTES


def test_physical_ceiling_never_negative():
    assert physical_vram_ceiling(total_bytes=0) == 0


def test_runner_shrinks_when_peak_exceeds_ceiling_without_exception():
    """Windows WDDM oversubscribes silently instead of raising.

    The runner must treat a peak allocation above the physical ceiling as a
    pressure signal, even though no exception was raised.
    """
    runner = TileRunner(tile=512, overlap=0, scale=1, min_tile=128, vram_ceiling=1000)
    img = torch.rand(1, 3, 64, 64)
    runner.run(lambda t: t, img, peak_bytes=5000)
    assert runner.tile < 512, "peak above ceiling must shrink the tile"


def test_runner_does_not_shrink_when_peak_within_ceiling():
    runner = TileRunner(tile=512, overlap=0, scale=1, min_tile=128, vram_ceiling=10_000)
    img = torch.rand(1, 3, 64, 64)
    runner.run(lambda t: t, img, peak_bytes=5000)
    assert runner.tile == 512


def test_runner_ignores_peak_when_no_ceiling_configured():
    runner = TileRunner(tile=512, overlap=0, scale=1, min_tile=128)
    img = torch.rand(1, 3, 64, 64)
    runner.run(lambda t: t, img, peak_bytes=10 ** 12)
    assert runner.tile == 512
