import pytest
from enhancer.vram import Tile, plan_tiles


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


from enhancer.vram import TileFloorReached, TileRunner


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


def test_halves_tile_on_oom_and_succeeds():
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32)
    img = torch.rand(1, 3, 64, 64)
    out = runner.run(_oom_for_first(1), img)
    assert torch.equal(out, img)
    assert runner.tile == 128


def test_halves_repeatedly_until_success():
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
    runner = TileRunner(tile=256, overlap=0, scale=1, min_tile=32, recover_after=3)
    img = torch.rand(1, 3, 64, 64)
    runner.run(_oom_for_first(1), img)
    assert runner.tile == 128
    for _ in range(3):
        runner.run(lambda t: t, img)
    assert runner.tile == 256, "tile should step back up after sustained success"


def test_recovery_never_exceeds_starting_tile():
    runner = TileRunner(tile=128, overlap=0, scale=1, min_tile=32, recover_after=1)
    img = torch.rand(1, 3, 64, 64)
    for _ in range(10):
        runner.run(lambda t: t, img)
    assert runner.tile == 128
