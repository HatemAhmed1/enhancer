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
