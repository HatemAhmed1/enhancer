import numpy as np
import pytest
from PIL import Image

from enhancer.images import (
    IMAGE_SUFFIXES,
    is_image,
    load_image,
    save_image,
    upscale_image,
)


class _Doubler:
    scale = 2
    device = "cpu"

    def process(self, frame):
        return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


@pytest.fixture
def rgb_png(tmp_path, rng):
    p = tmp_path / "in.png"
    Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(p)
    return p


@pytest.fixture
def rgba_png(tmp_path, rng):
    p = tmp_path / "in_alpha.png"
    Image.fromarray(rng.integers(0, 256, (40, 60, 4), dtype=np.uint8), "RGBA").save(p)
    return p


def test_recognises_still_formats():
    assert is_image("a.png") and is_image("a.JPG") and is_image("a.webp")


def test_does_not_claim_video_formats():
    assert not is_image("a.mp4") and not is_image("a.mkv")


def test_every_listed_suffix_is_lowercase_with_a_dot():
    assert all(s.startswith(".") and s.islower() for s in IMAGE_SUFFIXES)


def test_load_returns_rgb_without_alpha(rgb_png):
    rgb, alpha = load_image(rgb_png)
    assert rgb.shape == (40, 60, 3)
    assert rgb.dtype == np.uint8
    assert alpha is None


def test_load_separates_alpha(rgba_png):
    rgb, alpha = load_image(rgba_png)
    assert rgb.shape == (40, 60, 3)
    assert alpha is not None and alpha.shape == (40, 60)


def test_save_round_trips_pixels(tmp_path, rng):
    data = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    out = save_image(tmp_path / "o.png", data)
    assert np.array_equal(np.asarray(Image.open(out).convert("RGB")), data)


def test_save_preserves_alpha_for_png(tmp_path, rng):
    rgb = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    out = save_image(tmp_path / "o.png", rgb, alpha)
    assert np.array_equal(np.asarray(Image.open(out))[:, :, 3], alpha)


def test_jpeg_drops_alpha_rather_than_failing(tmp_path, rng):
    """JPEG cannot store alpha; the write must still succeed."""
    rgb = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    alpha = np.full((8, 8), 128, dtype=np.uint8)
    out = save_image(tmp_path / "o.jpg", rgb, alpha)
    assert Image.open(out).mode == "RGB"


def test_upscale_doubles_the_dimensions(rgb_png, tmp_path):
    out = upscale_image(rgb_png, tmp_path / "out.png", _Doubler())
    assert Image.open(out).size == (120, 80)


def test_upscale_writes_the_requested_format(rgb_png, tmp_path):
    out = upscale_image(rgb_png, tmp_path / "out.jpg", _Doubler())
    assert out.suffix == ".jpg"
    assert Image.open(out).format == "JPEG"


def test_upscale_keeps_alpha_and_scales_it(rgba_png, tmp_path):
    out = upscale_image(rgba_png, tmp_path / "out.png", _Doubler())
    result = np.asarray(Image.open(out))
    assert result.shape == (80, 120, 4)


def test_upscale_creates_missing_output_directories(rgb_png, tmp_path):
    out = upscale_image(rgb_png, tmp_path / "deep" / "nested" / "out.png", _Doubler())
    assert out.exists()


def test_upscale_applies_texture_work_when_enabled(rgb_png, tmp_path):
    calls = []

    class _Texture:
        enabled = True

        def apply(self, output, source, index=0):
            calls.append(index)
            return output

    upscale_image(rgb_png, tmp_path / "o.png", _Doubler(), texture=_Texture())
    assert calls == [0]


def test_upscale_skips_texture_when_disabled(rgb_png, tmp_path):
    class _Texture:
        enabled = False

        def apply(self, output, source, index=0):
            raise AssertionError("must not be called when disabled")

    upscale_image(rgb_png, tmp_path / "o.png", _Doubler(), texture=_Texture())


def test_greyscale_source_is_promoted_to_rgb(tmp_path):
    p = tmp_path / "grey.png"
    Image.fromarray(np.full((8, 8), 120, dtype=np.uint8), "L").save(p)
    rgb, alpha = load_image(p)
    assert rgb.shape == (8, 8, 3)
    assert alpha is None
