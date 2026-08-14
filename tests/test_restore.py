import pytest

from enhancer.analyze import ScanType
from enhancer.restore import RestoreSettings, build_filter_chain


def test_progressive_clean_source_needs_no_filters():
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert chain == ""


def test_telecined_source_uses_inverse_telecine_not_deinterlace():
    """Film-sourced 30i must be IVTC'd; deinterlacing would discard real frames."""
    chain = build_filter_chain(
        ScanType.TELECINED, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert "fieldmatch" in chain
    assert "decimate" in chain
    assert "bwdif" not in chain


def test_interlaced_source_uses_deinterlacer_not_ivtc():
    chain = build_filter_chain(
        ScanType.INTERLACED, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert "bwdif" in chain
    assert "fieldmatch" not in chain


def test_bottom_field_first_is_passed_to_the_deinterlacer():
    chain = build_filter_chain(
        ScanType.INTERLACED, field_order="bff",
        settings=RestoreSettings(deblock=0.0, degrain=0.0),
    )
    assert "parity=1" in chain


def test_deblock_is_included_when_requested():
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.5, degrain=0.0),
    )
    assert "deblock" in chain


def test_degrain_is_included_when_requested():
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=0.5),
    )
    assert "hqdn3d" in chain


def test_degrain_biases_temporal_over_spatial():
    """Spatial denoising destroys skin texture; temporal mostly does not."""
    chain = build_filter_chain(
        ScanType.PROGRESSIVE, field_order="tff",
        settings=RestoreSettings(deblock=0.0, degrain=1.0),
    )
    params = chain.split("hqdn3d=")[1].split(",")[0]
    luma_spatial, _chroma_spatial, luma_temporal, _chroma_temporal = (
        float(v) for v in params.split(":")
    )
    assert luma_temporal > luma_spatial


def test_default_degrain_is_light():
    """Over-degraining is itself the polishing agent this project avoids."""
    assert RestoreSettings().degrain <= 0.35


def test_filters_are_ordered_scan_then_deblock_then_degrain():
    chain = build_filter_chain(
        ScanType.INTERLACED, field_order="tff",
        settings=RestoreSettings(deblock=0.5, degrain=0.5),
    )
    assert chain.index("bwdif") < chain.index("deblock") < chain.index("hqdn3d")


def test_strength_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        build_filter_chain(
            ScanType.PROGRESSIVE, field_order="tff",
            settings=RestoreSettings(deblock=2.0, degrain=0.0),
        )


import numpy as np
import torch

from enhancer.restore import apply_detail_retention, gaussian_blur


def test_gaussian_blur_preserves_shape():
    x = torch.rand(1, 3, 32, 48)
    assert gaussian_blur(x, sigma=1.5).shape == x.shape


def test_gaussian_blur_reduces_high_frequency_energy():
    x = torch.rand(1, 3, 64, 64)
    blurred = gaussian_blur(x, sigma=2.0)
    assert blurred.var() < x.var()


def test_gaussian_blur_leaves_a_constant_image_unchanged():
    x = torch.full((1, 3, 32, 32), 0.5)
    assert torch.allclose(gaussian_blur(x, sigma=2.0), x, atol=1e-5)


def test_alpha_zero_returns_model_output_unchanged():
    source = torch.rand(1, 3, 16, 16)
    output = torch.rand(1, 3, 32, 32)
    result = apply_detail_retention(output, source, alpha=0.0)
    assert torch.equal(result, output)


def test_detail_retention_adds_high_frequency_energy():
    """The residual must actually reach the output."""
    rng = torch.Generator().manual_seed(0)
    source = torch.rand(1, 3, 32, 32, generator=rng)
    output = gaussian_blur(
        torch.nn.functional.interpolate(source, scale_factor=2, mode="bicubic"),
        sigma=2.0,
    )
    enhanced = apply_detail_retention(output, source, alpha=0.5)
    assert enhanced.var() > output.var()


def test_detail_retention_output_stays_in_range():
    source = torch.rand(1, 3, 16, 16)
    output = torch.rand(1, 3, 32, 32)
    result = apply_detail_retention(output, source, alpha=1.0)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_detail_retention_is_deterministic():
    source = torch.rand(1, 3, 16, 16)
    output = torch.rand(1, 3, 32, 32)
    a = apply_detail_retention(output, source, alpha=0.4)
    b = apply_detail_retention(output, source, alpha=0.4)
    assert torch.equal(a, b)


def test_detail_retention_rejects_alpha_out_of_range():
    with pytest.raises(ValueError):
        apply_detail_retention(torch.rand(1, 3, 8, 8), torch.rand(1, 3, 4, 4), alpha=1.5)


def test_detail_retention_handles_non_integer_scale():
    """Source and output need not differ by a whole-number factor."""
    source = torch.rand(1, 3, 17, 23)
    output = torch.rand(1, 3, 40, 55)
    assert apply_detail_retention(output, source, alpha=0.3).shape == output.shape


from enhancer.restore import apply_regrain


def test_amount_zero_returns_the_input_unchanged():
    x = torch.rand(1, 3, 32, 32)
    assert torch.equal(apply_regrain(x, amount=0.0, seed=0), x)


def test_regrain_increases_local_variance():
    x = torch.full((1, 3, 64, 64), 0.5)
    grained = apply_regrain(x, amount=0.5, seed=0)
    assert grained.var() > x.var()


def test_regrain_is_deterministic_for_a_given_seed():
    x = torch.full((1, 3, 32, 32), 0.5)
    assert torch.equal(
        apply_regrain(x, amount=0.5, seed=42), apply_regrain(x, amount=0.5, seed=42)
    )


def test_different_seeds_produce_different_grain():
    x = torch.full((1, 3, 32, 32), 0.5)
    assert not torch.equal(
        apply_regrain(x, amount=0.5, seed=1), apply_regrain(x, amount=0.5, seed=2)
    )


def test_regrain_output_stays_in_range():
    x = torch.rand(1, 3, 32, 32)
    result = apply_regrain(x, amount=1.0, seed=0)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_stronger_amount_produces_more_grain():
    x = torch.full((1, 3, 64, 64), 0.5)
    light = apply_regrain(x, amount=0.2, seed=0)
    heavy = apply_regrain(x, amount=0.8, seed=0)
    assert heavy.var() > light.var()


def test_grain_is_suppressed_in_highlights_and_shadows():
    """Film grain is most visible in midtones; clipped areas show little."""
    mid = apply_regrain(torch.full((1, 3, 64, 64), 0.5), amount=0.8, seed=0)
    bright = apply_regrain(torch.full((1, 3, 64, 64), 0.99), amount=0.8, seed=0)
    assert mid.var() > bright.var()


def test_regrain_rejects_amount_out_of_range():
    with pytest.raises(ValueError):
        apply_regrain(torch.rand(1, 3, 8, 8), amount=-0.1, seed=0)
