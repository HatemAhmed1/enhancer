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
