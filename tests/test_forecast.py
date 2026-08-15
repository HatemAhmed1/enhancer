"""The forecast exists so a bad setting costs a glance, not a night."""

import pytest

from enhancer.forecast import Forecast, forecast, throughput_for

HD = dict(width=1920, height=1080, fps=24.0, frames=172800)
SD = dict(width=854, height=480, fps=24.0, frames=172800)


# --- what comes out ---------------------------------------------------------


def test_output_size_is_the_source_times_the_scale():
    f = forecast(**HD, scale=2, model_name="2xParimgCompact")
    assert (f.width, f.height) == (3840, 2160)


def test_four_times_quadruples():
    f = forecast(**HD, scale=4, model_name="4xPurePhoto-span")
    assert (f.width, f.height) == (7680, 4320)


def test_familiar_size_names():
    assert forecast(**HD, scale=2, model_name="m").label == "4K"
    assert forecast(**HD, scale=4, model_name="m").label == "8K"


def test_an_unfamiliar_size_is_left_unnamed():
    """1708x960 is not 720p. A wrong familiar name misleads worse than none."""
    assert forecast(**SD, scale=2, model_name="m").label == ""


def test_frame_rate_is_unchanged_without_interpolation():
    assert forecast(**HD, scale=2, model_name="m").fps == 24.0


def test_interpolation_changes_rate_and_frame_count():
    f = forecast(**HD, scale=2, model_name="m", target_fps=60.0)
    assert f.fps == 60.0
    assert f.frames == pytest.approx(432000, rel=0.01)


# --- the steps that will run ------------------------------------------------


def test_steps_read_in_the_order_they_happen():
    f = forecast(**HD, scale=2, model_name="m", scan="interlaced",
                 deblock=0.3, degrain=0.25, detail_retention=0.25, regrain=0.6,
                 target_fps=60.0)
    def position(fragment):
        return next(i for i, step in enumerate(f.steps) if fragment in step)

    assert (position("Deinterlace")
            < position("compression artifacts")
            < position("Reduce noise")
            < position("Enlarge")
            < position("Restore original")
            < position("Add film grain")
            < position("Smooth motion"))


def test_telecine_is_described_as_recovering_film_frames():
    f = forecast(**SD, scale=2, model_name="m", scan="telecined")
    assert any("film frames" in s for s in f.steps)


def test_disabled_stages_are_not_listed():
    f = forecast(**HD, scale=2, model_name="m")
    assert not any("grain" in s.lower() for s in f.steps)
    assert any("Enlarge" in s for s in f.steps)


# --- time, against numbers actually measured on this machine ----------------


@pytest.mark.parametrize("kw,hours", [
    (dict(SD, scale=2, model_name="2xParimgCompact"), 2.5),
    (dict(HD, scale=2, model_name="2xParimgCompact"), 16.8),
    (dict(HD, scale=2, model_name="2xModernSpanimationV1"), 22.5),
    (dict(HD, scale=2, model_name="RealESRGAN_x2plus"), 155.0),
    (dict(SD, scale=4, model_name="4xPurePhoto-span"), 2.9),
])
def test_time_estimate_matches_measurement(kw, hours):
    """Within 20% of what was actually timed on an RTX 3060 Laptop."""
    predicted = forecast(**kw).seconds / 3600
    assert predicted == pytest.approx(hours, rel=0.20)


def test_restoration_costs_time():
    plain = forecast(**HD, scale=2, model_name="2xParimgCompact").seconds
    restored = forecast(**HD, scale=2, model_name="2xParimgCompact", degrain=0.25).seconds
    assert restored > plain


def test_interpolation_roughly_doubles_the_time():
    plain = forecast(**HD, scale=2, model_name="2xParimgCompact").seconds
    smooth = forecast(**HD, scale=2, model_name="2xParimgCompact", target_fps=48).seconds
    assert smooth == pytest.approx(plain * 2, rel=0.1)


def test_throughput_falls_off_with_resolution():
    """Measured: about a quarter lost going from 480p to 1080p."""
    sd = forecast(**SD, scale=2, model_name="2xParimgCompact").seconds / 172800
    hd = forecast(**HD, scale=2, model_name="2xParimgCompact").seconds / 172800
    sd_rate = (854 * 480 / 1e6) / sd
    hd_rate = (1920 * 1080 / 1e6) / hd
    assert hd_rate == pytest.approx(sd_rate * 0.75, rel=0.05)


def test_cpu_is_dramatically_slower():
    gpu = forecast(**SD, scale=2, model_name="2xParimgCompact").seconds
    cpu = forecast(**SD, scale=2, model_name="2xParimgCompact", cpu=True).seconds
    assert cpu > gpu * 20


def test_time_is_phrased_for_humans():
    assert "seconds" in Forecast(1, 1, 1, 1, 1, seconds=30).time_estimate
    assert "minutes" in Forecast(1, 1, 1, 1, 1, seconds=600).time_estimate
    assert "hours" in Forecast(1, 1, 1, 1, 1, seconds=7200).time_estimate
    assert "days" in Forecast(1, 1, 1, 1, 1, seconds=400000).time_estimate


# --- warnings ---------------------------------------------------------------


def test_a_four_times_model_on_hd_is_flagged():
    f = forecast(**HD, scale=4, model_name="4xPurePhoto-span")
    assert any("2x already reaches 4K" in w for w in f.warnings)


def test_a_four_times_model_on_sd_is_not_flagged():
    f = forecast(**SD, scale=4, model_name="4xPurePhoto-span")
    assert not any("already reaches 4K" in w for w in f.warnings)


def test_an_overnight_job_says_so():
    f = forecast(**HD, scale=2, model_name="RealESRGAN_x2plus")
    assert any("overnight" in w for w in f.warnings)


def test_a_quick_job_is_not_flagged_as_overnight():
    f = forecast(width=854, height=480, fps=24, frames=240, scale=2,
                 model_name="2xParimgCompact")
    assert not any("overnight" in w for w in f.warnings)


def test_telecine_with_interpolation_is_flagged_as_unsupported():
    f = forecast(**SD, scale=2, model_name="m", scan="telecined", target_fps=60)
    assert any("cannot be combined" in w for w in f.warnings)


def test_telecine_warns_that_resume_is_unavailable():
    f = forecast(**SD, scale=2, model_name="m", scan="telecined")
    assert any("cannot resume" in w for w in f.warnings)


# --- stills -----------------------------------------------------------------


def test_an_image_is_a_single_frame():
    f = forecast(width=4000, height=3000, fps=1, frames=1, scale=2,
                 model_name="2xParimgCompact", is_image=True)
    assert f.frames == 1
    assert f.width == 8000


def test_an_image_is_quick():
    f = forecast(width=4000, height=3000, fps=1, frames=1, scale=2,
                 model_name="2xParimgCompact", is_image=True)
    assert f.seconds < 60


# --- throughput lookup ------------------------------------------------------


def test_known_model_uses_its_measured_rate():
    assert throughput_for("2xParimgCompact.pth") == 8.0


def test_lookup_ignores_case_and_extension():
    assert throughput_for("2XPARIMGCOMPACT") == throughput_for("2xParimgCompact.pth")


def test_unknown_model_falls_back_to_its_architecture():
    assert throughput_for("4x_someones_span_model.pth") == 5.5


def test_a_completely_unknown_model_still_gets_a_rate():
    assert throughput_for("mystery.pth") > 0
