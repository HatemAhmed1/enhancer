"""Hardware detection and encoder selection.

Nothing here may assume a GPU: the whole point of the module is that it works
on a machine that has none.
"""

import subprocess

import pytest

from enhancer import system


@pytest.fixture(autouse=True)
def _clean_encoder_cache():
    """The probed encoder is remembered for the process; tests must not share it."""
    system.reset_encoder_cache()
    yield
    system.reset_encoder_cache()


# --------------------------------------------------------------------------
# The quality flag table, as measured
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoder, expected",
    [
        ("hevc_nvenc", ["-cq", "20"]),
        ("h264_nvenc", ["-cq", "20"]),
        ("hevc_qsv", ["-global_quality", "20"]),
        ("hevc_amf", ["-rc", "cqp", "-qp_i", "20", "-qp_p", "20"]),
        ("hevc_vaapi", ["-qp", "20"]),
        ("libx265", ["-crf", "20"]),
        ("libx264", ["-crf", "20"]),
    ],
)
def test_quality_args_match_the_measured_table(encoder, expected):
    assert system.quality_args(encoder, 20) == expected


def test_videotoolbox_quality_is_inverted():
    """Every other encoder is "lower is better"; videotoolbox is the opposite.

    Passing a CRF number straight through would request the worst quality the
    encoder can produce whenever the user asked for the best.
    """
    best = system.quality_args("hevc_videotoolbox", 0)
    worst = system.quality_args("hevc_videotoolbox", 51)
    assert int(best[-1]) > int(worst[-1])
    assert 1 <= int(worst[-1]) <= 100
    assert 1 <= int(best[-1]) <= 100


def test_an_unknown_encoder_gets_crf():
    """Software encoders overwhelmingly take -crf; it is the least bad guess."""
    assert system.quality_args("libsvtav1", 30) == ["-crf", "30"]


def test_quality_args_never_shares_a_flag_across_families():
    """The bug this module fixes: -cq passed to libx265 is silently ignored.

    ffmpeg logs "Codec AVOption cq ... has not been used for any stream", exits
    0, and encodes at the encoder default. Nothing downstream ever notices.
    """
    assert "-cq" not in system.quality_args("libx265", 20)
    assert "-crf" not in system.quality_args("hevc_nvenc", 20)
    assert "-crf" not in system.quality_args("hevc_qsv", 20)


# --------------------------------------------------------------------------
# Pixel formats and bit depth
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "encoder, expected",
    [
        ("hevc_nvenc", "p010le"),
        ("hevc_qsv", "p010le"),
        ("hevc_amf", "p010le"),
        ("hevc_videotoolbox", "p010le"),
        ("libx265", "yuv420p10le"),
    ],
)
def test_ten_bit_pixel_formats(encoder, expected):
    assert system.pixel_format(encoder, 10) == expected


@pytest.mark.parametrize(
    "encoder",
    ["hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265", "libx264"],
)
def test_eight_bit_is_yuv420p_everywhere(encoder):
    assert system.pixel_format(encoder, 8) == "yuv420p"


def test_h264_encoders_are_treated_as_eight_bit():
    """h264_nvenc/qsv/amf genuinely cannot do 10-bit, and most libx264 builds
    are compiled 8-bit only. libx264 is a last-resort floor, so the safe answer
    costs nothing.
    """
    for encoder in ("libx264", "h264_nvenc", "h264_qsv", "h264_amf"):
        assert system.supports_10bit(encoder) is False
        assert system.pixel_format(encoder, 10) == "yuv420p"


def test_hevc_encoders_support_ten_bit():
    for encoder in ("hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"):
        assert system.supports_10bit(encoder) is True


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


def test_the_probe_frame_is_larger_than_nvencs_minimum():
    """NVENC's HEVC encoder rejects anything at or below 128 in either axis.

    Measured on an RTX 3060 Laptop: 128x128 fails with "Frame dimensions are
    less than the minimum supported value" and 129x129 succeeds. A 64x64 probe
    therefore reports NVENC as broken on a machine where it is the fastest
    encoder available, which is exactly the misdiagnosis this module prevents.
    """
    assert system.PROBE_SIZE > 128


def test_libx265_passes_a_real_probe():
    """The guaranteed floor has to actually run, or every fallback is a lie."""
    assert system.probe_encoder("libx265") is True


def test_a_nonexistent_encoder_fails_the_probe():
    assert system.probe_encoder("definitely_not_an_encoder") is False


def test_listing_includes_the_software_floor():
    listed = system.available_encoders()
    assert "libx265" in listed
    assert "libx264" in listed


def test_listing_alone_is_not_proof(monkeypatch):
    """ffmpeg advertises what it was compiled with, not what works.

    This build lists hevc_qsv, hevc_amf and hevc_vaapi on a laptop whose only
    working HEVC hardware is NVIDIA's; hevc_amf dies with "DLL amfrt64.dll
    failed to open". So a listed name must still be probed.
    """
    monkeypatch.setattr(system, "available_encoders", lambda: set(system.CANDIDATE_ENCODERS))
    probed = []

    def fake_probe(name, quality=20):
        probed.append(name)
        return name == "libx265"

    monkeypatch.setattr(system, "probe_encoder", fake_probe)
    assert system.choose_encoder(probe=True, use_cache=False) == "libx265"
    assert probed[0] == "hevc_nvenc", "must try the best candidate first"
    assert "hevc_amf" in probed, "must keep probing past a listed-but-dead encoder"


def test_probing_stops_at_the_first_success(monkeypatch):
    monkeypatch.setattr(system, "available_encoders", lambda: set(system.CANDIDATE_ENCODERS))
    probed = []

    def fake_probe(name, quality=20):
        probed.append(name)
        return True

    monkeypatch.setattr(system, "probe_encoder", fake_probe)
    assert system.choose_encoder(probe=True, use_cache=False) == "hevc_nvenc"
    assert probed == ["hevc_nvenc"], "probing costs real time; stop at the first hit"


def test_everything_failing_still_yields_an_encoder(monkeypatch):
    monkeypatch.setattr(system, "available_encoders", lambda: set(system.CANDIDATE_ENCODERS))
    monkeypatch.setattr(system, "probe_encoder", lambda name, quality=20: False)
    assert system.choose_encoder(probe=True, use_cache=False) == system.FALLBACK_ENCODER


def test_unlistable_ffmpeg_falls_back_rather_than_guessing(monkeypatch):
    """Without a listing and without probing there is nothing to decide on.

    Guessing the top candidate is precisely how the NVIDIA-only default got in.
    """
    monkeypatch.setattr(system, "available_encoders", set)
    assert system.choose_encoder(probe=False) == system.FALLBACK_ENCODER


def test_no_probe_mode_never_runs_ffmpeg(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("probe_encoders=False must not run an encode")

    monkeypatch.setattr(system, "probe_encoder", explode)
    hw = system.detect(probe_encoders=False)
    assert hw.encoder


def test_the_probe_result_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(system, "available_encoders", lambda: {"libx265"})

    def counting_probe(name, quality=20):
        calls.append(name)
        return True

    monkeypatch.setattr(system, "probe_encoder", counting_probe)
    first = system.default_encoder()
    second = system.default_encoder()
    assert first == second
    assert len(calls) == 1, "a few hundred ms per candidate; probe once"

    system.reset_encoder_cache()
    system.default_encoder()
    assert len(calls) == 2, "reset must be a real bypass, for tests"


def test_a_hanging_ffmpeg_does_not_wedge_the_probe(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(system.proc, "run", timeout)
    assert system.probe_encoder("hevc_nvenc") is False
    assert system.available_encoders() == set()


def test_a_missing_ffmpeg_does_not_raise(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(system.proc, "run", missing)
    assert system.probe_encoder("libx265") is False
    assert system.choose_encoder(probe=True, use_cache=False) == system.FALLBACK_ENCODER


# --------------------------------------------------------------------------
# detect()
# --------------------------------------------------------------------------


def test_detect_reads_this_machine():
    hw = system.detect(probe_encoders=False)
    assert hw.os_name in {"Windows", "Linux", "Darwin"}
    assert hw.machine
    assert hw.cpu_name and hw.cpu_name != "unknown"
    assert hw.cpu_cores >= 1
    assert hw.ram_bytes > (1 << 30), "any machine that can run torch has over 1 GiB"
    assert hw.accelerator in {"cuda", "mps", "xpu", "cpu"}
    assert hw.encoder


def test_no_accelerator_means_no_gpu_name():
    hw = system.detect(probe_encoders=False)
    if hw.accelerator == "cpu":
        assert hw.gpu_name == ""
        assert hw.vram_bytes == 0
    else:
        assert hw.gpu_name


def test_encoder_is_hardware_agrees_with_the_name():
    hw = system.detect(probe_encoders=False)
    assert hw.encoder_is_hardware == (hw.encoder in system.HARDWARE_ENCODERS)
    if hw.encoder.startswith("lib"):
        assert hw.encoder_is_hardware is False


def test_detect_survives_torch_being_unimportable(monkeypatch):
    """A machine with no torch is a CPU machine, not a crash."""
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    hw = system.detect(probe_encoders=False)
    assert hw.accelerator == "cpu"
    assert hw.gpu_name == ""
    assert hw.vram_bytes == 0


def test_detect_survives_a_torch_that_raises(monkeypatch):
    """Older and broken torch builds raise from is_available(), not return False."""
    torch = pytest.importorskip("torch")

    def explode():
        raise RuntimeError("driver is unhappy")

    monkeypatch.setattr(torch.cuda, "is_available", explode)
    hw = system.detect(probe_encoders=False)
    assert hw.accelerator in {"mps", "xpu", "cpu"}


def test_detect_survives_unreadable_ram_and_cpu(monkeypatch):
    monkeypatch.setattr(system, "_ram_bytes", lambda: 0)
    monkeypatch.setattr(system, "_cpu_name", lambda: "unknown")
    hw = system.detect(probe_encoders=False)
    assert hw.ram_bytes == 0
    assert "unknown" in system.describe(hw)


def test_detect_survives_encoder_selection_blowing_up(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("selection is broken")

    monkeypatch.setattr(system, "choose_encoder", explode)
    assert system.detect(probe_encoders=False).encoder == system.FALLBACK_ENCODER


# --------------------------------------------------------------------------
# describe()
# --------------------------------------------------------------------------


def test_describe_is_a_few_short_lines():
    text = system.describe(system.detect(probe_encoders=False))
    lines = text.splitlines()
    assert 3 <= len(lines) <= 8
    assert all(len(line) < 100 for line in lines)


def test_describe_names_the_encoder_and_whether_it_is_hardware():
    hw = system.detect(probe_encoders=False)
    text = system.describe(hw)
    assert hw.encoder in text
    assert ("hardware" in text) == hw.encoder_is_hardware


def test_describe_says_so_when_there_is_no_gpu():
    hw = system.Hardware(
        os_name="Linux", machine="x86_64", cpu_name="Some CPU", cpu_cores=4,
        ram_bytes=8 << 30, accelerator="cpu", gpu_name="", vram_bytes=0,
        encoder="libx265", encoder_is_hardware=False,
    )
    text = system.describe(hw)
    assert "CPU only" in text
    assert "8.0 GiB" in text
    assert "libx265" in text and "software" in text
