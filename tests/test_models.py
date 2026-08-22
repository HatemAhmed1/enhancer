import hashlib
import json
from pathlib import Path

import pytest
import torch

from enhancer.models import (
    LoadedModel,
    ModelEntry,
    ModelRegistry,
    VerificationError,
    load_model,
    scan_custom_dir,
    sha256_file,
)

WEIGHTS_DIR = Path(__file__).parent.parent / "models"


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"indian cinema" * 1000)
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_sha256_file_handles_empty_file(tmp_path):
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_entry_parses_from_dict():
    entry = ModelEntry.from_dict(
        {
            "id": "span-2x",
            "tier": "fast",
            "arch": "SPAN",
            "scale": 2,
            "url": "https://example.invalid/span.pth",
            "sha256": "a" * 64,
            "size": 8_000_000,
        }
    )
    assert entry.id == "span-2x"
    assert entry.scale == 2
    assert entry.tier == "fast"


def test_entry_rejects_malformed_sha256():
    with pytest.raises(ValueError, match="sha256"):
        ModelEntry.from_dict(
            {
                "id": "bad",
                "tier": "fast",
                "arch": "SPAN",
                "scale": 2,
                "url": "https://example.invalid/x.pth",
                "sha256": "tooshort",
                "size": 1,
            }
        )


def test_verify_accepts_matching_file(tmp_path):
    p = tmp_path / "m.pth"
    p.write_bytes(b"weights")
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(b"weights").hexdigest(), size=7,
    )
    entry.verify(p)  # must not raise


def test_verify_rejects_corrupted_file(tmp_path):
    p = tmp_path / "m.pth"
    p.write_bytes(b"corrupted")
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(b"weights").hexdigest(), size=7,
    )
    with pytest.raises(VerificationError):
        entry.verify(p)


def test_registry_loads_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [
        {
            "id": "span-2x", "tier": "fast", "arch": "SPAN", "scale": 2,
            "url": "https://example.invalid/span.pth",
            "sha256": "a" * 64, "size": 8_000_000,
        }
    ]}))
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")
    assert [e.id for e in reg.list()] == ["span-2x"]


def test_registry_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    dup = {
        "id": "same", "tier": "fast", "arch": "SPAN", "scale": 2,
        "url": "https://example.invalid/a.pth", "sha256": "a" * 64, "size": 1,
    }
    manifest.write_text(json.dumps({"models": [dup, dup]}))
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")


def test_scan_custom_dir_finds_pth_files(tmp_path):
    d = tmp_path / "custom"
    d.mkdir()
    (d / "my_finetune.pth").write_bytes(b"x")
    (d / "other.safetensors").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")
    found = {p.name for p in scan_custom_dir(d)}
    assert found == {"my_finetune.pth", "other.safetensors"}


def test_scan_custom_dir_returns_empty_when_missing(tmp_path):
    assert scan_custom_dir(tmp_path / "nope") == []


def test_download_verifies_and_keeps_file(tmp_path, monkeypatch):
    payload = b"model-weights-payload"
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"models": []}')
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")

    monkeypatch.setattr(
        "enhancer.models._fetch", lambda url, dest, on_progress=None: dest.write_bytes(payload)
    )
    path = reg.ensure(entry)
    assert path.read_bytes() == payload


def test_download_deletes_file_on_hash_mismatch(tmp_path, monkeypatch):
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256="b" * 64, size=5,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"models": []}')
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")

    monkeypatch.setattr(
        "enhancer.models._fetch", lambda url, dest, on_progress=None: dest.write_bytes(b"wrong")
    )
    with pytest.raises(VerificationError):
        reg.ensure(entry)
    assert not reg.path_for(entry).exists(), "corrupt download must not be left behind"


def test_ensure_skips_download_when_file_already_valid(tmp_path, monkeypatch):
    payload = b"already-here"
    entry = ModelEntry(
        id="m", tier="fast", arch="SPAN", scale=2,
        url="https://example.invalid/m.pth",
        sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"models": []}')
    reg = ModelRegistry(manifest_path=manifest, cache_dir=tmp_path / "models")
    reg.cache_dir.mkdir(parents=True, exist_ok=True)
    reg.path_for(entry).write_bytes(payload)

    def boom(*a, **k):
        raise AssertionError("should not re-download a valid cached file")

    monkeypatch.setattr("enhancer.models._fetch", boom)
    assert reg.ensure(entry).read_bytes() == payload


@pytest.mark.weights
def test_load_model_detects_scale_and_arch():
    candidates = list(WEIGHTS_DIR.glob("*.pth"))
    if not candidates:
        pytest.skip("no weights downloaded; run the model manager first")
    m = load_model(candidates[0], device="cpu", half=False)
    assert isinstance(m, LoadedModel)
    assert m.scale in (1, 2, 4, 8)
    assert m.arch


@pytest.mark.weights
def test_loaded_model_upscales_by_declared_scale():
    candidates = list(WEIGHTS_DIR.glob("*.pth"))
    if not candidates:
        pytest.skip("no weights downloaded; run the model manager first")
    m = load_model(candidates[0], device="cpu", half=False)
    out = m(torch.rand(1, 3, 32, 32))
    assert out.shape[-2:] == (32 * m.scale, 32 * m.scale)


def test_load_model_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pth", device="cpu", half=False)


# ---------------------------------------------------------------------------
# The surface the per-frame CPU fallback depends on.
#
# LoadedModel shipped without a to() method for a long time, while
# upscale.Upscaler._process_on_cpu called model.to("cpu"). Nothing caught it,
# because the only test double had a to() of its own. These tests pin the
# contract on the real class, where a double cannot stand in for it.
# ---------------------------------------------------------------------------


def test_loaded_model_exposes_what_the_cpu_fallback_calls():
    import inspect

    assert callable(LoadedModel.to)
    params = inspect.signature(LoadedModel.to).parameters
    assert "device" in params, "the fallback moves the model to the processor"
    assert "dtype" in params, "and must cast it, since CPU float16 is a shim"
    # Both are read back off the parameters, so they cannot go stale after a
    # move; a plain attribute set once in __init__ could.
    assert isinstance(LoadedModel.device, property)
    assert isinstance(LoadedModel.dtype, property)


@pytest.mark.weights
def test_to_round_trips_device_and_dtype_and_stays_honest():
    """A real model, moved and cast exactly as the CPU fallback moves it."""
    candidates = list(WEIGHTS_DIR.glob("*.pth"))
    if not candidates:
        pytest.skip("no weights downloaded; run the model manager first")
    m = load_model(candidates[0], device="cpu", half=False)
    scale, arch, half_support = m.scale, m.arch, m.supports_half
    assert m.device.type == "cpu"
    assert m.dtype == torch.float32

    if half_support:
        m.to(dtype=torch.float16)
        assert m.dtype == torch.float16, "dtype must report the weights, not a flag"
        m.to("cpu", torch.float32)
        assert m.dtype == torch.float32

    # The architecture's own facts survive being moved about.
    assert (m.scale, m.arch, m.supports_half) == (scale, arch, half_support)
    assert m(torch.rand(1, 3, 16, 16)).shape[-2:] == (16 * scale, 16 * scale)


@pytest.mark.gpu
@pytest.mark.weights
def test_a_half_model_survives_a_trip_to_the_processor_and_back():
    """fp16 -> cpu/fp32 -> cuda/fp16 is the fallback's whole life cycle, and
    the cast back is lossless because every float16 value is exact in float32."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    candidates = list(WEIGHTS_DIR.glob("*.pth"))
    if not candidates:
        pytest.skip("no weights downloaded; run the model manager first")
    m = load_model(candidates[0], device="cuda", half=True)
    if m.dtype != torch.float16:
        pytest.skip(f"{m.arch} does not support half precision")
    before = next(m._d.model.parameters()).detach().clone()

    m.to("cpu", torch.float32)
    assert m.device.type == "cpu" and m.dtype == torch.float32
    out = m(torch.rand(1, 3, 16, 16))  # must not raise a dtype mismatch
    assert out.shape[-2:] == (16 * m.scale, 16 * m.scale)

    m.to("cuda", torch.float16)
    assert m.device.type == "cuda" and m.dtype == torch.float16
    assert torch.equal(next(m._d.model.parameters()), before)
