import hashlib
import json

import pytest

from enhancer.models import (
    ModelEntry,
    ModelRegistry,
    VerificationError,
    scan_custom_dir,
    sha256_file,
)


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
