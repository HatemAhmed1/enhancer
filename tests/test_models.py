import hashlib
import json

import pytest

from enhancer.models import ModelEntry, ModelRegistry, VerificationError, sha256_file


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
