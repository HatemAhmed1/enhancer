import pytest

from enhancer.sources import (
    FORMAT_SELECTOR,
    SearchResult,
    _parse_entry,
    build_search_query,
    download,
    search,
)


def test_build_search_query_uses_ytsearch_extractor():
    assert build_search_query("kathakali dance", 5) == "ytsearch5:kathakali dance"


def test_build_search_query_rejects_nonpositive_count():
    with pytest.raises(ValueError):
        build_search_query("x", 0)


def test_parse_entry_extracts_fields():
    r = _parse_entry({
        "id": "abc123",
        "title": "Test Video",
        "duration": 754,
        "uploader": "Some Channel",
        "view_count": 12345,
        "url": "https://www.youtube.com/watch?v=abc123",
    })
    assert isinstance(r, SearchResult)
    assert r.video_id == "abc123"
    assert r.title == "Test Video"
    assert r.duration == 754
    assert r.uploader == "Some Channel"


def test_parse_entry_tolerates_missing_optional_fields():
    r = _parse_entry({"id": "xyz", "title": "Bare"})
    assert r.duration == 0
    assert r.uploader == ""
    assert r.view_count == 0
    assert r.url == "https://www.youtube.com/watch?v=xyz"


def test_parse_entry_requires_id():
    with pytest.raises(KeyError):
        _parse_entry({"title": "No ID"})


def test_duration_hms_formats_readably():
    assert _parse_entry({"id": "a", "title": "t", "duration": 754}).duration_hms == "12:34"
    assert _parse_entry({"id": "a", "title": "t", "duration": 3725}).duration_hms == "1:02:05"
    assert _parse_entry({"id": "a", "title": "t", "duration": 0}).duration_hms == "?"


def test_format_selector_prefers_vp9_av1_over_h264():
    """YouTube gives VP9/AV1 more bitrate at equal resolution (spec §5.0)."""
    assert "vp9" in FORMAT_SELECTOR.lower() or "av01" in FORMAT_SELECTOR.lower()
    assert "bestaudio" in FORMAT_SELECTOR


def test_search_returns_parsed_results(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, query, download):
            assert download is False, "search must never download"
            assert query == "ytsearch2:test"
            return {"entries": [
                {"id": "a", "title": "First", "duration": 60},
                {"id": "b", "title": "Second", "duration": 120},
            ]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    results = search("test", limit=2)
    assert [r.video_id for r in results] == ["a", "b"]


def test_search_skips_null_entries(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, query, download):
            return {"entries": [None, {"id": "a", "title": "Only"}, None]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    assert len(search("test", limit=3)) == 1


def test_download_returns_resolved_path(monkeypatch, tmp_path):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            assert download is True
            return {"id": "abc", "title": "Vid", "ext": "mp4",
                    "requested_downloads": [{"filepath": str(tmp_path / "Vid.mp4")}]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    out = download("https://www.youtube.com/watch?v=abc", tmp_path)
    assert out == tmp_path / "Vid.mp4"


def test_download_reports_progress(monkeypatch, tmp_path):
    seen = []

    class FakeYDL:
        def __init__(self, opts):
            self.hook = opts["progress_hooks"][0]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            self.hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100})
            return {"id": "abc", "ext": "mp4",
                    "requested_downloads": [{"filepath": str(tmp_path / "v.mp4")}]}

    monkeypatch.setattr("enhancer.sources.YoutubeDL", FakeYDL)
    download("u", tmp_path, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(50, 100)]
