"""YouTube search and download via yt-dlp (spec §5.0).

yt-dlp is used as a library rather than a subprocess so that search results are
structured data. The `ytsearchN:` extractor is built in, so no API key or Google
account is needed.

Downloaded files are ordinary sources: they flow into SourceProfile.probe()
unchanged and get no special-case handling downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL

ProgressFn = Callable[[int, int], None]

# Prefer VP9/AV1 over H.264 at equal resolution: YouTube allocates them more
# bitrate, which matters when the result is about to be upscaled.
FORMAT_SELECTOR = (
    "bestvideo[vcodec^=av01]+bestaudio/"
    "bestvideo[vcodec^=vp9]+bestaudio/"
    "bestvideo+bestaudio/best"
)

WATCH_URL = "https://www.youtube.com/watch?v={}"


@dataclass(frozen=True)
class SearchResult:
    video_id: str
    title: str
    duration: int
    uploader: str
    view_count: int
    url: str

    @property
    def duration_hms(self) -> str:
        if not self.duration:
            return "?"
        h, rem = divmod(self.duration, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_search_query(query: str, limit: int) -> str:
    """Build a yt-dlp ytsearch pseudo-URL."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return f"ytsearch{limit}:{query}"


def _parse_entry(entry: dict) -> SearchResult:
    video_id = entry["id"]
    return SearchResult(
        video_id=video_id,
        title=str(entry.get("title") or ""),
        duration=int(entry.get("duration") or 0),
        uploader=str(entry.get("uploader") or ""),
        view_count=int(entry.get("view_count") or 0),
        url=str(entry.get("url") or WATCH_URL.format(video_id)),
    )


def search(query: str, limit: int = 10) -> list[SearchResult]:
    """Search YouTube. Returns metadata only — never downloads."""
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(build_search_query(query, limit), download=False)
    return [_parse_entry(e) for e in (info.get("entries") or []) if e]


def download(
    url: str,
    dest_dir: str | Path,
    on_progress: ProgressFn | None = None,
    format_selector: str = FORMAT_SELECTOR,
) -> Path:
    """Download a video and return the local path to the muxed file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            on_progress(int(d.get("downloaded_bytes") or 0), int(total))

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": format_selector,
        "merge_output_format": "mkv",
        "outtmpl": str(dest_dir / "%(title)s.%(ext)s"),
        "progress_hooks": [hook],
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    downloads = info.get("requested_downloads") or []
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])
    return dest_dir / f"{info.get('title', info['id'])}.{info.get('ext', 'mkv')}"
