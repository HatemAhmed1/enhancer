import subprocess
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_clip(tmp_path):
    """Generate a 2-second 320x240 25fps test clip with ffmpeg."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def synthetic_clip_with_audio(tmp_path):
    """Generate a 2-second 320x240 25fps test clip that also has an audio track."""
    path = tmp_path / "clip_audio.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path
