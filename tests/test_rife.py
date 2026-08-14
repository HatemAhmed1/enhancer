from pathlib import Path

import numpy as np
import pytest
import torch

from enhancer.rife import RIFE_DIR, RifeModel, load_rife


def _weights():
    return sorted(RIFE_DIR.glob("*.pkl")) + sorted(RIFE_DIR.glob("*.pth"))


def test_load_rife_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rife(tmp_path / "absent.pkl", device="cpu")


@pytest.mark.weights
def test_rife_loads_and_reports_its_device():
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    assert isinstance(model, RifeModel)
    assert model.device.type == "cpu"


@pytest.mark.weights
def test_rife_synthesises_a_frame_of_the_right_shape():
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    out = model(a, b, 0.5)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


@pytest.mark.weights
def test_rife_midpoint_lies_between_its_neighbours():
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 200, dtype=np.uint8)
    assert 0 < int(model(a, b, 0.5).mean()) < 200


@pytest.mark.weights
def test_rife_handles_dimensions_not_divisible_by_32():
    """RIFE pads internally; the caller must get its original size back."""
    found = _weights()
    if not found:
        pytest.skip("no RIFE weights present; see the setup guide")
    model = load_rife(found[0], device="cpu")
    a = np.zeros((70, 130, 3), dtype=np.uint8)
    b = np.full((70, 130, 3), 128, dtype=np.uint8)
    assert model(a, b, 0.5).shape == (70, 130, 3)
