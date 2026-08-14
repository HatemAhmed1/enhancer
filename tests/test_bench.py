import numpy as np
import pytest
import torch

from enhancer.bench import BenchResult, benchmark


class FakeModel:
    scale = 2
    arch = "Fake"
    device = torch.device("cpu")

    def __call__(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    def to(self, device):
        return self


def test_benchmark_returns_result_with_positive_fps():
    r = benchmark(FakeModel(), width=64, height=64, frames=5, tile=64, overlap=0, device="cpu")
    assert isinstance(r, BenchResult)
    assert r.fps > 0
    assert r.frames == 5


def test_benchmark_records_output_dimensions():
    r = benchmark(FakeModel(), width=64, height=48, frames=2, tile=64, overlap=0, device="cpu")
    assert r.out_width == 128 and r.out_height == 96


def test_benchmark_estimates_feature_length_hours():
    r = benchmark(FakeModel(), width=32, height=32, frames=3, tile=32, overlap=0, device="cpu")
    expected = (172_800 / r.fps) / 3600
    assert r.feature_hours == pytest.approx(expected, rel=1e-6)
