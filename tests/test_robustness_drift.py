import numpy as np
import pytest

from terraforge.training.drift import psi, reference_histograms, status


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(20000, 3))
    edges, probs = reference_histograms(ref)
    live = rng.normal(size=(20000, 3))
    assert psi(live, edges, probs).max() < 0.02


def test_psi_flags_shifted_band_only():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(20000, 3))
    edges, probs = reference_histograms(ref)
    live = rng.normal(size=(20000, 3))
    live[:, 1] += 1.0  # simulate a processing-baseline offset on band 1
    v = psi(live, edges, probs)
    assert status(v[1]) == "drifted" and status(v[0]) == "stable"


torch = pytest.importorskip("torch")

from terraforge.training.robustness import (  # noqa: E402
    band_dropout, cloud_patches, haze, robustness_report)


def test_corruptions_change_input_and_keep_shape():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 13, 64, 64)
    for fn in (band_dropout, haze, cloud_patches):
        y = fn(x, 0.5, g)
        assert y.shape == x.shape and not torch.equal(x, y)


def test_cloud_patches_cover_requested_area():
    g = torch.Generator().manual_seed(0)
    y = cloud_patches(torch.zeros(1, 13, 64, 64), 0.25, g)
    assert abs((y[0, 0] == 3.0).float().mean().item() - 0.25) < 1e-6


def test_report_is_deterministic_and_clean_matches_severity_zero():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(13 * 8 * 8, 4))
    x, y = torch.randn(32, 13, 8, 8), torch.randint(0, 4, (32,))
    loader = [(x, y)]
    r1 = robustness_report(model, loader, severities=(0.0, 0.5))
    r2 = robustness_report(model, loader, severities=(0.0, 0.5))
    assert r1 == r2
    assert len({r1[k][0.0] for k in r1}) == 1  # clean accuracy identical across corruptions
