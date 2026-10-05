import numpy as np
import pytest

torch = pytest.importorskip("torch")

from terraforge.models.explain import attention_rollout  # noqa: E402
from terraforge.models.mae import MaskedAutoencoder, patchify  # noqa: E402
from terraforge.models.vit import ViT  # noqa: E402
from terraforge.training.calibration import (  # noqa: E402
    expected_calibration_error, fit_temperature)
from terraforge.training.schedule import EMA, warmup_cosine  # noqa: E402


def small_vit():
    return ViT(dim=32, depth=2, heads=2, dropout=0.0)


def test_patchify_roundtrip_values():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    p = patchify(x, 2)
    assert p.shape == (2, 4, 12)
    assert p[0, 0].sort().values.tolist() == x[0, :, :2, :2].flatten().sort().values.tolist()


def test_mae_masks_requested_fraction_and_loss_finite():
    mae = MaskedAutoencoder(small_vit(), decoder_dim=16, decoder_depth=1, decoder_heads=2)
    loss, pred, mask = mae(torch.randn(4, 13, 64, 64), mask_ratio=0.75)
    assert pred.shape == (4, 64, 8 * 8 * 13)
    assert mask.sum(1).tolist() == [48.0] * 4  # 75% of 64 patches hidden
    assert torch.isfinite(loss)


def test_mae_loss_decreases_on_fixed_batch():
    torch.manual_seed(0)
    mae = MaskedAutoencoder(small_vit(), decoder_dim=32, decoder_depth=1, decoder_heads=2)
    x = torch.randn(8, 13, 64, 64)
    opt = torch.optim.AdamW(mae.parameters(), 2e-3)
    first = None
    for _ in range(40):
        opt.zero_grad()
        loss, *_ = mae(x)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.9


def test_mae_encoder_weights_transfer_to_classifier():
    enc = small_vit()
    MaskedAutoencoder(enc)  # shares the very same ViT object
    clf = small_vit()
    clf.load_state_dict(enc.state_dict())  # identical architecture -> loads cleanly


def test_warmup_cosine_shape():
    lrs = [warmup_cosine(s, 100, 10) for s in range(100)]
    assert lrs[0] < lrs[9] <= 1.0 and lrs[10] > lrs[60] > lrs[99] >= 0.01


def test_ema_tracks_model():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert not torch.allclose(ema.shadow.weight, m.weight)


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(np.eye(2)[y], y) == 0.0
    wrong = np.array([[1.0, 0.0]] * 4)  # 100% confident, 50% right
    assert abs(expected_calibration_error(wrong, y) - 0.5) < 1e-9


def test_temperature_scaling_reduces_overconfidence_without_changing_argmax():
    torch.manual_seed(0)
    y = torch.randint(0, 3, (500,))
    logits = torch.randn(500, 3) * 0.5
    logits[torch.arange(500), y] += 1.0
    logits = logits * 6  # artificially over-confident
    t = fit_temperature(logits, y)
    assert t > 1.5
    assert torch.equal(logits.argmax(1), (logits / t).argmax(1))
    before = expected_calibration_error(logits.softmax(1).numpy(), y.numpy())
    after = expected_calibration_error((logits / t).softmax(1).numpy(), y.numpy())
    assert after < before


def test_attention_rollout_is_distribution():
    m = small_vit()
    r = attention_rollout(m, torch.randn(2, 13, 64, 64))
    assert r.shape == (2, 8, 8)
    assert torch.allclose(r.flatten(1).sum(1), torch.ones(2), atol=1e-5)


def test_ddp_replicas_stay_identical(tmp_path):
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    from terraforge.training.distributed import run_ddp_smoke
    assert run_ddp_smoke(2, str(tmp_path)) < 1e-6
