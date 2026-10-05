import pytest

torch = pytest.importorskip("torch")

from terraforge.models.cnn import SmallCNN  # noqa: E402
from terraforge.models.vit import MultiHeadSelfAttention, ViT  # noqa: E402


def test_cnn_output_shape():
    assert SmallCNN()(torch.randn(2, 13, 64, 64)).shape == (2, 10)


def test_vit_output_shape_and_token_count():
    m = ViT()
    assert m(torch.randn(2, 13, 64, 64)).shape == (2, 10)
    assert m.pos.shape[1] == 64 + 1  # 8x8 patches + CLS


def test_attention_rows_sum_to_one():
    a = MultiHeadSelfAttention(32, 4)
    a(torch.randn(2, 5, 32))
    assert torch.allclose(a.last_attention.sum(-1), torch.ones(2, 4, 5), atol=1e-5)


def test_vit_rejects_bad_patch():
    with pytest.raises(ValueError):
        ViT(img_size=64, patch=7)


def test_vit_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = ViT(dim=32, depth=2, heads=2, dropout=0.0)
    x, y = torch.randn(8, 13, 64, 64), torch.randint(0, 10, (8,))
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(60):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.5
