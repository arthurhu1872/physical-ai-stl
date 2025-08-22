import pytest

# Skip these tests if PyTorch is not installed in the environment
torch = pytest.importorskip("torch")

from physical_ai_stl.monitoring import stl_soft as stl  # noqa: E402

def test_pred_leq_basic():
    u = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float32)
    c = 0.5
    margins = stl.pred_leq(u, c)
    # Expected margins: c - u for each element
    expected = torch.tensor([0.4, 0.1, 0.0], dtype=torch.float32)
    assert torch.allclose(margins, expected, atol=1e-8)

def test_softmin_softmax_bounds():
    x = torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32)
    # Soft-min should be approximately the min of x, soft-max approx the max
    sm = stl.softmin(x, temp=0.1)
    sM = stl.softmax(x, temp=0.1)
    true_min = x.min().item()
    true_max = x.max().item()
    # softmin should not exceed the true min by more than a tiny tolerance
    assert sm.item() <= true_min + 1e-6
    # softmax should not fall below the true max by more than a tiny tolerance
    assert sM.item() >= true_max - 1e-6
    # For a single-element tensor, softmin/softmax should return the element itself
    y = torch.tensor([0.42], dtype=torch.float32)
    assert torch.allclose(stl.softmin(y), y, atol=1e-8)
    assert torch.allclose(stl.softmax(y), y, atol=1e-8)

def test_stl_penalty_behavior():
    penalty = stl.STLPenalty(weight=1.0, margin=0.0)
    # For large positive robustness, penalty should be near 0
    rob_high = torch.tensor([10.0])
    # For large negative robustness (violation), penalty should be sizable
    rob_low = torch.tensor([-10.0])
    val_high = penalty(rob_high).item()
    val_low = penalty(rob_low).item()
    assert val_high < 1e-3
    assert val_low > 1.0
    # If weight=0, the penalty output should be 0 regardless of robustness
    penalty_zero = stl.STLPenalty(weight=0.0, margin=0.0)
    out = penalty_zero(torch.tensor([-1.0, 1.0]))
    assert float(out) == 0.0
