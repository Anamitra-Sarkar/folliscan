import numpy as np
import pytest
import torch

from ml.train.losses import masked_bce, pinn_penalty, FolliscanLoss
from ml.train.uq import ConformalCalibrator


def test_masked_bce_ignores_unmasked():
    logits = torch.tensor([[5.0, 5.0]])
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])   # second task unlabelled
    loss = masked_bce(logits, labels, mask)
    assert float(loss) < 0.05           # confident correct prediction on task 0 only


def test_pinn_penalty_pushes_against_alert_contradiction():
    # alert fires; model predicts low hazard on constrained tox tasks -> positive penalty
    flags = [{"mutagenicity_alert": True, "sensitization_alert": False,
              "irritancy_alert": False, "cosing_alert": False}]
    probs_low = torch.full((1, 21), 0.9)
    probs_low[0, 14] = 0.05   # SR-p53
    probs_low[0, 15] = 0.05   # SR-ATAD5
    pen_low = pinn_penalty(probs_low, flags)

    probs_high = torch.full((1, 21), 0.95)   # hazard predicted everywhere
    pen_high = pinn_penalty(probs_high, flags)
    assert pen_low.item() > 0
    assert pen_high.item() < pen_low.item()
    assert pen_high.item() == pytest.approx(0.0, abs=1e-6)


def test_folliscan_loss_combines_terms():
    crit = FolliscanLoss(use_pinn=True, pinn_lambda=0.3)
    logits = torch.randn(4, 21)
    labels = (torch.rand(4, 21) > 0.5).float()
    mask = (torch.rand(4, 21) > 0.3).float()
    out = crit(logits, labels, mask,
               [{"mutagenicity_alert": True}] * 4)
    assert "total" in out and "loss_hair" in out and "loss_pinn" in out
    assert torch.isfinite(out["total"])


def test_conformal_coverage_near_nominal():
    rng = np.random.default_rng(0)
    cal = ConformalCalibrator(alpha=0.1, n_tasks=3)
    probs = rng.random((2000, 3))
    labels = (rng.random((2000, 3)) < probs).astype(float)
    mask = np.ones((2000, 3))
    cal.fit(probs[:1000], labels[:1000], mask[:1000])
    cov = cal.coverage(probs[1000:], labels[1000:], mask[1000:])
    assert cov["mean"] >= 0.88          # close to the 90% nominal target


def test_conformal_serialization_roundtrip(tmp_path):
    cal = ConformalCalibrator(alpha=0.05)
    rng = np.random.default_rng(1)
    cal.fit(rng.random((100, 21)), (rng.random((100, 21)) > 0.5).astype(float),
            np.ones((100, 21)))
    p = tmp_path / "calib.json"
    cal.save_json(str(p))
    cal2 = ConformalCalibrator.load_json(str(p))
    x = np.full((1, 21), 0.5)
    assert np.allclose(cal.predict_set(x), cal2.predict_set(x))
