import numpy as np

from ml.train.metrics import evaluate, expected_calibration_error, brier_score
from ml.data.task_registry import TASK_IDS


def test_perfect_classifier_metrics():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 21)) > 0.7).astype(float)
    mask = y.copy()  # label exactly where positives are + some negatives
    mask[:, :] = (rng.random((500, 21)) > 0.5).astype(float)
    probs = y * mask  # perfect where labelled
    rep = evaluate(y, probs, mask, TASK_IDS)
    assert rep["summary"]["macro_auroc"] == 1.0


def test_ece_of_perfect_calibration_is_small():
    rng = np.random.default_rng(1)
    p = rng.random(10000)
    y = (rng.random(10000) < p).astype(float)
    assert expected_calibration_error(y, p) < 0.03


def test_brier_bounds():
    y = np.array([1.0, 0.0])
    assert brier_score(y, np.array([1.0, 0.0])) == 0.0
    assert brier_score(y, np.array([0.0, 1.0])) == 1.0
