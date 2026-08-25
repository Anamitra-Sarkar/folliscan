import torch
from torch_geometric.data import Batch

from ml.data.featurize import smiles_to_graph
from ml.data.motifs import motif_multihot, N_MOTIFS
from ml.models.folliscan_net import FolliscanNet, DEFAULT_CONFIG


def _batch(smis=("CCO", "c1ccccc1O", "O=C(C)Cc1ccccc1")):
    graphs = [smiles_to_graph(s) for s in smis]
    batch = Batch.from_data_list([g for g in graphs if g is not None])
    mo = torch.stack([torch.tensor(motif_multihot(s)) for s in smis])
    return batch, mo


def test_forward_shapes():
    net = FolliscanNet()
    batch, mo = _batch()
    out = net(batch, mo)
    assert out["logits"].shape == (3, 21)
    assert out["pathway_relevance"].shape[0] == 3
    assert out["motif_contrib"].shape == (3, N_MOTIFS)


def test_param_budget_under_50m():
    net = FolliscanNet()
    n = sum(p.numel() for p in net.parameters())
    assert n < 50_000_000, f"model too large: {n/1e6:.1f}M params"


def test_ablation_switches():
    net = FolliscanNet({"use_sme": False, "use_pathway": False})
    assert net.config["use_sme"] is False
    batch, mo = _batch(("CCO", "CCN"))
    out = net(batch, mo)
    assert out["logits"].shape == (2, 21)
    assert out["motif_contrib"] is None and out["pathway_relevance"] is None


def test_backward_pass():
    net = FolliscanNet()
    batch, mo = _batch()
    out = net(batch, mo)
    out["logits"].sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_mc_dropout_changes_predictions():
    from ml.train.uq import mc_predict

    torch.manual_seed(0)
    net = FolliscanNet()
    net.eval()
    batch, mo = _batch()
    m1, s1 = mc_predict(net, batch, mo, n_samples=4)
    m2, s2 = mc_predict(net, batch, mo, n_samples=4)
    assert s1.max() > 0            # dropout active -> stochastic variance
    assert abs(float(m1.mean()) - float(m2.mean())) < 0.5
