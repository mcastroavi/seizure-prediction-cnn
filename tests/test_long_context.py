"""Long-context histories (causality, slots, gaps) and an end-to-end run on synthetic EDFs."""

import os
import sys
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import evaluate, long_context as LC, preprocess, train  # noqa: E402
from tests import make_synthetic_chbmit  # noqa: E402


def _history(times, emb, anchors, mask_only=False):
    hard = [np.zeros(len(a), np.int64) for a in anchors]
    risk = [np.zeros(len(a), np.float32) for a in anchors]
    return LC.History(times, emb, anchors, hard, risk, torch.device("cpu"), mask_only)


def test_slots_are_causal_and_complete():
    t = np.arange(0, 600, 5.0)                     # 10 min of contiguous 5-s windows
    e = np.repeat(t[:, None], 2, axis=1)           # embedding = window start time
    H = _history([t], [e], [np.array([300.0])])
    x = H.batch(torch.tensor([0]), 4)[0]           # last 2 minutes in four 30-s slots
    assert x.shape == (4, 3)
    assert torch.allclose(x[:, -1], torch.ones(4))                    # every slot fully recorded
    # newest slot = windows starting in (270, 300] -> 275..300, mean 287.5; nothing after 300
    assert np.isclose(float(x[-1, 0]), 287.5)
    assert np.isclose(float(x[0, 0]), 287.5 - 90)
    assert float(x[:, 0].max()) <= 300


def test_gaps_and_recording_start_are_empty_slots():
    t = np.concatenate([np.arange(0, 60, 5.0), np.arange(180, 240, 5.0)])   # 2-min gap
    e = np.ones((len(t), 2))
    H = _history([t], [e], [np.array([235.0])])
    frac = H.batch(torch.tensor([0]), 10)[0, :, -1].numpy()          # 5 min back: (-65, 235]
    # slots (-65,-35], (-35,-5] before the recording; (-5,25], (25,55] recorded; 4 gap slots; 2 recorded
    assert frac.tolist() == pytest.approx([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])


def test_subjects_do_not_mix_and_mask_only_hides_eeg():
    t = np.arange(0, 120, 5.0)
    H = _history([t, t], [np.zeros((24, 2)), np.ones((24, 2))], [np.array([115.0]), np.array([115.0])])
    x = H.batch(torch.tensor([0, 1]), 4)
    assert torch.all(x[0, :, 0] == 0) and torch.all(x[1, :, 0] == 1)
    M = _history([t, t], [np.zeros((24, 2)), np.ones((24, 2))], [np.array([115.0]), np.array([115.0])], True)
    xm = M.batch(torch.tensor([0, 1]), 4)
    assert torch.all(xm[..., :-1] == 0) and torch.equal(xm[..., -1], x[..., -1])


def test_end_to_end_on_synthetic_edfs():
    with tempfile.TemporaryDirectory() as d:
        raw, proc, res = (os.path.join(d, x) for x in ("raw", "processed", "results"))
        make_synthetic_chbmit.main(["--out", raw])
        preprocess.main(["--raw_dir", raw, "--out_dir", proc, "--preictal_min", "10",
                         "--postictal_min", "2", "--buffer_min", "20"])
        enc = os.path.join(res, "eegnet")
        train.main(["--processed_dir", proc, "--arch", "eegnet", "--protocol", "lopo", "--n_val", "1",
                    "--epochs", "2", "--samples_per_epoch", "1024", "--batch_size", "128",
                    "--num_workers", "0", "--out_dir", enc])
        cache = os.path.join(d, "cache")
        LC.main(["cache", "--processed_dir", proc, "--source", enc, "--out", cache])
        e = np.load(os.path.join(cache, "chb01", "chb01.npy"))
        meta = np.load(os.path.join(proc, "chb01", "meta.npz"))
        assert e.shape == (len(meta["t_start"]), 64)                    # every recorded window

        for extra in ([], ["--mask_only"], ["--timeline", "labelled"]):
            out = os.path.join(res, "long" + "".join(extra))
            LC.main(["train", "--processed_dir", proc, "--embeddings", cache, "--folds_from", enc,
                     "--out_dir", out, "--minutes", "3", "--epochs", "2", "--samples_per_epoch", "1024"] + extra)
            s = evaluate.evaluate(out, "fpr", 2.0, refractory_min=10, sop_min=10, min_preictal_min=3)
            assert s["n_test_subjects"] == 4 and s["seizures"] == 9
