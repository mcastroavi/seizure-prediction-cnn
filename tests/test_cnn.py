"""CNN encoders, GPU augmentation, batched loading, and an end-to-end run on synthetic EDFs.

Runs on CPU in about a minute.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import context, evaluate, preprocess, train  # noqa: E402
from src.cnn import EMBED_DIM, LogSpectrogram, augment_batch, build, zscore  # noqa: E402
from src.data import load_subject  # noqa: E402
from tests import make_synthetic_chbmit  # noqa: E402


@pytest.mark.parametrize("arch", ["eegnet", "spectro"])
def test_encoder_shapes(arch):
    m = build(arch).eval()
    x = torch.randn(4, 18, 640)
    assert m.embed(x).shape == (4, EMBED_DIM)
    assert m(x).shape == (4, 1)


def test_spectrogram_band_and_shape():
    spec = LogSpectrogram(fs=128)
    t = torch.arange(640) / 128.0
    x = torch.sin(2 * np.pi * 10 * t).repeat(1, 18, 1)          # 10 Hz on every channel
    s = spec(x)
    assert s.shape == (1, 18, 41, 21)                            # 0..40 Hz in 1-Hz bins, 21 frames
    assert int(s[0, 0].mean(dim=1).argmax()) == 10


def test_zscore_per_channel():
    x = torch.randn(2, 18, 640) * 50 + 7
    z = zscore(x)
    assert torch.allclose(z.mean(-1), torch.zeros(2, 18), atol=1e-4)
    assert torch.allclose(z.std(-1), torch.ones(2, 18), atol=1e-3)


def test_augment_batch_is_label_free_and_varied():
    g = torch.Generator().manual_seed(0)
    x = zscore(torch.randn(64, 18, 640))
    y = augment_batch(x, g)
    assert y.shape == x.shape and torch.isfinite(y).all()
    changed = (y - x).abs().flatten(1).max(1).values > 1e-6
    assert 0 < changed.sum() < 64 or changed.all()               # independent per window
    # all transforms off -> identity
    z = augment_batch(x, g, p_shift=0, p_gain=0, p_noise=0, p_mask=0, p_chdrop=0)
    assert torch.equal(z, x)
    # channel dropout only: at most 2 channels zeroed per window
    z = augment_batch(x, g, p_shift=0, p_gain=0, p_noise=0, p_mask=0, p_chdrop=1.0)
    zeroed = (z.abs().sum(-1) == 0).sum(1)
    assert zeroed.min() >= 1 and zeroed.max() <= 2


def test_end_to_end_on_synthetic_edfs():
    with tempfile.TemporaryDirectory() as d:
        raw, proc, res = (os.path.join(d, x) for x in ("raw", "processed", "results"))
        make_synthetic_chbmit.main(["--out", raw])
        preprocess.main(["--raw_dir", raw, "--out_dir", proc, "--preictal_min", "10",
                         "--postictal_min", "2", "--buffer_min", "20"])

        # batched loading returns exactly the windows a per-window read would
        d1 = load_subject(proc, "chb01")
        ds = train.WindowBatches([d1])
        pos = np.array([5, 0, 3, len(d1) - 1])
        x, risk, hard = ds[pos]
        for k, i in enumerate(pos):
            assert np.array_equal(x[k].numpy(), np.asarray(d1.window(i)[0]))
        assert np.array_equal(hard.numpy(), d1.hard[pos])

        out = os.path.join(res, "eegnet")
        train.main(["--processed_dir", proc, "--arch", "eegnet", "--protocol", "lopo", "--n_val", "1",
                    "--epochs", "3", "--samples_per_epoch", "1024", "--batch_size", "128",
                    "--num_workers", "0", "--augment", "--out_dir", out])
        s = evaluate.evaluate(out, "fpr", 2.0, refractory_min=10, sop_min=10, min_preictal_min=3)
        assert s["n_test_subjects"] == 4 and s["seizures"] == 9
        # the synthetic preictal signal (9 Hz oscillation) is easy: windows must separate
        assert s["window_auc_mean_per_subject"] > 0.8

        ctx = os.path.join(res, "context_eegnet")
        context.main(["--processed_dir", proc, "--source", out, "--out_dir", ctx, "--epochs", "2",
                      "--samples_per_epoch", "2048", "--seq_len", "12"])
        s = evaluate.evaluate(ctx, "fpr", 2.0, refractory_min=10, sop_min=10, min_preictal_min=3)
        assert s["n_test_subjects"] == 4 and s["window_auc_mean_per_subject"] > 0.8

        out = os.path.join(res, "spectro_chrono")
        train.main(["--processed_dir", proc, "--arch", "spectro", "--protocol", "chrono", "--epochs", "2",
                    "--samples_per_epoch", "512", "--batch_size", "128", "--num_workers", "0",
                    "--out_dir", out])
        assert os.path.exists(os.path.join(out, "chb03", "predictions.npz"))
