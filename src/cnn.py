"""CNN window encoders: EEGNet on the raw signal, and a CNN on log-spectrograms.

Both take one 5-second window, ``x`` of shape (B, C=18, T=640) at 128 Hz, already z-scored per
channel, and return

* ``embed(x)``  -> (B, 64) window embedding (what the context GRU reads), and
* ``forward(x)`` -> (B, 1) logit of preictal risk.

**EEGNet** (Lawhern et al., 2018, J. Neural Eng.): a temporal convolution learns frequency
filters, a depthwise convolution across all 18 channels learns spatial filters per frequency
filter, and a separable convolution summarises the result over time. It has no notion of a
channel graph: spatial structure is learnt as fixed linear combinations of channels.

**SpectroCNN** (in the spirit of Truong et al., 2018, Neural Networks): each channel's
short-time Fourier transform (1-s Hann window, 0.25-s hop, 0-40 Hz) is turned into a log-power
spectrogram, and a small 2-D CNN treats the 18 channels as input planes over (frequency, time).
The STFT runs on the GPU inside the model, so no extra preprocessing is stored on disk.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 64


class EEGNet(nn.Module):
    def __init__(self, n_channels: int = 18, n_samples: int = 640, fs: int = 128,
                 F1: int = 16, D: int = 2, F2: int = 32, dropout: float = 0.25):
        super().__init__()
        k1 = fs // 2                                   # 0.5-s temporal kernel: resolves >= 2 Hz
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),   # spatial, per filter
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            n_flat = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_samples))).numel()
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(n_flat, EMBED_DIM), nn.ELU(), nn.Dropout(dropout))
        self.head = nn.Linear(EMBED_DIM, 1)

    def embed(self, x):
        return self.proj(self.block2(self.block1(x.unsqueeze(1))))

    def forward(self, x):
        return self.head(self.embed(x))


class LogSpectrogram(nn.Module):
    """(B, C, T) -> (B, C, F, frames) log-power STFT, 0..fmax Hz, standardised per window."""

    def __init__(self, fs: int = 128, n_fft: int = 128, hop: int = 32, fmax: float = 40.0):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.n_freq = int(fmax * n_fft / fs) + 1
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def forward(self, x):
        B, C, T = x.shape
        s = torch.stft(x.reshape(B * C, T).float(), self.n_fft, self.hop, window=self.window,
                       center=True, return_complex=True)
        # power = re^2 + im^2 (avoids complex abs(), whose CUDA kernel is compiled at run time)
        p = torch.log(torch.view_as_real(s[:, :self.n_freq]).pow(2).sum(-1) + 1e-6)
        p = p.reshape(B, C, self.n_freq, -1)
        mu = p.mean(dim=(1, 2, 3), keepdim=True)
        sd = p.std(dim=(1, 2, 3), keepdim=True)
        return (p - mu) / (sd + 1e-5)


def _block(c_in, c_out, pool):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(pool))


class SpectroCNN(nn.Module):
    def __init__(self, n_channels: int = 18, fs: int = 128, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.spec = LogSpectrogram(fs)
        self.features = nn.Sequential(
            _block(n_channels, width, (2, 2)),
            _block(width, width * 2, (2, 2)),
            _block(width * 2, width * 4, (2, 1)),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, EMBED_DIM), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.head = nn.Linear(EMBED_DIM, 1)

    def embed(self, x):
        return self.proj(self.features(self.spec(x)))

    def forward(self, x):
        return self.head(self.embed(x))


ARCHS = {"eegnet": EEGNet, "spectro": SpectroCNN}


def build(arch: str, **kw) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch '{arch}' (choose from {sorted(ARCHS)})")
    return ARCHS[arch](**kw)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Batch helpers (GPU) ──────────────────────────────────────────────────────

def zscore(x: torch.Tensor) -> torch.Tensor:
    """Per-window, per-channel z-score (removes amplitude differences between patients)."""
    x = x.float()
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


def augment_batch(x: torch.Tensor, gen: torch.Generator | None = None,
                  p_shift=0.5, p_gain=0.5, gain=(0.8, 1.2), p_noise=0.5, noise_max=0.2,
                  p_mask=0.3, mask_frac=0.1, p_chdrop=0.3, max_chdrop=2) -> torch.Tensor:
    """Waveform augmentations of ``src/augment.py``, vectorised over a z-scored batch (B, C, T).

    Each transform is applied to each window independently with its own probability.
    """
    B, C, T = x.shape
    dev = x.device

    def coin(p):
        return torch.rand(B, device=dev, generator=gen) < p

    x = x.clone()
    # circular time shift up to +-50 %
    m = coin(p_shift)
    if m.any():
        sh = torch.randint(-T // 2, T // 2 + 1, (B,), device=dev, generator=gen) * m
        idx = (torch.arange(T, device=dev)[None, :] - sh[:, None]) % T
        x = x.gather(2, idx[:, None, :].expand(B, C, T))
    # per-channel gain
    g = torch.empty(B, C, 1, device=dev).uniform_(*gain, generator=gen)
    x = torch.where(coin(p_gain)[:, None, None], x * g, x)
    # additive gaussian noise with random SD
    sd = torch.rand(B, 1, 1, device=dev, generator=gen) * noise_max
    x = torch.where(coin(p_noise)[:, None, None], x + sd * torch.randn(x.shape, device=dev, generator=gen), x)
    # time mask (all channels)
    m = coin(p_mask)
    w = torch.randint(1, max(2, int(mask_frac * T)) + 1, (B,), device=dev, generator=gen)
    s = (torch.rand(B, device=dev, generator=gen) * (T - w + 1)).long()
    t = torch.arange(T, device=dev)[None, :]
    tm = (t >= s[:, None]) & (t < (s + w)[:, None]) & m[:, None]
    x = x.masked_fill(tm[:, None, :], 0.0)
    # channel dropout: up to max_chdrop channels zeroed
    m = coin(p_chdrop)
    k = torch.randint(1, max_chdrop + 1, (B,), device=dev, generator=gen)
    rank = torch.rand(B, C, device=dev, generator=gen).argsort(dim=1).argsort(dim=1)
    drop = (rank < k[:, None]) & m[:, None]
    return x.masked_fill(drop[:, :, None], 0.0)
