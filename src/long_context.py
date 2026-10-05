"""Long-context model: predict from the last 5-60 minutes of EEG, read from every recorded window.

    # 1) cache the window embeddings of every recorded window, once per encoder run
    python -m src.long_context cache --processed_dir data/processed_v3 --source results/eegnet_lopo \\
        --out cache/emb/eegnet_lopo
    # 2) train the context model (LOPO folds taken from the encoder run)
    python -m src.long_context train --processed_dir data/processed_v3 --embeddings cache/emb/eegnet_lopo \\
        --folds_from results/eegnet_lopo --minutes 60 --seed 1 --out_dir results/long/eegnet_60min_s1

Why every recorded window
-------------------------
Only interictal and preictal windows have labels; seizures, the 5 minutes after them and the
30-60-minute buffer before and after them are not scored. If the history is built from labelled
windows only (as ``src.context`` does), it breaks wherever unlabelled windows were dropped, and
those drops sit exactly in the 30 minutes before every preictal period. Then *how much* history
a window has gives away its label: every preictal window has < 30 min of labelled history, versus
35% of interictal windows. A live system sees the EEG continuously, so here the history is read
from **every** recorded window (labels are used only to pick and score the anchor windows) and
only real recording gaps leave holes. ``--timeline labelled`` rebuilds the old behaviour, and
``--mask_only`` trains on the gap pattern alone (no EEG) to measure how much any model could
gain from where the holes are.

History format
--------------
The history is cut into 30-s slots ending at the anchor window (causal: the newest slot is the
anchor's own 30 s; no later window is ever used). Each slot holds the mean embedding of the
windows recorded in it plus the fraction of the slot that was recorded (0 for a gap).
A GRU reads ``2 x minutes`` slots and outputs the risk for the anchor window.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .context import ContextGRU
from .data import SubjectData, list_subjects, load_subject
from .losses import SoftSeizureLoss

SLOT_SEC = 30.0
WINDOW_SEC = 5.0
PER_SLOT = int(SLOT_SEC / WINDOW_SEC)
SUBJECT_GAP = 1e9                      # time offset between subjects in the stacked timeline


# ── Timelines and embedding cache ────────────────────────────────────────────

def timeline(processed_dir: str, subject: str, which: str = "all"):
    """Sorted start times and X-row indices of the windows the history may read.

    ``all``: every recorded window. ``labelled``: interictal and preictal windows only.
    """
    meta = np.load(os.path.join(processed_dir, subject, "meta.npz"))
    t, label = meta["t_start"].astype(np.float64), meta["label"]
    rows = np.arange(len(t)) if which == "all" else np.where((label == 0) | (label == 1))[0]
    rows = rows[np.argsort(t[rows], kind="stable")]
    return t[rows], rows


def all_windows(d: SubjectData) -> SubjectData:
    """The same subject with every recorded window exposed (labels unused)."""
    n = d.X.shape[0]
    z = np.zeros(n, np.int64)
    return SubjectData(d.subject, d.X, d.plv, np.arange(n), z, z.astype(np.float32), z - 1, z,
                       None, dict(d.meta), d.plv_bands, d.bandpow)


def read_folds(source: str, subjects: list[str]) -> dict:
    folds = {}
    for f in sorted(os.listdir(source)):
        fj = os.path.join(source, f, "fold.json")
        if os.path.exists(fj):
            info = json.load(open(fj))
            if info.get("protocol", "lopo") != "lopo":
                raise SystemExit("long_context supports leave-one-patient-out folds only")
            if not info.get("train"):
                info["train"] = [s for s in subjects if s not in info["val"] + info["test"]]
            folds[f] = info
    if not folds:
        raise SystemExit(f"no fold.json under {source}")
    return folds


def cache_cnn_embeddings(source, processed_dir, out, folds=None, device=None):
    """Embed every recorded window of every subject a fold needs, with that fold's CNN encoder.

    Writes ``<out>/<fold>/<subject>.npy`` (float16, one row per X row)."""
    from .train import load_encoder, run_model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" else None
    subjects = list_subjects(processed_dir)
    for name, info in read_folds(source, subjects).items():
        if folds and name not in folds:
            continue
        enc = None
        for s in dict.fromkeys(info["train"] + info["val"] + info["test"]):
            path = os.path.join(out, name, f"{s}.npy")
            if os.path.exists(path):
                continue
            if enc is None:
                enc = load_encoder(os.path.join(source, name, "model.pt"), device)
            e = run_model(enc, all_windows(load_subject(processed_dir, s)), device, amp, embed=True)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, e.astype(np.float16))
        print(f"  {name}: embeddings cached", flush=True)


# ── Histories on the GPU ─────────────────────────────────────────────────────

class History:
    """Stacked timelines of several subjects with prefix sums, for fast slot histories.

    times[k]   (N_k,) sorted start times of the readable windows of subject k
    emb[k]     (N_k, D) their embeddings
    anchors[k] (M_k,) start times of the labelled windows of subject k (in prediction order)
    """

    def __init__(self, times, emb, anchors, hard, risk, device, mask_only=False):
        tg, eg, ag = [], [], []
        for k, (t, e, a) in enumerate(zip(times, emb, anchors)):
            tg.append(np.asarray(t, np.float64) + k * SUBJECT_GAP)
            eg.append(np.asarray(e, np.float64))
            ag.append(np.asarray(a, np.float64) + k * SUBJECT_GAP)
        E = np.concatenate(eg)
        self.D = E.shape[1]
        self.mask_only = mask_only
        self.T = torch.as_tensor(np.concatenate(tg), device=device)
        cum = np.concatenate([np.zeros((1, self.D)), np.cumsum(E, axis=0)])
        self.cum = torch.as_tensor(cum, device=device)
        self.ta = torch.as_tensor(np.concatenate(ag), device=device)
        self.hard = torch.as_tensor(np.concatenate(hard), device=device)
        self.risk = torch.as_tensor(np.concatenate(risk), dtype=torch.float32, device=device)
        self.bounds = np.cumsum([0] + [len(a) for a in anchors])     # anchor ranges per subject

    def __len__(self):
        return len(self.ta)

    def batch(self, idx: torch.Tensor, n_slots: int) -> torch.Tensor:
        """(B,) anchor indices -> (B, n_slots, D + 1): mean embedding and recorded fraction per
        30-s slot, oldest first. Slot j covers window starts in (t - 30(n-j), t - 30(n-j-1)]."""
        ta = self.ta[idx]
        steps = torch.arange(n_slots, -1, -1, device=ta.device, dtype=ta.dtype)
        edges = ta[:, None] - SLOT_SEC * steps[None, :]                      # (B, n_slots + 1)
        pos = torch.searchsorted(self.T, edges.contiguous(), right=True)
        lo, hi = pos[:, :-1], pos[:, 1:]
        cnt = (hi - lo).clamp(max=PER_SLOT)
        frac = (cnt.float() / PER_SLOT)[..., None]
        if self.mask_only:
            return torch.cat([torch.zeros(*cnt.shape, self.D, device=ta.device), frac], dim=-1)
        mean = (self.cum[hi] - self.cum[lo]) / (hi - lo).clamp(min=1)[..., None]
        return torch.cat([mean.float(), frac], dim=-1)


def load_history(subjects, parts, emb_dir, processed_dir, which, device, mask_only=False):
    times, emb = [], []
    for s in subjects:
        t, rows = timeline(processed_dir, s, which)
        e = np.load(os.path.join(emb_dir, f"{s}.npy"), mmap_mode="r")
        times.append(t)
        emb.append(np.asarray(e[rows], np.float32))
    return History(times, emb, [p.t_start for p in parts], [p.hard for p in parts],
                   [p.risk for p in parts], device, mask_only)


# ── Train / predict ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, H: History, idx: np.ndarray, n_slots: int, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        t = torch.as_tensor(idx[s:s + bs], device=H.ta.device)
        out.append(torch.sigmoid(model(H.batch(t, n_slots)).float()).view(-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def train_model(H: History, tr_idx, va_idx, args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n_slots = int(round(args.minutes * 60 / SLOT_SEC))
    model = ContextGRU(H.D, args.hidden, args.dropout).to(device)
    crit = SoftSeizureLoss(alpha=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hard = H.hard.cpu().numpy()
    pos, neg = tr_idx[hard[tr_idx] == 1], tr_idx[hard[tr_idx] == 0]

    vpos, vneg = va_idx[hard[va_idx] == 1], va_idx[hard[va_idx] == 0]
    if len(vneg) > 20000:
        vneg = rng.choice(vneg, 20000, replace=False)
    vsel = np.sort(np.concatenate([vpos, vneg]))
    vy = hard[vsel]

    best, best_state, stale, hist = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        anchors = np.concatenate([rng.choice(pos, args.samples_per_epoch // 2),
                                  rng.choice(neg, args.samples_per_epoch // 2)])
        rng.shuffle(anchors)
        anchors = torch.as_tensor(anchors, device=device)
        tot = 0.0
        for s in range(0, len(anchors), args.batch_size):
            t = anchors[s:s + args.batch_size]
            loss = crit(model(H.batch(t, n_slots)), H.risk[t], H.hard[t])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        p = predict(model, H, vsel, n_slots)
        auc = roc_auc_score(vy, p) if len(np.unique(vy)) > 1 else float("nan")
        hist.append({"epoch": ep, "loss": tot, "val_auc": auc})
        flag = ""
        if not np.isnan(auc) and auc > best:
            best, stale, flag = auc, 0, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if args.verbose:
            print(f"    ep {ep:02d}  val AUC {auc:.4f}  [{time.time() - t0:.1f}s]{flag}", flush=True)
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = list_subjects(args.processed_dir)
    folds = read_folds(args.folds_from, subjects)
    if args.folds:
        folds = {k: v for k, v in folds.items() if k in args.folds}
    n_slots = int(round(args.minutes * 60 / SLOT_SEC))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"long context | {args.minutes:g} min ({n_slots} slots of 30 s) | timeline={args.timeline} "
          f"| {'NO EEG (gap pattern only)' if args.mask_only else 'embeddings ' + args.embeddings} | seed {args.seed}",
          flush=True)
    data = {}
    for name, info in folds.items():
        t0 = time.time()
        subs = list(dict.fromkeys(info["train"] + info["val"] + info["test"]))
        for s in subs:
            if s not in data:
                data[s] = load_subject(args.processed_dir, s)
        parts = [data[s] for s in subs]
        H = load_history(subs, parts, os.path.join(args.embeddings, name), args.processed_dir,
                         args.timeline, device, args.mask_only)
        rng_of = {s: np.arange(H.bounds[k], H.bounds[k + 1]) for k, s in enumerate(subs)}
        tr_idx = np.concatenate([rng_of[s] for s in info["train"]])
        va_idx = np.concatenate([rng_of[s] for s in info["val"]])
        model, hist, best = train_model(H, tr_idx, va_idx, args, device)

        arrays = {}
        for split in ("val", "test"):
            for s in info[split]:
                p = data[s]
                key = f"{split}__{s}"
                arrays.update({f"{key}__probs": predict(model, H, rng_of[s], n_slots), f"{key}__hard": p.hard,
                               f"{key}__block": p.block, f"{key}__run": p.run, f"{key}__t": p.t_start})
        fd = os.path.join(args.out_dir, name)
        os.makedirs(fd, exist_ok=True)
        np.savez_compressed(os.path.join(fd, "predictions.npz"), **arrays)
        with open(os.path.join(fd, "fold.json"), "w") as fh:
            json.dump({"fold": name, "protocol": "lopo", "model": "long_context_gru", "minutes": args.minutes,
                       "timeline": args.timeline, "mask_only": args.mask_only, "embeddings": args.embeddings,
                       "train": info["train"], "val": info["val"], "test": info["test"],
                       "best_val_auc": best, "history": hist, "args": vars(args)}, fh, indent=2)
        print(f"  {name}: best val AUC {best:.3f}, {len(hist)} epochs, {time.time() - t0:.0f}s", flush=True)
        del H
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Done. Now run:  python -m src.evaluate --results_dir {args.out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache", help="embed every recorded window with each fold's CNN encoder")
    c.add_argument("--processed_dir", required=True)
    c.add_argument("--source", required=True, help="results dir of src.train (LOPO)")
    c.add_argument("--out", required=True)
    c.add_argument("--folds", nargs="*", default=None)
    t = sub.add_parser("train", help="train and predict the long-context model")
    t.add_argument("--processed_dir", required=True)
    t.add_argument("--embeddings", required=True, help="embedding cache dir (one sub-dir per fold)")
    t.add_argument("--folds_from", required=True, help="results dir whose fold.json files define the folds")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--minutes", type=float, default=60)
    t.add_argument("--timeline", choices=["all", "labelled"], default="all")
    t.add_argument("--mask_only", action="store_true", help="control: no EEG, only the recorded/gap pattern")
    t.add_argument("--folds", nargs="*", default=None)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--dropout", type=float, default=0.3)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--samples_per_epoch", type=int, default=40000)
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--verbose", action="store_true", help="print every epoch")
    a = ap.parse_args(argv)
    if a.cmd == "cache":
        cache_cnn_embeddings(a.source, a.processed_dir, a.out, a.folds)
    else:
        run(a)


if __name__ == "__main__":
    main()
