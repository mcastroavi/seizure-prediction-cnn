"""Train a CNN window encoder under a leakage-free protocol and save per-window risk scores.

    # leave-one-patient-out (20 train / 3 validation / 1 test patients per fold)
    python -m src.train --processed_dir data/processed_v3 --arch eegnet  --protocol lopo --out_dir results/eegnet_lopo
    python -m src.train --processed_dir data/processed_v3 --arch spectro --protocol lopo --out_dir results/spectro_lopo
    # patient-specific, forward in time
    python -m src.train --processed_dir data/processed_v3 --arch eegnet --protocol chrono --out_dir results/eegnet_chrono
    # a quick look at two folds
    python -m src.train --processed_dir data/processed_v3 --arch eegnet --folds chb01 chb05 --epochs 5

Each fold writes ``<out_dir>/<fold>/predictions.npz`` (validation and test risk for every
window, in recording order), ``model.pt`` and ``fold.json``. Then
``python -m src.evaluate --results_dir <out_dir>``.

Data loading: windows are read from the float16 ``X.npy`` memory maps a whole batch at a time
(indices sorted so reads are mostly sequential); z-scoring and augmentation run on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler

from .cnn import augment_batch, build, count_parameters, zscore
from .data import SubjectData, list_subjects, load_subject
from .losses import SoftSeizureLoss
from .splits import chronological_split, lopo_folds


# ── Data ─────────────────────────────────────────────────────────────────────

class WindowBatches(Dataset):
    """Item = a whole batch. ``ds[list_of_positions]`` -> (x float16 (B,C,T), risk, hard)."""

    def __init__(self, parts: list[SubjectData]):
        self.parts = parts
        self.part_id = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)]).astype(np.int64)
        self.local = np.concatenate([np.arange(len(p)) for p in parts]).astype(np.int64)
        self.hard = np.concatenate([p.hard for p in parts]).astype(np.int64)
        self.risk = np.concatenate([p.risk for p in parts]).astype(np.float32)

    def __len__(self):
        return len(self.local)

    def __getitem__(self, pos):
        pos = np.asarray(pos, dtype=np.int64)
        C, T = self.parts[0].X.shape[1:]
        x = np.empty((len(pos), C, T), dtype=np.float16)
        for k in np.unique(self.part_id[pos]):
            m = np.where(self.part_id[pos] == k)[0]
            p = self.parts[k]
            rows = p.sel[self.local[pos[m]]]
            order = np.argsort(rows)
            x[m[order]] = p.X[rows[order]]
        return torch.from_numpy(x), torch.from_numpy(self.risk[pos]), torch.from_numpy(self.hard[pos])


class BalancedBatches(Sampler):
    """``n`` windows per epoch, half preictal and half interictal (with replacement), in batches."""

    def __init__(self, hard: np.ndarray, n: int, batch_size: int, seed: int = 0):
        self.pos, self.neg = np.where(hard == 1)[0], np.where(hard == 0)[0]
        self.n, self.bs = n, batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        idx = np.concatenate([self.rng.choice(self.pos, self.n // 2), self.rng.choice(self.neg, self.n - self.n // 2)])
        self.rng.shuffle(idx)
        for s in range(0, self.n, self.bs):
            yield idx[s:s + self.bs]


class SequentialBatches(Sampler):
    def __init__(self, n: int, batch_size: int):
        self.n, self.bs = n, batch_size

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        for s in range(0, self.n, self.bs):
            yield np.arange(s, min(s + self.bs, self.n))


def loader(parts, sampler, num_workers):
    return DataLoader(WindowBatches(parts), sampler=sampler, batch_size=None, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation_subset(part: SubjectData, max_interictal: int, seed: int) -> SubjectData:
    """All preictal windows plus a fixed random sample of interictal ones (per-epoch AUC)."""
    neg = np.where(part.hard == 0)[0]
    if len(neg) > max_interictal:
        neg = np.sort(np.random.default_rng(seed).choice(neg, max_interictal, replace=False))
    return part.subset(np.sort(np.concatenate([np.where(part.hard == 1)[0], neg])))


# ── Train / predict ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_model(model, part: SubjectData, device, amp_dtype, batch_size=1024, num_workers=4, embed=False):
    """Risk scores (or 64-d embeddings) for one subject's windows, in recording order."""
    model.eval()
    out = []
    for x, _, _ in loader([part], SequentialBatches(len(part), batch_size), num_workers):
        x = zscore(x.to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            y = model.embed(x) if embed else torch.sigmoid(model(x).float()).view(-1)
        out.append(y.float().cpu().numpy())
    if out:
        return np.concatenate(out)
    return np.zeros((0, 64) if embed else 0, np.float32)


def train_fold(train_parts, val_parts, args, device, amp_dtype, init_state=None):
    seed_everything(args.seed)
    hard = np.concatenate([p.hard for p in train_parts])
    if (hard == 1).sum() == 0:
        raise RuntimeError("training set has no preictal windows")
    train_loader = loader(train_parts, BalancedBatches(hard, args.samples_per_epoch, args.batch_size, args.seed),
                          args.num_workers)
    val_small = [validation_subset(p, args.val_max_interictal, args.seed) for p in val_parts]

    model = build(args.arch).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    crit = SoftSeizureLoss(args.alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    best_auc, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, nb = 0.0, 0
        for x, risk, y in train_loader:
            x = zscore(x.to(device, non_blocking=True))
            if args.augment:
                x = augment_batch(x, gen)
            risk, y = risk.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = crit(logits, risk, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()

        # model selection on validation patients / validation period only
        probs = np.concatenate([run_model(model, p, device, amp_dtype, num_workers=args.num_workers) for p in val_small])
        vh = np.concatenate([p.hard for p in val_small])
        val_auc = roc_auc_score(vh, probs) if len(np.unique(vh)) > 1 else float("nan")
        history.append({"epoch": epoch, "train_loss": total / max(1, nb), "val_auc": val_auc})
        flag = ""
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, stale, flag = val_auc, 0, "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"    ep {epoch:02d}  loss {history[-1]['train_loss']:.4f}  val AUC {val_auc:.4f}  "
              f"[{time.time() - t0:.0f}s]{flag}", flush=True)
        if args.patience and stale >= args.patience:
            print(f"    early stop (no val improvement for {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_auc


def save_fold(out_dir, name, model, history, best_auc, val_parts, test_parts, args, device, amp_dtype,
              train_subjects):
    fold_dir = os.path.join(out_dir, name)
    os.makedirs(fold_dir, exist_ok=True)
    arrays = {}
    for split, parts in (("val", val_parts), ("test", test_parts)):
        for p in parts:
            key = f"{split}__{p.subject}"
            arrays[f"{key}__probs"] = run_model(model, p, device, amp_dtype, num_workers=args.num_workers)
            arrays[f"{key}__hard"] = p.hard
            arrays[f"{key}__block"] = p.block
            arrays[f"{key}__run"] = p.run
            if p.t_start is not None:
                arrays[f"{key}__t"] = p.t_start
    np.savez_compressed(os.path.join(fold_dir, "predictions.npz"), **arrays)
    torch.save({"model_state": model.state_dict(), "arch": args.arch, "val_auc": best_auc},
               os.path.join(fold_dir, "model.pt"))
    with open(os.path.join(fold_dir, "fold.json"), "w") as f:
        json.dump({"fold": name, "protocol": args.protocol, "arch": args.arch, "augment": args.augment,
                   "train": train_subjects, "val": [p.subject for p in val_parts],
                   "test": [p.subject for p in test_parts], "best_val_auc": best_auc,
                   "history": history, "args": vars(args)}, f, indent=2, default=str)


def load_encoder(path, device):
    ck = torch.load(path, map_location=device)
    model = build(ck["arch"]).to(device)
    model.load_state_dict(ck["model_state"])
    return model.eval()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--arch", choices=["eegnet", "spectro"], required=True)
    ap.add_argument("--protocol", choices=["lopo", "chrono"], default="lopo")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--folds", nargs="*", default=None, help="subjects to run (default: all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8, help="early stopping on val AUC (0 = off)")
    ap.add_argument("--samples_per_epoch", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5, help="MSE / BCE trade-off of the loss")
    ap.add_argument("--augment", action="store_true", help="waveform augmentation (shift, gain, noise, "
                                                           "time mask, channel dropout)")
    ap.add_argument("--n_val", type=int, default=3, help="validation patients per LOPO fold")
    ap.add_argument("--val_max_interictal", type=int, default=20000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or os.path.join("results", f"{args.arch}_{args.protocol}")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None
    torch.backends.cudnn.benchmark = True
    print(f"device={device}  amp={amp_dtype}  arch={args.arch} ({count_parameters(build(args.arch)):,} params)  "
          f"protocol={args.protocol}  augment={args.augment}")

    subjects = list_subjects(args.processed_dir)
    data = {s: load_subject(args.processed_dir, s) for s in subjects}

    if args.protocol == "lopo":
        folds = lopo_folds(subjects, {s: d.n_seizures for s, d in data.items()}, n_val=args.n_val, seed=args.seed)
        if args.folds:
            folds = [f for f in folds if f.name in args.folds]
        for f in folds:
            t0 = time.time()
            print(f"\n== fold {f.name}: train {len(f.train)} patients | val {f.val} | test {f.test}", flush=True)
            tr, va, te = ([data[s] for s in grp] for grp in (f.train, f.val, f.test))
            model, hist, auc = train_fold(tr, va, args, device, amp_dtype)
            save_fold(args.out_dir, f.name, model, hist, auc, va, te, args, device, amp_dtype, f.train)
            print(f"   fold done in {(time.time() - t0) / 60:.1f} min (best val AUC {auc:.3f})", flush=True)
    else:
        for s in args.folds or subjects:
            d = data[s]
            sp = chronological_split(d.block)
            if sp is None:
                print(f"\n== {s}: skipped ({d.n_seizures} seizures; need >= 3)")
                continue
            print(f"\n== {s}: train {len(sp['train'])} | val {len(sp['val'])} | test {len(sp['test'])} windows",
                  flush=True)
            tr, va, te = [d.subset(sp["train"])], [d.subset(sp["val"])], [d.subset(sp["test"])]
            model, hist, auc = train_fold(tr, va, args, device, amp_dtype)
            save_fold(args.out_dir, s, model, hist, auc, va, te, args, device, amp_dtype, [s])

    print(f"\nDone. Now run:  python -m src.evaluate --results_dir {args.out_dir}")


if __name__ == "__main__":
    main()
