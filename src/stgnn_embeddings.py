"""Cache ST-GNN window embeddings of every recorded window, using the ST-GNN repository's code.

Run with the ST-GNN repository available (it needs torch-geometric, like that repo):

    python src/stgnn_embeddings.py --stgnn_repo ~/seizure_stgnn_v3 \\
        --source ~/seizure_stgnn_v3/results/stgnn_bands_lopo --out cache/emb/stgnn_bands_lopo

Writes ``<out>/<fold>/<subject>.npy`` (float16, one row per X row), the same format as
``python -m src.long_context cache`` writes for the CNN encoders.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stgnn_repo", required=True)
    ap.add_argument("--source", required=True, help="ST-GNN LOPO results dir (one model.pt per fold)")
    ap.add_argument("--processed_dir", default=None, help="default: <stgnn_repo>/data/processed_v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args(argv)
    repo = os.path.abspath(os.path.expanduser(a.stgnn_repo))
    source, out = os.path.abspath(os.path.expanduser(a.source)), os.path.abspath(a.out)
    proc = os.path.abspath(os.path.expanduser(a.processed_dir or os.path.join(repo, "data/processed_v3")))
    sys.path.insert(0, repo)                     # import the ST-GNN repository's src package

    import torch
    from src.context import embed_part, load_encoder        # noqa: E402  (ST-GNN repo)
    from src.data import SubjectData, list_subjects, load_subject  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" else None
    subjects = list_subjects(proc)
    for name in sorted(os.listdir(source)):
        fj = os.path.join(source, name, "fold.json")
        if not os.path.exists(fj) or (a.folds and name not in a.folds):
            continue
        info = json.load(open(fj))
        train = info.get("train") or [s for s in subjects if s not in info["val"] + info["test"]]
        t0, enc = time.time(), None
        for s in dict.fromkeys(train + info["val"] + info["test"]):
            path = os.path.join(out, name, f"{s}.npy")
            if os.path.exists(path):
                continue
            if enc is None:
                enc, enc_args = load_encoder(os.path.join(source, name, "model.pt"), device)
                enc_args.num_workers = a.num_workers
            d = load_subject(proc, s)
            n = d.X.shape[0]
            z = np.zeros(n, np.int64)
            full = SubjectData(s, d.X, d.plv, np.arange(n), z, z.astype(np.float32), z - 1, z, None,
                               dict(d.meta), d.plv_bands, d.bandpow)
            e = embed_part(enc, full, enc_args, device, amp)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, e.astype(np.float16))
        print(f"  {name}: embeddings cached in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
