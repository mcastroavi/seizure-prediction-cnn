# CNN seizure prediction on CHB-MIT

Does the graph in a spatio-temporal graph neural network (ST-GNN) help seizure prediction, or
would a plain CNN do as well? This repository replaces the ST-GNN window encoder of
[seizure-prediction-stgnn](https://github.com/mcastroavi/seizure-prediction-stgnn) with two CNNs
and keeps everything else identical, so any difference comes from the encoder alone:

* the same CHB-MIT windows and labels (5-s windows, 18 bipolar channels, 30-min preictal period),
* the same leakage-free protocols (leave-one-patient-out, and chronological per patient),
* the same 5-minute context model (GRU) on top of the 64-d window embeddings,
* the same seizure-level scoring: alarms, sensitivity, false alarms per hour, a random-predictor
  test, and a separate score on lead seizures only (≥ 4 h after the previous seizure).

| Encoder | Input | Parameters |
|---|---|---|
| **EEGNet** (Lawhern et al., 2018) | raw window, 18 × 640 samples | ~45 k |
| **Spectrogram CNN** (after Truong et al., 2018) | 0–40 Hz log-spectrogram per channel, computed on the GPU | ~0.3 M |

**Results:** in progress. See section 8 of the notebook.

## How to run

**Start with [`CNN_walkthrough.ipynb`](CNN_walkthrough.ipynb).** It explains each step, shows the
model code, trains one fold you can watch, starts the full experiments in the background and
collects the results. Run its cells top to bottom.

Requirements: the preprocessed data from the ST-GNN repo (`data/processed_v3`). The notebook finds
`~/seizure_stgnn_v3/data/processed_v3` automatically; otherwise link it:

```bash
git clone https://github.com/mcastroavi/seizure-prediction-cnn && cd seizure-prediction-cnn
pip install -r requirements.txt
ln -s /path/to/seizure-prediction-stgnn/data data     # or build it: python -m src.preprocess --raw_dir /path/to/chb-mit --out_dir data/processed_v3
python -m pytest -q tests                              # checks, ~1 min on CPU
jupyter notebook CNN_walkthrough.ipynb
```

Everything without the notebook: `bash run_all.sh /path/to/chb-mit`.

## Repository layout

| Path | Contents |
|---|---|
| `CNN_walkthrough.ipynb` | the walkthrough — start here |
| `src/cnn.py` | EEGNet, spectrogram CNN, GPU z-scoring and augmentation |
| `src/train.py` | training under LOPO / chronological splits, batched loading from the memory-mapped windows |
| `src/context.py` | 5-minute context GRU on the CNN embeddings |
| `src/losses.py` | soft-risk loss (MSE on risk + BCE on label) |
| `src/preprocess.py`, `data.py`, `splits.py`, `metrics.py`, `evaluate.py`, `figures.py`, ... | the tested pipeline from the ST-GNN repo, unchanged |
| `tests/` | unit tests and an end-to-end run on synthetic EDF files |
| `run_all.sh` | all experiments from one command |

`data/`, `results/` and `logs/` are created locally and are not stored in git.

## References

* Lawhern, V. J. et al. (2018). EEGNet: a compact convolutional neural network for EEG-based
  brain–computer interfaces. *Journal of Neural Engineering* 15(5).
* Truong, N. D. et al. (2018). Convolutional neural networks for seizure prediction using
  intracranial and scalp electroencephalogram. *Neural Networks* 105.
* Shoeb, A. (2009). Application of machine learning to epileptic seizure onset detection and
  treatment. PhD thesis, MIT (CHB-MIT Scalp EEG Database, PhysioNet).
* Schelter, B. et al. (2006). Testing statistical significance of multivariate time series
  analysis techniques for epileptic seizure prediction. *Chaos* 16(1).
