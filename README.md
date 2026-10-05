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

## Results

Leave-one-patient-out: 24 folds, each tested on a patient the model never saw (151 test seizures,
65 of them lead seizures). Alarms use a threshold chosen on validation patients (≤ 0.5 false alarms/h
target). *Chance* is the sensitivity of a random predictor with the same false-alarm rate; *p* tests
the model against it. *Lead seizures* start ≥ 4 h after the previous seizure, so clustering cannot be
exploited. ST-GNN encoders are from [seizure-prediction-stgnn](https://github.com/mcastroavi/seizure-prediction-stgnn),
same data and protocol.

> **Correction (5 Oct 2026).** An earlier version of this table reported 5-minute context models
> (e.g. 66/151 and 21/65 lead seizures for the ST-GNN). Their history was built from labelled windows
> only, which leaks the label: a control model given **no EEG**, only that history's gap pattern,
> predicts **120/151** seizures. The context rows below use history read from every recorded window
> (`src/long_context.py`); the same no-EEG control then predicts 4/151. Details in
> [`Long_context_walkthrough.ipynb`](Long_context_walkthrough.ipynb).

### Window encoders (no context)

| Encoder | All seizures: predicted | FA/h | p | Lead seizures: predicted | chance | p |
|---|---|---|---|---|---|---|
| ST-GNN + bands | 52/151 (34%) | 0.53 | 0.001 | 16/65 | 18% | 0.12 |
| EEGNet | 50/151 (33%) | 0.62 | 0.05 | 11/65 | 20% | 0.77 |
| Spectrogram CNN | 39/151 (26%) | 0.50 | 0.17 | 12/65 | 17% | 0.39 |

### Context model on window embeddings (mean ± SD over 3 seeds)

| Encoder | History | All seizures: predicted | FA/h | p (median) | Lead seizures: predicted | chance | p (median) |
|---|---|---|---|---|---|---|---|
| none (no-EEG control) | 5 min, labelled windows only ⚠ | 120/151 | 0.21 | 9 × 10⁻⁹⁰ | 58/65 | 10% | 2 × 10⁻⁵⁰ |
| ST-GNN + bands | 5 min, labelled windows only ⚠ | 98 ± 2/151 | 0.41 | 2 × 10⁻³⁴ | 35 ± 2/65 | 17% | 4 × 10⁻¹¹ |
| none (no-EEG control) | 5 min, every window | 4/151 | 0.04 | 0.34 | 2/65 | 2% | 0.33 |
| none (no-EEG control) | 60 min, every window | 15/151 | 0.08 | 0.001 | 5/65 | 3% | 0.04 |
| **ST-GNN + bands** | **5 min, every window** | **61 ± 1/151 (41%)** | **0.48** | **4 × 10⁻⁸** | **14 ± 2/65** | 15% | 0.04 |
| ST-GNN + bands | 60 min, every window | 60 ± 3/151 | 0.43 | 5 × 10⁻⁹ | 14 ± 1/65 | 18% | 0.24 |
| EEGNet | 5 min, every window | 53 ± 3/151 | 0.54 | 7 × 10⁻⁴ | 15 ± 1/65 | 20% | 0.36 |
| EEGNet | 60 min, every window | 61 ± 2/151 | 0.56 | 7 × 10⁻⁶ | 19 ± 1/65 | 20% | 0.05 |

⚠ = leaky history, shown only to measure the leak. 15- and 30-minute results are in the notebook.

**Findings**
1. **No encoder generalises well to a new patient on its own.** Raw-signal CNN, spectrogram CNN and
   channel-synchrony graph all reach a window AUC of about 0.51, and none beats chance on lead seizures.
2. **Context built from labelled windows only leaks the label.** The gaps it leaves sit right before
   every preictal period; a model that never sees EEG exploits them to predict 120/151 seizures.
3. **With honest context, time still helps on all seizures** (ST-GNN 52 → 61/151 at a lower false-alarm
   rate), but on lead seizures no model is robustly above chance (ST-GNN 5 min: significant in 2 of 3 seeds).
4. **Longer history (15–60 min) does not help the ST-GNN** (60–63/151 at every length) and helps EEGNet a
   little (53 → 61/151); at 30–60 minutes the encoders are indistinguishable.
5. **Patient-specific (chronological) window models are too data-limited to rank**: 5–8 of 31 test
   seizures for every encoder (notebook `CNN_walkthrough.ipynb`, section 8).

**Bottom line:** without leaks, about 40% of seizures are predicted at ~0.5 false alarms/h, well above
chance overall but at best marginally above chance on seizures that do not follow another seizure.
Encoder choice and history length are not the bottleneck.

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
| `CNN_walkthrough.ipynb` | walkthrough 1: CNN encoders vs. the ST-GNN — start here |
| `Long_context_walkthrough.ipynb` | walkthrough 2: the history leak and 5–60-minute context |
| `src/cnn.py` | EEGNet, spectrogram CNN, GPU z-scoring and augmentation |
| `src/train.py` | training under LOPO / chronological splits, batched loading from the memory-mapped windows |
| `src/context.py` | 5-minute context GRU (history from labelled windows only: leaks, kept for reference) |
| `src/long_context.py` | context model with history from every recorded window, 5–60 min |
| `src/stgnn_embeddings.py` | embeds every recorded window with the ST-GNN repo's models |
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
