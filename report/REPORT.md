# Spacecraft anomaly detection with an LSTM predictor on SMAP/MSL telemetry

## 1. Problem

Detect anomalous periods in NASA spacecraft telemetry (SMAP satellite and MSL Curiosity
rover, 82 channels, 105 labeled anomaly intervals). The main metric is event-level
precision and recall: an anomaly is found if any alert overlaps its labeled interval.
Accuracy is not used, because anomalies are a tiny fraction of all timesteps.

## 2. Method

- **Preprocessing:** each channel is scaled to [0, 1] with min-max statistics fit on the
  training series only. Test data is scaled with those same statistics.
- **Model:** a 2-layer LSTM (80 hidden units, dropout 0.3, about 86k parameters) reads the
  previous 250 steps (all telemetry and command columns) and predicts the next telemetry value.
- **Training:** one model per channel. Adam (lr 1e-3), batch size 64, MSE loss, gradient
  clipping at 1.0, up to 35 epochs. The last 20% of the training series is held out for
  validation and used for early stopping (patience 10). Seed 42.
- **Detection:** absolute prediction error is smoothed with an EWMA (span 30). A threshold
  is chosen per channel by a dynamic search in the style of Telemanom, using median and MAD
  so that long anomalies do not inflate it. Consecutive points above the threshold form
  alerts. Alerts less than 20 steps apart are merged. Alerts whose peak is not clearly above
  the highest normal error are pruned (p = 0.13).

## 3. Results

Pre-committed main setting (p = 0.13), 78 channels:

| Subset | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| All | 37 | 30 | 64 | 0.552 | 0.366 | 0.440 |
| SMAP | 27 | 29 | 40 | 0.482 | 0.403 | 0.439 |
| MSL | 10 | 1 | 24 | 0.909 | 0.294 | 0.444 |

By anomaly type: point anomalies 30 of 58 found (52%), contextual 7 of 43 (16%).

Sensitivity checks (reported, not used to choose the main setting):

- Including P-10 and P-14: precision 0.565, recall 0.379, F1 0.453.
- Pruning off (p = 0.0), 80 channels: precision 0.520, recall 0.515, F1 0.517.
- Clipping scaled test inputs to [-0.5, 1.5]: no change in any metric.

## 4. Data handling and decisions

- **D-12** excluded: its training series has 312 steps, too short for a 250-step window.
- **P-10 and P-14** excluded from the main result: their test files are byte-identical but their
  labels differ, so at least one label set cannot be correct. Included in a sensitivity check.
- **P-2** has two label rows, [5350, 6575] and [5300, 6420]. They were merged into [5300, 6575].
- **T-10** has training and test data but no label row, so it cannot be scored.
- **Clipping** was chosen after looking at test-set range statistics. It is therefore a
  post-hoc check. It had no measurable effect, so the main result does not depend on it.

## 5. Limitations

- **Near-constant training telemetry.** A-1, R-1, D-5, P-14, E-6 and P-10 have almost no
  variation in training (A-1 and R-1 have a standard deviation near 1e-16). The model learns
  a flat line, and ordinary movement in the test data appears as error. Clipping does not help.
- **One channel dominates false alarms.** G-7 produces 31 alerts, 28 of them false. This
  accounts for 28 of the 29 SMAP false alarms, which suggests its normal behaviour is poorly
  modelled rather than that it contains many anomalies.
- **Contextual anomalies are mostly missed.** The next-step predictor only detects a value
  when it is far from the prediction, which suits point anomalies but not values that look
  normal in isolation.
- **Gap to published results.** The Telemanom paper reports F1 of about 0.71 on this data.
  Possible reasons: Telemanom predicts several steps ahead, uses a more elaborate threshold
  and pruning, and was tuned on this data. The comparison here is not like-for-like.

## 6. Reproducibility

The notebook `code/smap_msl_lstm_train.ipynb` contains the full run. Random seed 42.
Dependencies are listed in `requirements.txt`. Trained models and predictions are generated
by the notebook and are not included in the repository.

## 7. Possible next steps

- Multi-step prediction (predict k future values and score the combined error), which is
  the main difference from Telemanom and is the most likely route to better contextual detection.
- Investigate G-7 to find out why its normal behaviour is unstable.
- Re-evaluate on channels with non-constant training data separately.
