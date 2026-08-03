# HW2 — Time-Series Anomaly Detection

This project compares four unsupervised anomaly-scoring approaches on the Wafer and ECG200 datasets. Models are fitted on normal training sequences, then evaluated with ROC-AUC on a test split containing normal and abnormal samples.

## Methods

- **k-nearest neighbors:** mean Euclidean distance to the `k` closest normal training samples.
- **PCA reconstruction:** Euclidean reconstruction error after fitting PCA on normal training data.
- **Discrete Fourier transform:** k-NN distance over low-frequency magnitude coefficients.
- **Discrete wavelet transform:** k-NN distance over prefixes of a multilevel Haar representation.

The experiment runner uses a fixed NumPy seed (`0`). It retains all normal training samples, excludes training outliers, and resamples the evaluation data to an outlier ratio of 0.1.

## Reported results

These values are transcribed from the tracked course report. They were not regenerated during repository cleanup because the datasets are not bundled.

### Required setting (`k = 5`)

| Method | Wafer parameter | Wafer AUROC | ECG200 parameter | ECG200 AUROC |
| --- | ---: | ---: | ---: | ---: |
| k-NN | — | 0.988409 | — | 0.921875 |
| PCA | `n = 1` | 0.948328 | `n = 5` | 0.942708 |
| DFT | `M = 30` | 0.998330 | `M = 38` | 0.908854 |
| DWT | `S = 8` | **0.998590** | `S = 32` | **0.947917** |

### Best parameter search

| Method | Wafer setting | Wafer AUROC | ECG200 setting | ECG200 AUROC |
| --- | --- | ---: | --- | ---: |
| k-NN | `k = 1` | 0.991350 | `k = 2` | 0.955729 |
| PCA | `k = 10, n = 1` | 0.948331 | `k = 6, n = 5` | 0.950521 |
| DFT | `k = 2, M = 30` | 0.998467 | `k = 1, M = 50` | 0.927083 |
| DWT | `k = 10, S = 8` | **0.998673** | `k = 3, S = 32` | **0.966146** |

## Visual evidence

| Dataset | Raw sequences | PCA reconstruction | DFT reconstruction |
| --- | --- | --- | --- |
| Wafer | ![Wafer raw samples](images/Q1_Wafer.png) | ![Wafer PCA reconstruction](images/Q3_Wafer.png) | ![Wafer DFT reconstruction](images/Q4_Wafer.png) |
| ECG200 | ![ECG200 raw samples](images/Q1_ECG.png) | ![ECG200 PCA reconstruction](images/Q3_ECG.png) | ![ECG200 DFT reconstruction](images/Q4_ECG.png) |

The remaining plots in [`images/`](images/) record parameter sweeps for each method and dataset.

## Run locally

1. Create an environment and install the root requirements.
2. Obtain Wafer and ECG200 from the UCR Time Series Classification Archive.
3. Arrange the extracted TSV files as follows:

   ```text
   <data-root>/
   ├── Wafer/
   │   ├── Wafer_TRAIN.tsv
   │   └── Wafer_TEST.tsv
   └── ECG200/
       ├── ECG200_TRAIN.tsv
       └── ECG200_TEST.tsv
   ```

4. Run one dataset at a time:

   ```bash
   python Homework2_Time_Series_Anomaly_Detection/main.py \
     --data-root <data-root> \
     --category Wafer \
     --output results-wafer.json
   ```

The exhaustive PCA and DFT sweeps can be slow. Use `--max-k` to reduce the k-NN search during a smoke run.

## Limitations

- The public repository does not include the datasets, so the numeric tables are historical reported results rather than cleanup-time reruns.
- The evaluation split is resampled and therefore does not represent the datasets' natural anomaly prevalence.
- Hyperparameters are selected on the same evaluation data used for reporting, so the best-search table should be interpreted as exploratory coursework rather than a held-out benchmark.
