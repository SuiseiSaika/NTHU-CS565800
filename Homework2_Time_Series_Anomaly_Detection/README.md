# HW2 - Time-Series Anomaly Detection

This assignment compares four unsupervised anomaly scores on the Wafer and ECG200 datasets. Training uses normal sequences; evaluation uses a resampled mix of normal and abnormal sequences.

## Methods

- **k-nearest neighbors:** mean Euclidean distance to the `k` closest normal training samples.
- **PCA reconstruction:** reconstruction-based scoring across principal-component counts.
- **Discrete Fourier transform:** k-NN scoring over truncated frequency coefficients.
- **Discrete wavelet transform:** k-NN scoring over prefixes of a multilevel Haar representation.

The original experiment uses NumPy seed `0`, removes anomalies from the training subset, sets the evaluation anomaly ratio to 0.1, and performs the original PCA, DFT, and DWT parameter sweeps.

## Recorded results

### Required setting (`k = 5`)

| Method | Wafer parameter | Wafer ROC-AUC | ECG200 parameter | ECG200 ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| k-NN | - | 0.988409 | - | 0.921875 |
| PCA | `n = 1` | 0.948328 | `n = 5` | 0.942708 |
| DFT | `M = 30` | 0.998330 | `M = 38` | 0.908854 |
| DWT | `S = 8` | **0.998590** | `S = 32` | **0.947917** |

### Best parameter search

| Method | Wafer setting | Wafer ROC-AUC | ECG200 setting | ECG200 ROC-AUC |
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

The remaining files in [`images/`](images/) show the parameter sweeps for each transformation and dataset.

## Run

Download Wafer and ECG200 from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) and arrange them as follows:

```text
Homework2_Time_Series_Anomaly_Detection/data/
|-- Wafer/
|   |-- Wafer_TRAIN.tsv
|   `-- Wafer_TEST.tsv
`-- ECG200/
    |-- ECG200_TRAIN.tsv
    `-- ECG200_TEST.tsv
```

Set `category` near the start of the main block to `Wafer` or `ECG200`, then run:

```bash
python -m pip install -r requirements.txt
python Homework2_Time_Series_Anomaly_Detection/main.py
```

For another dataset location, set `AD_HW2_DATA_ROOT` to the directory containing the `Wafer` and `ECG200` folders.

## Interpretation notes

- The evaluation distribution is resampled to a fixed anomaly ratio.
- Best-search values select parameters on the evaluation set and represent exploratory coursework.
- The exhaustive PCA and DFT searches require substantially more time than a single setting.
