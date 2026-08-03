# Machine Learning for Anomaly Detection — Spring 2024

Selected coursework from NTHU CS5658, taught by Prof. Shang-Hong Lai. The repository focuses on two complementary settings: classical time-series anomaly detection and self-supervised video anomaly detection.

## Published work

| Project | Problem | Methods | Best reported result |
| --- | --- | --- | --- |
| [HW2: Time-Series Anomaly Detection](Homework2_Time_Series_Anomaly_Detection/) | Detect abnormal Wafer and ECG200 sequences | k-NN, PCA reconstruction, DFT features, Haar DWT features | Wafer AUROC 0.9987; ECG200 AUROC 0.9661 |
| [HW4: Video Anomaly Detection](Homework4_Video_Anomaly_Detection/) | Detect unusual events in CUHK Avenue video | Jigsaw-VAD and two temporal-classification variants | micro-AUROC 0.7944 |

Only HW2 and the HW4 experiment report are present in this public repository. Earlier references to HW1, HW3, and a final project did not correspond to tracked artifacts and have been removed from the navigation.

## Repository layout

```text
.
├── Homework2_Time_Series_Anomaly_Detection/
│   ├── images/          # Tracked experiment plots
│   ├── main.py          # Portable experiment runner
│   └── README.md        # Methods, results, and reproduction notes
├── Homework4_Video_Anomaly_Detection/
│   └── README.md        # Experiment design and reported results
└── requirements.txt
```

## Reproduction scope

The datasets, model checkpoints, and the modified HW4 training implementation are not included. HW2 can be rerun after obtaining Wafer and ECG200 separately; see its project README for the expected directory layout. HW4 is preserved as a results report and cannot be reproduced from this repository alone.

## Attribution

- The Wafer and ECG200 datasets are distributed through the UCR Time Series Classification Archive.
- HW4 builds on Wang et al., “Video Anomaly Detection by Solving Decoupled Spatio-Temporal Jigsaw Puzzles,” ECCV 2022. See the [paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6451_ECCV_2022_paper.php) and [official implementation](https://github.com/gdwang08/Jigsaw-VAD).

Coursework is retained for portfolio and educational review. No repository-level software license is asserted; third-party datasets and reference implementations remain subject to their own terms.
