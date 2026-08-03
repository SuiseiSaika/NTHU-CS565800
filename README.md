# Machine Learning for Anomaly Detection - Spring 2024

Coursework from NTHU CS5658, taught by Prof. Shang-Hong Lai. The four assignments move from classical anomaly scores on images and time series to deep industrial and video anomaly detection.

## Projects

| Assignment | Setting | Main methods | Recorded outcome |
| --- | --- | --- | --- |
| [HW1: MNIST Anomaly Detection](Homework1_MNIST_Anomaly_Detection/) | One digit is normal; the other digits are anomalies | k-NN, k-means, cosine and Minkowski distance, Mahalanobis distance, LOF | Best average ROC-AUC: 0.9792 |
| [HW2: Time-Series Anomaly Detection](Homework2_Time_Series_Anomaly_Detection/) | Wafer and ECG200 sequences | k-NN, PCA reconstruction, DFT, Haar DWT | Best ROC-AUC: 0.9987 on Wafer and 0.9661 on ECG200 |
| [HW3: Industrial Anomaly Detection](Homework3_Industrial_Anomaly_Detection/) | MVTec AD `leather` and `metal_nut` | SimpleNet, feature adaptation, synthetic anomalies, image and pixel scoring | Executed notebook with qualitative results and ablations |
| [HW4: Video Anomaly Detection](Homework4_Video_Anomaly_Detection/) | CUHK Avenue video | Jigsaw-VAD and two temporal objectives | Best micro-AUROC: 0.7944 |

## Repository layout

```text
.
|-- Homework1_MNIST_Anomaly_Detection/
|   |-- images/lof_tsne.png
|   |-- main.py
|   `-- README.md
|-- Homework2_Time_Series_Anomaly_Detection/
|   |-- images/
|   |-- main.py
|   `-- README.md
|-- Homework3_Industrial_Anomaly_Detection/
|   |-- AD_HW3.ipynb
|   `-- README.md
|-- Homework4_Video_Anomaly_Detection/
|   `-- README.md
`-- requirements.txt
```

## Environment

Create a Python environment and install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

Each assignment README describes its dataset layout, execution path, evaluation protocol, and interpretation notes. Datasets are acquired from their official sources and retain their original terms.

## Attribution

The assignments build on course starter material and the cited datasets and papers. HW3 follows [SimpleNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf), and HW4 follows [Jigsaw-VAD](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6451_ECCV_2022_paper.php). Third-party data and reference implementations remain under their respective terms.
