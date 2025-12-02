# Machine Learning for Anomaly Detection (2024 Spring)

This repository contains all homework and the final project for the "Machine Learning for Anomaly Detection" course taught by **Prof. Shang-Hong Lai** at NTHU.

## 📚 Course Overview
This course introduces machine learning methods for detecting anomalies in images, time series, industrial data, and videos. 

---

## 📦 Assignments

### HW1 - MNIST Anomaly Detection
Apply various anomaly detection methods on the MNIST dataset by:
- Using one digit as the normal class, others as anomalies.
- Applying PCA for dimensionality reduction (to 30).
- Training only on normal data.
- Evaluating with ROC-AUC over all digit classes.

👉 Folder: [`Homework1_MNIST_Anomaly_Detection/`](./Homework1_MNIST_Anomaly_Detection)

---

### HW2 - Time Series Anomaly Detection
Detect anomalies in:
- **Wafer** and **ECG200** datasets.
- Use normal samples (label=1) for training, 9:1 ratio for testing.
- Feature extraction & anomaly algorithms applied.
- ROC-AUC used for evaluation.

👉 Folder: [`Homework2_Time_Series_Anomaly_Detection/`](./Homework2_Time_Series_Anomaly_Detection)

---

### HW3 - Industrial Anomaly Detection
Use **MVTecAD** datasets:
- Classes: `metal_nut` and `leather`
- Model: Self-implemented `SimpleNet` (CVPR 2023)
- Tasks: Anomaly detection + segmentation

References:
- [SimpleNet Paper (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
- [GitHub Repo](https://github.com/DonaldRR/SimpleNet)

👉 Folder: [`Homework3_Industrial_Anomaly_Detection/`](./Homework3_Industrial_Anomaly_Detection)

---

### HW4 - Video Anomaly Detection
Dataset: **CUHK Avenue**
- Training videos: Normal only
- Testing videos: Normal + Abnormal events
- Spatio-temporal cubes (5 frames)
- Epoch = 20, Filter ratio = 0.9

👉 Folder: [`Homework4_Video_Anomaly_Detection/`](./Homework4_Video_Anomaly_Detection)

---

### 🎬 Final Project
(Please fill in once your project topic is determined.)

👉 Folder: [`Final_Project/`](./Final_Project)

---
