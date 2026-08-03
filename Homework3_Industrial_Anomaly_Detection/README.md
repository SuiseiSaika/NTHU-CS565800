# HW3 - Industrial Anomaly Detection

## Overview

This executed notebook implements a compact SimpleNet-style pipeline for image-level and pixel-level anomaly detection on the MVTec AD `leather` and `metal_nut` categories.

## Method

- Wide ResNet-50 features from intermediate layers form a multi-scale representation.
- A feature adaptor maps pretrained features into the anomaly-detection space.
- Gaussian noise creates synthetic anomalies around normal training features.
- A discriminator produces image anomaly scores and dense anomaly maps.
- Image-level ROC-AUC, qualitative masks, model-size comparisons, adaptor ablation, and noise-scale ablation are recorded in the notebook.

The course configuration uses 16 training images per category, 20 epochs, and repeated runs for the main comparison. The notebook preserves the executed outputs, including curves, parameter comparisons, qualitative masks, and ablation plots.

## Repository contents

- `AD_HW3.ipynb`: implementation, executed experiments, plots, masks, and ablations.
- `README.md`: method, data layout, execution, evaluation, and interpretation notes.

## Data

Download [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) and place the selected categories under this directory:

```text
Homework3_Industrial_Anomaly_Detection/mvtec/
|-- leather/
`-- metal_nut/
```

## Execution

Install dependencies and start Jupyter from the assignment directory:

```bash
python -m pip install -r ../requirements.txt
cd Homework3_Industrial_Anomaly_Detection
jupyter lab AD_HW3.ipynb
```

The training cells use CUDA. The notebook is organized in assignment order so the architecture, training, evaluation, visualization, and ablation sections can be followed sequentially.

## Evaluation

The notebook reports image-level ROC-AUC, qualitative pixel masks, model parameter comparisons, and ablations for the feature adaptor and synthetic-noise scale.

## Results

The `leather` category was more difficult in these runs. Its texture defects can resemble normal local variation, making the anomaly boundary less distinct than the structural defects in `metal_nut`. Repeated training also showed sensitivity to initialization and synthetic-noise settings.

## Limitations

- The notebook uses a few-shot subset of normal images and selects the strongest run from repeated training.
- Image ROC-AUC and qualitative masks cover complementary aspects of detector quality.
- The ablations isolate the contribution of feature adaptation and synthetic-noise scale within the course configuration.

## References

The implementation follows the NTHU CS5658 assignment structure and is based on Liu et al., ["SimpleNet: A Simple Network for Image Anomaly Detection and Localization"](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf), CVPR 2023. The [official SimpleNet repository](https://github.com/DonaldRR/SimpleNet) provides the upstream reference implementation.
