"""Run the CS5658 HW2 time-series anomaly-detection experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score
from tqdm import tqdm


RANDOM_SEED = 0


def resample(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    outlier_ratio: float,
    normal_label: int = 1,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all normal samples plus a requested ratio of abnormal samples."""
    normal_data = data[labels == normal_label]
    abnormal_data = data[labels != normal_label]
    abnormal_count = int(len(normal_data) * outlier_ratio)

    if abnormal_count > len(abnormal_data):
        raise ValueError(
            f"Requested {abnormal_count} outliers, but only {len(abnormal_data)} exist."
        )

    if abnormal_count:
        indices = rng.choice(len(abnormal_data), abnormal_count, replace=False)
        selected_abnormal = abnormal_data[indices]
    else:
        selected_abnormal = abnormal_data[:0]

    new_data = np.concatenate((normal_data, selected_abnormal))
    new_labels = np.concatenate(
        (np.zeros(len(normal_data)), np.ones(len(selected_abnormal)))
    )
    return new_data, new_labels


def mean_neighbor_score(
    train_features: np.ndarray, test_features: np.ndarray, k: int
) -> np.ndarray:
    """Score samples by mean distance to their k closest training samples."""
    distances = pairwise_distances(test_features, train_features, metric="euclidean")
    return np.mean(np.partition(distances, k - 1, axis=1)[:, :k], axis=1)


def knn_auc(
    train_data: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, k: int
) -> float:
    return float(roc_auc_score(test_labels, mean_neighbor_score(train_data, test_data, k)))


def pca_auc(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    n_components: int,
) -> float:
    """Fit PCA on normal training data and score test reconstruction error."""
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    reconstructed = pca.inverse_transform(pca.transform(test_data))
    scores = np.linalg.norm(test_data - reconstructed, axis=1)
    return float(roc_auc_score(test_labels, scores))


def dft_features(data: np.ndarray, coefficient_count: int) -> np.ndarray:
    """Keep matching low-frequency magnitudes at both ends of the FFT."""
    magnitudes = np.abs(np.fft.fft(data, axis=1))
    half = coefficient_count // 2
    selected = np.zeros_like(magnitudes)
    if half:
        selected[:, :half] = magnitudes[:, :half]
        selected[:, -half:] = magnitudes[:, -half:]
    return selected


def dft_auc(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    coefficient_count: int,
    k: int,
) -> float:
    train_features = dft_features(train_data, coefficient_count)
    test_features = dft_features(test_data, coefficient_count)
    scores = mean_neighbor_score(train_features, test_features, k)
    return float(roc_auc_score(test_labels, scores))


def haar_features(data: np.ndarray) -> np.ndarray:
    """Return a multilevel Haar representation, padding to a power of two."""
    feature_count = data.shape[1]
    padded_count = int(2 ** np.ceil(np.log2(feature_count)))
    approximation = np.pad(data, ((0, 0), (0, padded_count - feature_count)))
    details: list[np.ndarray] = []

    while approximation.shape[1] > 1:
        even = approximation[:, 0::2]
        odd = approximation[:, 1::2]
        details.insert(0, (even - odd) / 2)
        approximation = (even + odd) / 2

    return np.hstack((approximation, *details))


def dwt_results(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    k_values: range,
) -> dict[str, dict[str, float]]:
    train_features = haar_features(train_data)
    test_features = haar_features(test_data)
    prefix_sizes = [2**level for level in range(int(np.log2(train_features.shape[1])) + 1)]
    results: dict[str, dict[str, float]] = {}
    for k in k_values:
        results[str(k)] = {}
        for size in prefix_sizes:
            scores = mean_neighbor_score(train_features[:, :size], test_features[:, :size], k)
            results[str(k)][str(size)] = float(roc_auc_score(test_labels, scores))
    return results


def load_dataset(data_root: Path, category: str) -> tuple[np.ndarray, np.ndarray]:
    category_dir = data_root / category
    train_path = category_dir / f"{category}_TRAIN.tsv"
    test_path = category_dir / f"{category}_TEST.tsv"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s): " + ", ".join(missing))
    train = pd.read_csv(train_path, sep="\t", header=None).to_numpy()
    test = pd.read_csv(test_path, sep="\t", header=None).to_numpy()
    return train, test


def run_experiments(data_root: Path, category: str, max_k: int) -> dict[str, object]:
    raw_train, raw_test = load_dataset(data_root, category)
    rng = np.random.default_rng(RANDOM_SEED)
    train_data, _ = resample(
        raw_train[:, 1:],
        raw_train[:, 0],
        outlier_ratio=0.0,
        normal_label=1,
        rng=rng,
    )
    test_data, test_labels = resample(
        raw_test[:, 1:],
        raw_test[:, 0],
        outlier_ratio=0.1,
        normal_label=1,
        rng=rng,
    )

    k_values = range(1, max_k + 1)
    if max_k > len(train_data):
        raise ValueError("max-k cannot exceed the number of normal training samples")

    knn = {str(k): knn_auc(train_data, test_data, test_labels, k) for k in k_values}

    component_limit = min(train_data.shape)
    pca_k5 = {
        str(n): pca_auc(train_data, test_data, test_labels, n)
        for n in tqdm(range(1, component_limit + 1), desc="PCA")
    }
    best_components = max(pca_k5, key=pca_k5.get)

    dft: dict[str, dict[str, float]] = {}
    for k in k_values:
        dft[str(k)] = {
            str(count): dft_auc(train_data, test_data, test_labels, count, k)
            for count in tqdm(
                range(2, train_data.shape[1] + 1), desc=f"DFT k={k}", leave=False
            )
        }

    return {
        "category": category,
        "seed": RANDOM_SEED,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "knn": knn,
        "pca_k5": pca_k5,
        "pca_best_components": int(best_components),
        "dft": dft,
        "dwt": dwt_results(train_data, test_data, test_labels, k_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", choices=("Wafer", "ECG200"), required=True)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiments(args.data_root, args.category, args.max_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
