# -*- coding: utf-8 -*-

########################################################
########  Do not modify the sample code segment ########
########################################################

import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, pairwise_distances

np.random.seed(0)

def resample(data, label, outlier_ratio=0.01, target_label=0):
    """
    Resample the data to balance classes.

    Parameters:
        data: np.array, shape=(n_samples, n_features)
            Input data.
        label: np.array, shape=(n_samples,)
            Labels corresponding to the data samples.
        outlier_ratio: float, optional (default=0.01)
            Ratio of outliers to include in the resampled data.
        target_label: int, optional (default=0)
            The label to be treated as normal.

    Returns:
        new_data: np.array
            Resampled data.
        new_label: np.array
            Resampled labels.
    """
    new_data = []
    new_label = []
    for i in [1, -1]:
        if i != target_label:
            i_data = data[label == i]
            target_size = len(data[label == target_label])
            num = target_size * outlier_ratio
            idx = np.random.choice(
                list(range(len(i_data))), int(num), replace=False
            )
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx)) * 1)
        else:
            new_data.append(data[label == i])
            new_label.append(np.ones(len(data[label == i])) * 0)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label

from google.colab import drive, files
drive.mount('/content/gdrive', force_remount=True)

if __name__=='__main__':
    # Load the data
    category = "Wafer" # Wafer / ECG200
    # category = "ECG200" # Wafer / ECG200
    print(f"Dataset: {category}")

    # train_data = pd.read_csv(f'./{category}/{category}_TRAIN.tsv', sep='\t', header=None).to_numpy()
    # test_data = pd.read_csv(f'./{category}/{category}_TEST.tsv', sep='\t', header=None).to_numpy()
    train_data = pd.read_csv(f'/content/gdrive/MyDrive/Colab Notebooks/ADHW/HW2/HW2/{category}/{category}_TRAIN.tsv', sep='\t', header=None).to_numpy()
    test_data = pd.read_csv(f'/content/gdrive/MyDrive/Colab Notebooks/ADHW/HW2/HW2/{category}/{category}_TEST.tsv', sep='\t', header=None).to_numpy()

    print(train_data.shape, test_data.shape)
    # # Q1 Visualization
    # normal_data = train_data[train_data[:, 0]==1]
    # normal_data = normal_data[np.random.choice(normal_data.shape[0], 10, replace=False), :][:, 1:]

    # abnormal_data = train_data[train_data[:, 0]==-1]
    # abnormal_data = abnormal_data[np.random.choice(normal_data.shape[0], 10, replace=False), :][:, 1:]
    # fig, axs = plt.subplots(2,1,figsize=(10,8))
    # for data in normal_data:
    #   axs[0].plot(np.arange(data.shape[0]), data, c='b')
    # for data in abnormal_data:
    #   axs[1].plot(np.arange(data.shape[0]), data, c='r')
    # axs[0].title.set_text("Normal Sample")
    # axs[1].title.set_text("Abnormal Sample")
    # plt.show()

    # Q3 Visualization
    # normal_data = train_data[train_data[:, 0]==1]
    # normal_data = normal_data[np.random.choice(normal_data.shape[0], 10, replace=False), :][:, 1:]

    # abnormal_data = train_data[train_data[:, 0]==-1]
    # abnormal_data = abnormal_data[np.random.choice(normal_data.shape[0], 10, replace=False), :][:, 1:]

    # print(normal_data.shape)
    # pca = PCA(5)
    # proj_normal = pca.inverse_transform(pca.fit_transform(normal_data))
    # proj_abnormal = pca.inverse_transform(pca.fit_transform(abnormal_data))

    # # print(normal_data.shape)
    # fig, axs = plt.subplots(2,1,figsize=(10,8))
    # for data in proj_normal:
    #   axs[0].plot(np.arange(data.shape[0]), data, c='b')
    # for data in proj_abnormal:
    #   axs[1].plot(np.arange(data.shape[0]), data, c='r')
    # axs[0].title.set_text("Normal Sample")
    # axs[1].title.set_text("Abnormal Sample")
    # plt.show()


    # # Q4 Visualization
    # M = 38
    # m = M//2
    # normal_data = train_data[train_data[:, 0]==1]
    # normal_data = normal_data[np.random.choice(normal_data.shape[0], 10, replace=False), :][:, 1:]

    # abnormal_data = train_data[train_data[:, 0]==-1]
    # abnormal_data = abnormal_data[np.random.choice(normal_data.shape[0], 10, replace=False), :][:, 1:]

    # normal_dft_coefficients = np.fft.fft(normal_data)
    # abnormal_dft_coefficients = np.fft.fft(abnormal_data, axis=1)

    # normal_sorted_indices = np.argsort(np.abs(normal_dft_coefficients), axis=1)
    # normal_selected_indices = normal_sorted_indices[:, :M]
    # abnormal_sorted_indices = np.argsort(np.abs(abnormal_dft_coefficients), axis=1)
    # abnormal_selected_indices = abnormal_sorted_indices[:, :M]

    # normal_selected_coefficients = np.zeros_like(normal_dft_coefficients)
    # abnormal_selected_coefficients = np.zeros_like(abnormal_dft_coefficients)

    # normal_selected_coefficients[:, :m] = normal_dft_coefficients[:, :m]
    # normal_selected_coefficients[:,-m:] = normal_dft_coefficients[:,-m:]
    # abnormal_selected_coefficients[:, :m] = abnormal_dft_coefficients[:, :m]
    # abnormal_selected_coefficients[:,-m:] = abnormal_dft_coefficients[:,-m:]

    # inversed_normal = np.fft.ifft(normal_selected_coefficients)
    # inversed_abnormal = np.fft.ifft(abnormal_selected_coefficients)

    # # print(inversed_normal.shape, inversed_abnormal.shape)

    # fig, axs = plt.subplots(2,1,figsize=(10,8))
    # for data in inversed_normal:
    #   axs[0].plot(np.arange(data.shape[0]), data, c='b')
    # for data in inversed_abnormal:
    #   axs[1].plot(np.arange(data.shape[0]), data, c='r')
    # axs[0].title.set_text("Normal Sample")
    # axs[1].title.set_text("Abnormal Sample")
    # plt.show()


    train_label = train_data[:, 0].flatten()
    train_data = train_data[:, 1:]
    train_data, train_label = resample(train_data, train_label, outlier_ratio=0.0, target_label=1)

    test_label = test_data[:, 0].flatten()
    test_data = test_data[:, 1:]
    test_data, test_label = resample(test_data, test_label, outlier_ratio=0.1, target_label=1)

# Q2 KNN
def KNN(train_data, train_label, test_data, test_label, k, record):
  # 計算每筆test_data與每筆train_data的Euclidean Distance
  distances = pairwise_distances(test_data, train_data, metric='euclidean')
  # 透過np.sort()升序排列距離後取得前k筆最近的距離，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[:k]) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(test_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, 0)
  pos += auc_score
  record[k] = pos

# Q3 PCA
def PCA_reconstruction(train_data, train_label, test_data, test_label, n_components, k, record):
  # 設定PCA
  pca = PCA(n_components)
  # Fit the model with train_data
  pca.fit(train_data)
  # Apply the dimensionality reduction on test_data and reconstruct it.
  proj = pca.inverse_transform(pca.fit_transform(test_data))
  # 計算每筆test_data與每筆reconstruction_data的Euclidean Distance
  distances = pairwise_distances(test_data, proj, metric='euclidean')
  # 透過np.sort()升序排列距離後取得前k筆最近的距離，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[:k]) for distance in distances]
  # # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(test_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, {})
  pos[n_components] = auc_score
  record[k] = pos

# Q4 DFT
def DFT(train_data, train_label, test_data, test_label, M, k, record):
  # DFT 轉換
  m = M//2
  train_dft_coefficients = np.abs(np.fft.fft(train_data, axis=1))
  test_dft_coefficients = np.abs(np.fft.fft(test_data, axis=1))

  # 創建補零後的 DFT 系數矩陣
  train_selected_coefficients = np.zeros_like(train_dft_coefficients)
  test_selected_coefficients = np.zeros_like(test_dft_coefficients)

  train_selected_coefficients[:, :m] = train_dft_coefficients[:, :m]
  train_selected_coefficients[:,-m:] = train_dft_coefficients[:,-m:]
  test_selected_coefficients[:, :m] = test_dft_coefficients[:, :m]
  test_selected_coefficients[:,-m:] = test_dft_coefficients[:,-m:]
  # 計算每筆test_data與每筆train_data的Euclidean Distance
  distances = pairwise_distances(test_selected_coefficients, train_selected_coefficients, metric='euclidean')
  # 透過np.sort()升序排列距離後取得前k筆最近的距離，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[:k]) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(test_label, anomaly_scores)

  # print(auc_score)
  # 紀錄auc_score
  pos = record.get(k, {})
  pos[M] = auc_score
  record[k] = pos

# Q5 DWT
def DWT(train_data, train_label, test_data, test_label, k, record):
  def haar_wavelet_transform(data):
    even_indices = np.arange(0, data.shape[1], 2)
    odd_indices = np.arange(1, data.shape[1], 2)
    a = (data[:, even_indices] + data[:, odd_indices]) / 2
    d = (data[:, even_indices] - data[:, odd_indices]) / 2

    return a, d
  def discrete_wavelet_transform(data):
    # print(data.shape)
    feature_dim = data.shape[1]
    padded_dim = int(2 ** np.ceil(np.log2(feature_dim)))

    if padded_dim > feature_dim:
      padded_data = np.pad(data, ((0, 0), (0, padded_dim - feature_dim)), mode='constant')
    # print(padded_data.shape)

    levels = range(1, int(np.ceil(np.log2(padded_dim))) + 1)

    a = padded_data
    transformed = np.zeros((data.shape[0], 0))
    for i in levels:
      a, d = haar_wavelet_transform(a)
      transformed = np.hstack((d, transformed))
      # print(2 ** (levels[-1] - i), a.shape, transformed.shape)

    transformed = np.hstack((a, transformed))
    # print(transformed.shape)
    return transformed
  feature_dim = train_data.shape[1]
  padded_dim = int(2 ** np.ceil(np.log2(feature_dim)))
  levels = range(0, int(np.ceil(np.log2(padded_dim))) + 1)
  transformed_train_data = discrete_wavelet_transform(train_data)
  transformed_test_data = discrete_wavelet_transform(test_data)
  for level in levels:
    S = 2 ** (level)
    distances = pairwise_distances(transformed_test_data[:, :S], transformed_train_data[:, :S], metric='euclidean')
    anomaly_scores = [np.mean(np.sort(distance)[:k]) for distance in distances]
    auc_score = roc_auc_score(test_label, anomaly_scores)
    pos = record.get(k, {})
    pos[S] = auc_score
    record[k] = pos

record = {
    'KNN':{},
    'PCA':{},
    'DFT':{},
    'DWT':{},
}

k_range = range(1, 11)

for k in k_range:
  KNN(train_data, train_label, test_data, test_label, k, record['KNN'])

for n in tqdm.tqdm(range(1, min(train_data.shape[0], train_data.shape[1])+1)):
  PCA_reconstruction(train_data, train_label, test_data, test_label, n, 5, record['PCA'])
best_n = max(record['PCA'][5], key=record['PCA'][5].get)
for k in k_range:
  PCA_reconstruction(train_data, train_label, test_data, test_label, best_n, k, record['PCA'])

for k in k_range:
  for m in tqdm.tqdm(range(1, train_data.shape[1]+1)):
    DFT(train_data, train_label, test_data, test_label, m, k, record['DFT'])

for k in k_range:
  DWT(train_data, train_label, test_data, test_label, k, record['DWT'])

print(record['KNN'])
print(record['KNN'].keys())
plt.plot(record['KNN'].keys(), record['KNN'].values(), )
plt.title('KNN Performance for '+category+' Dataset.')

# print(record['PCA'].keys())
# for k in record['PCA'].keys():
#   # print(record['PCA'][k].keys())
#   print(k, max(record['PCA'][k], key=record['PCA'][k].get), record['PCA'][k][max(record['PCA'][k], key=record['PCA'][k].get)])

#   plt.plot(record['PCA'][k].keys(), record['PCA'][k].values(), label=str(k))
# plt.title('PCA Performance for '+category+' Dataset.')
# plt.legend()

plt.plot(record['PCA'][5].keys(), record['PCA'][5].values(), label=str(5))
plt.title('PCA Performance for '+category+' Dataset.(k=5)')

print(record['DFT'].keys())
for k in record['DFT'].keys():
  print(k, max(record['DFT'][k], key=record['DFT'][k].get), record['DFT'][k][max(record['DFT'][k], key=record['DFT'][k].get)])
  plt.plot(record['DFT'][k].keys(), record['DFT'][k].values(), label=str(k))
plt.title('DFT Performance for '+category+' Dataset.')
plt.legend()

plt.plot(record['DFT'][5].keys(), record['DFT'][5].values(), label=str(5))
plt.title('DFT Performance for '+category+' Dataset.(k=5)')

print(record['DWT'].keys())
for k in record['DWT'].keys():
  # print(record['DWT'][k].keys())
  print(k, max(record['DWT'][k], key=record['DWT'][k].get), record['DWT'][k][max(record['DWT'][k], key=record['DWT'][k].get)])
  plt.plot(record['DWT'][k].keys(), record['DWT'][k].values(), label=str(k))
plt.title('DWT Performance for '+category+' Dataset.')
plt.legend()

plt.plot(record['DWT'][5].keys(), record['DWT'][5].values(), label=str(5))
plt.title('DWT Performance for '+category+' Dataset.(k=5)')
