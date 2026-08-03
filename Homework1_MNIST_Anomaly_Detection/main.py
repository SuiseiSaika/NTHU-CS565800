# -*- coding: utf-8 -*-
########################################################
########  Do not modify the sample code segment ########
########################################################

from pathlib import Path

import torchvision
import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)

def resample_total(data,label,ratio=0.05):
    """
        data: np.array, shape=(n_samples, n_features)
        label: np.array, shape=(n_samples,)
        ratio: float, ratio of samples to be selected
    """
    new_data = []
    new_label = []
    for i in range(10):
        i_data = data[label==i]
        idx = np.random.choice(list(range(len(i_data))),int(len(i_data)*ratio),replace=False)
        new_data.append(i_data[idx])
        new_label.append(np.ones(len(idx))*i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label

def resample(data,label,outlier_ratio=0.01,target_label=0):
    """
        data: np.array, shape=(n_samples, n_features)
        label: np.array, shape=(n_samples,)
        outlier_ratio: float, ratio of outliers
        target_label: int, the label to be treated as normal
    """
    new_data = []
    new_label = []
    for i in range(10):
        if i != target_label:
            i_data = data[label==i]
            target_size = len(data[label==target_label])
            num = target_size*((outlier_ratio/9))
            idx = np.random.choice(list(range(len(i_data))),int(num),replace=False)
            new_data.append(i_data[idx])
            new_label.append(np.ones(len(idx))*i)
        else:
            new_data.append(data[label==i])
            new_label.append(np.ones(len(data[label==i]))*i)
    new_data = np.concatenate(new_data)
    new_label = np.concatenate(new_label)
    return new_data, new_label

def KNN(train_data, train_label, test_data, anomaly_label, k, record):
  # 計算每筆test_data與每筆train_data的Euclidean Distance
  distances = pairwise_distances(test_data, train_data, metric='euclidean')
  # 透過np.sort()升序排列距離後取得前k筆最近的距離，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[:k]) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(anomaly_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, 0)
  pos += auc_score/10
  record[k] = pos

def Kmeans(train_data, train_label, test_data, anomaly_label, k, record):
  # 初始化centroid
  centroid = train_data[np.random.choice(list(range(len(train_data))),int(k),replace=False)]
  # 初始化k個cluster
  clusters = [[] for _ in range(k)]
  # 計算每筆train_data與每個centroid的Euclidean Distance
  distances = pairwise_distances(train_data, centroid, metric='euclidean')
  # 透過每筆np.argmin()將每筆train_data分類到各個cluster
  cluster_list = [np.argmin(distance) for distance in distances]
  # 將train_data分配到各個cluster
  for idx, c in enumerate(cluster_list):
    clusters[c].append(train_data[idx])
  # 重新計算每個cluster的centroid
  centroid = np.array([np.mean(x, axis=0) for x in clusters])
  # 開始迭代
  while True:
    # 初始化k個cluster
    clusters = [[] for _ in range(k)]
    # 計算每筆train_data與每個centroid的Euclidean Distanc
    distances = pairwise_distances(train_data, centroid, metric='euclidean')
    # 透過每筆np.argmin()將每筆train_data分類到各個cluster
    temp = [np.argmin(distance) for distance in distances]
    # 如果分群結果沒有變動則停止迭代
    if cluster_list == temp:
      break
    # 更新分群結果
    cluster_list = temp
    # 將train_data分配到各個cluster
    for idx, c in enumerate(cluster_list):
      clusters[c].append(train_data[idx])
    # 重新計算每個cluster的centroid
    centroid = np.array([np.mean(x, axis=0) for x in clusters])
  # 計算每筆test_data與每個centroid的Euclidean Distance
  distances = pairwise_distances(test_data, centroid, metric='euclidean')
  # 計算anomaly score
  anomaly_scores = [np.mean(distance) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(anomaly_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, 0)
  pos += auc_score/10
  record[k] = pos

def cosine_distance_based(test_data, anomaly_label, record, k=5):
  # 根據公式定義cosine distance
  def cosine_distance(X, Y):
    return 1-(np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y)))
  # 計算每筆test_data間的cosine distance
  distances = pairwise_distances(test_data, test_data, metric=cosine_distance)
  # 透過np.sort()升序排列距離後取得前k筆最近的距離(去掉與自己的距離)，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[1:k+1]) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(anomaly_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, 0)
  pos += auc_score/10
  record[k] = pos

def Minkowski_distance_based(test_data, anomaly_label, record, r, k=5):
  # 根據公式定義Minkowski distance
  def Minkowski_1(X, Y):
    return np.power(sum(np.power(np.abs(X-Y), 1)), 1)
  def Minkowski_2(X, Y):
    return np.power(sum(np.power(np.abs(X-Y), 2)), 1/2)
  def Minkowski_inf(X, Y):
    return np.max(np.abs(X-Y))
  # 計算每筆test_data間的distance
  if r=='1':
    distances = pairwise_distances(test_data, test_data, metric=Minkowski_1)
  if r=='2':
    distances = pairwise_distances(test_data, test_data, metric=Minkowski_2)
  if r=='inf':
    distances = pairwise_distances(test_data, test_data, metric=Minkowski_inf)

  # 透過np.sort()升序排列距離後取得前k筆最近的距離(去掉與自己的距離)，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[1:k+1]) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(anomaly_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, 0)
  pos += auc_score/10
  record[k] = pos

def Mahalanobis_distance_based(train_data, train_label, test_data, anomaly_label, record, k=5):
  # 計算Covariance Matrix
  Covariance = np.cov(np.transpose(train_data))
  inv_covmat = np.linalg.inv(Covariance)
  # 根據公式定義Mahalanobis distance
  def diff(X, Y):
    return np.dot(np.dot(np.transpose(X-Y), inv_covmat), np.transpose(X-Y))
  # 計算每筆test_data間的distance
  distances = pairwise_distances(test_data, test_data, metric=diff)
  # 透過np.sort()升序排列距離後取得前k筆最近的距離(去掉與自己的距離)，透過np.mean()取平均後得到anomaly score
  anomaly_scores = [np.mean(np.sort(distance)[1:k+1]) for distance in distances]
  # 透過anomaly score計算auc_score
  auc_score = roc_auc_score(anomaly_label, anomaly_scores)
  # 紀錄auc_score
  pos = record.get(k, 0)
  pos += auc_score/10
  record[k] = pos

def LOF_detection(test_data, anomaly_label, record, cmap='copper', plot=False, k=5):
  # Reachability distance
  N=test_data.shape[0]
  distance_matrix = pairwise_distances(test_data, test_data, metric='euclidean')
  k_distance=np.sort(distance_matrix,axis=0)[k+1]
  k_distance_matrix=np.outer(np.ones(N),k_distance)
  reach_distacne=np.maximum(distance_matrix,k_distance_matrix)
  # local reachability density
  sort_index=np.argsort(distance_matrix,axis=1)[:,1:k+1]
  IRD=np.zeros(N)
  for i in range(N):
    IRD[i]=1/np.mean(reach_distacne[i,sort_index[i]])
  # LOF
  LOF=np.zeros(N)
  for i in range(N):
    LOF[i]=np.mean(IRD[sort_index[i]])/IRD[i]

  auc_score = roc_auc_score(anomaly_label, LOF)
  pos = record.get(k, 0)
  pos += auc_score/10
  record[k] = pos
  if plot:
    tsne = TSNE(2, random_state=0)
    X_2d = tsne.fit_transform(test_data)
    fig, axs = plt.subplots(1,2,figsize=(20,9))
    points = axs[0].scatter(X_2d[:,0], X_2d[:,1], c=LOF, s=10, cmap='Blues')
    axs[0].title.set_text("Predicted LOF Scores for Normal Digit 0")
    fig.colorbar(points)
    axs[1].scatter(X_2d[:,0], X_2d[:,1], c=anomaly_label, s=10, cmap=cmap)
    axs[1].title.set_text("Ground Truth Labels for Normal Digit 0")
    plt.show()

if __name__=="__main__":
    data_root = Path(__file__).resolve().parent / "data"
    orig_train_data = torchvision.datasets.MNIST(str(data_root), train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),target_transform=None,download=True) #下載並匯入MNIST訓練資料
    orig_test_data = torchvision.datasets.MNIST(str(data_root), train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),target_transform=None,download=True) #下載並匯入MNIST測試資料

    orig_train_label = orig_train_data.targets.numpy()
    orig_train_data = orig_train_data.data.numpy()
    orig_train_data = orig_train_data.reshape(60000,28*28)

    orig_test_label = orig_test_data.targets.numpy()
    orig_test_data = orig_test_data.data.numpy()
    orig_test_data = orig_test_data.reshape(10000,28*28)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30)
    pca_data = pca.fit_transform(np.concatenate([orig_train_data,orig_test_data]))
    orig_train_data = pca_data[:len(orig_train_label)]
    orig_test_data = pca_data[len(orig_train_label):]

    orig_train_data,orig_train_label = resample_total(orig_train_data,orig_train_label,ratio=0.1)

    auc_scores = {
        'KNN':{},
        'Cluster':{},
        'Cosine':{},
        'Minkowski':{
            'l1':{},
            'l2':{},
            'inf':{},
        },
        'Mahalanobis':{},
        'Density':{},
    }

    for i in tqdm.tqdm(range(10)):
      train_data = orig_train_data[orig_train_label==i]
      test_data,test_label = resample(orig_test_data,orig_test_label,target_label=i,outlier_ratio=0.1)
      train_label = np.zeros(len(train_data))
      anomaly_label = np.zeros(len(test_data))
      anomaly_label[test_label != i] = 1
      KNN(train_data, train_label, test_data, anomaly_label, 1, auc_scores['KNN'])
      KNN(train_data, train_label, test_data, anomaly_label, 5, auc_scores['KNN'])
      KNN(train_data, train_label, test_data, anomaly_label, 10, auc_scores['KNN'])
      Kmeans(train_data, train_label, test_data, anomaly_label, 1, auc_scores['Cluster'])
      Kmeans(train_data, train_label, test_data, anomaly_label, 5, auc_scores['Cluster'])
      Kmeans(train_data, train_label, test_data, anomaly_label, 10, auc_scores['Cluster'])
      cosine_distance_based(test_data, anomaly_label, auc_scores['Cosine'], k=1)
      cosine_distance_based(test_data, anomaly_label, auc_scores['Cosine'], k=5)
      cosine_distance_based(test_data, anomaly_label, auc_scores['Cosine'], k=10)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['l1'], r='1', k=1)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['l1'], r='1', k=5)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['l1'], r='1', k=10)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['l2'], r='2', k=1)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['l2'], r='2', k=5)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['l2'], r='2', k=10)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['inf'], r='inf', k=1)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['inf'], r='inf', k=5)
      Minkowski_distance_based(test_data, anomaly_label, auc_scores['Minkowski']['inf'], r='inf', k=10)
      Mahalanobis_distance_based(train_data, train_label, test_data, anomaly_label, auc_scores['Mahalanobis'], k=1)
      Mahalanobis_distance_based(train_data, train_label, test_data, anomaly_label, auc_scores['Mahalanobis'], k=5)
      Mahalanobis_distance_based(train_data, train_label, test_data, anomaly_label, auc_scores['Mahalanobis'], k=10)
      LOF_detection(test_data, anomaly_label, auc_scores['Density'], cmap='Dark2', k=1)
      LOF_detection(test_data, anomaly_label, auc_scores['Density'], plot=(i==0), cmap='Dark2', k=5)
      LOF_detection(test_data, anomaly_label, auc_scores['Density'], cmap='Dark2', k=10)

    for k in auc_scores:
      print(k, auc_scores[k])
