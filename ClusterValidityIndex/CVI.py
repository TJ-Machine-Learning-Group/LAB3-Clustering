import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, davies_bouldin_score
import time


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di


# Dunn Validity Index
def DVI(X, labels, n_clusters):
    # store the K-means results in a dataframe
    pred = pd.DataFrame(labels)
    pred.columns = ['Type']

    # merge this dataframe with X
    X_ = pd.DataFrame(X)
    prediction = pd.concat([X_, pred], axis=1)

    # store the clusters
    cluster_list = []
    for i in n_clusters:
        cluster_list.append(prediction.loc[prediction.Type == i])
    return dunn(cluster_list)


# 总评估函数
def Eval(X, labels):
    # 轮廓系数
    Silhouette = metrics.silhouette_score(X, labels, metric='euclidean')
    print("Silhouette = ", Silhouette)
    # Calinski-Harabaz Index
    print("CH = ", metrics.calinski_harabasz_score(X, labels))
    # 分类适确性指标
    print("DBI = ", davies_bouldin_score(X, labels))
    # 邓恩指数, 计算耗时较长
    # print("DVI = ", DVI(X, labels))


# 测试评估函数
if __name__ == "__main__":
    # test data
    X1, y1 = datasets.make_circles(n_samples=5000, factor=0.6, noise=0.05)
    X2, y2 = datasets.make_blobs(n_samples=1000,
                                 n_features=2,
                                 centers=[[1.2, 1.2]],
                                 cluster_std=[[.1]],
                                 random_state=9)
    X = np.concatenate((X1, X2))

    start_time = time.time()

    #Kmeans: core and distance
    n_clusters = 3
    y_pred = KMeans(n_clusters, random_state=9).fit(X)
    labels = y_pred.labels_

    print("run time = %.2f ms" % ((time.time() - start_time) * 1000))
    print("SSE = ", y_pred.inertia_ / n_clusters)  # SSE
    Eval(X, labels)