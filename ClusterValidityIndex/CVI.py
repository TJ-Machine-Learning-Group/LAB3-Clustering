import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import time
from typing import Tuple


def Silhouette(X, labels):
    return metrics.silhouette_score(X, labels, metric='euclidean')


def CH(X, labels):
    return metrics.calinski_harabasz_score(X, labels)


def DBI(X, labels):
    return metrics.davies_bouldin_score(X, labels)

def sigmoid(x):
    return 1/1+np.exp(-x)

# 总评估函数
def Eval(X: np.ndarray, labels: np.ndarray) -> Tuple[int,int,float,float,float]:
    n_clusters_ = len(set(labels)) - (-1 in labels)
    n_noise_ = (labels==-1).sum()

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)

    #没有labels_true 没法用这些评价指标
    #print("Homogeneity: %0.3f" %
    #      metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" %
    #      metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    #print("Adjusted Rand Index: %0.3f" %
    #      metrics.adjusted_rand_score(labels_true, labels))
    #print("Adjusted Mutual Information: %0.3f" %
    #      metrics.adjusted_mutual_info_score(labels_true, labels))

    # 轮廓系数
    # sc,ch,dbi = Silhouette(X,labels),CH(X,labels),DBI(X,labels)
    # print("Silhouette Coefficient: %0.3f" %
    sc = metrics.silhouette_score(X, labels)#)
    # Calinski-Harabaz Index
    #print("Calinski-Harabaz Index:%.3f" %
    ch = metrics.calinski_harabasz_score(X, labels)#)
    # 分类适确性指标
    #print("Davies-Bouldin Index:%.3f" %
    dbi = metrics.davies_bouldin_score(X, labels)#)
    fin = (sigmoid(sc)+ sigmoid(ch)+1-sigmoid(dbi))/3
    return [n_clusters_,n_noise_,sc,ch,dbi,fin]

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