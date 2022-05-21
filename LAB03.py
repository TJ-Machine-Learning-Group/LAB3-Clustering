from random import randint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from ClusterValidityIndex.CVI import Eval, DBI, CH, Silhouette
from DataPrepare.dataAnalyzeandPrepare import GetData, GetNorData, GetPCAData


def Predict(models: dict, data: np.ndarray) -> dict:
    labels = dict()
    #Kmeans: core and distance
    labels['kmeans'] = models['kmeans'].fit_predict(data)
    #DBSCN: near neighbor
    labels['dbscan'] = models['dbscan'].fit_predict(data)
    return labels


import sklearn.metrics
if __name__ == "__main__":
    models = dict()
    #初始化模型，要调参可以在这里调
    random_state = randint(1, 100)  #给定一个随机种子（划分数据集、确定初始点等）
    n_clusters = 12
    models['kmeans'] = KMeans(n_clusters=n_clusters, random_state=random_state)

    eps = 0.5
    min_samples = 10
    models['dbscan'] = DBSCAN(eps=eps, min_samples=min_samples)

    #标准化+PCA降维 保留8个分量
    csv_path = "./Live_20210128.csv"
    data = GetPCAData(csv_path=csv_path, n_components=8)

    labels = Predict(models, data)  #data是(m,8)的数组，默认使用欧氏距离（要用其它距离可以先算出来
    print(labels)

    for model_name, res_label in labels.items():
        print(model_name)
        #KMeans确实是n_clusters类,DBSCAN得看参数
        Eval(data, res_label)
