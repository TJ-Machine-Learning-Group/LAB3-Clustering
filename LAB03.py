from random import randint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from ClusterValidityIndex.CVI import Eval
from DataPrepare.dataAnalyzeandPrepare import GetData, GetNorData, GetPCAData


def Predict(models: dict, data: np.ndarray) -> dict:
    labels = dict()
    #Kmeans: core and distance
    labels['kmeans'] = models['kmeans'].fit_predict(data)
    #DBSCN: near neighbor
    labels['dbscan'] = models['dbscan'].fit_predict(data)
    return labels


if __name__ == "__main__":
    random_state = randint(1, 100)  #给定一个随机种子（划分数据集、确定初始点等）
    models = dict()
    #初始化模型，要调参可以在这里调
    models['kmeans'] = KMeans(n_clusters=10, random_state=random_state)
    models['dbscan'] = DBSCAN(eps=0.5, min_samples=10)
    data = GetPCAData("./Live_20210128.csv")
    labels = Predict(models, data)
    print(labels)
    for model_name, res_label in labels.items():
        print(model_name)
        Eval(data, res_label)
