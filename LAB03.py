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
    data = GetPCAData(csv_path=csv_path, n_components=2)
    labels = Predict(models, data)  #data是(m,8)的数组，默认使用欧氏距离（要用其它距离可以先算出来
    #rint(labels)

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8, 6))
    colors = ["#FF0000","#00FF00","#0000FF","#FFFF00","#00FFFF","#FF00FF","#FFF000","#00FFF0","#F000FF","#F00000","#00F000","#0000F0","#000000"]
    for i,j in enumerate(labels.items()):
        model_name, res_label = j
        print(model_name)
        se = set(res_label)
        used_colors = list()
        #print(se)
        #ax = plt.subplot(projection = '3d')
        for k in se:
            pdata = data[res_label==k]
            color = colors[k]
            ax[i].set_title(model_name)
            ax[i].scatter(pdata[:,0],pdata[:,1],s=5,color=color)
            used_colors.append(k)
        #KMeans确实是n_clusters类,DBSCAN得看参数
        ax[i].legend(used_colors)
        Eval(data, res_label)
    plt.savefig("test.jpg")
    plt.close()
