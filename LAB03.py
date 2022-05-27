from random import randint
import numpy as np
from typing import Dict,Union
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from ClusterValidityIndex.CVI import Eval, DBI, CH, Silhouette
from DataPrepare.dataAnalyzeandPrepare import GetData, GetNorData, GetPCAData
from Visual.visualization import Draw,make_xlsx_header,make_xlsx_body
from openpyxl.workbook import Workbook

def Predict(models: Dict[str,Union[KMeans,DBSCAN]], data: np.ndarray) -> dict:
    labels = dict()
    #Kmeans: core and distance
    # labels['kmeans'] = models['kmeans'].fit_predict(data)
    # #DBSCN: near neighbor
    # labels['dbscan'] = models['dbscan'].fit_predict(data)
    for i in models:
        labels[i] = models[i].fit_predict(data)
    return labels



if __name__ == "__main__":
    models = dict()
    #初始化模型，要调参可以在这里调
    random_state = randint(1, 100)  #给定一个随机种子（划分数据集、确定初始点等）
    #n_clusters = 12
    for n_clusters in range(4,13):
        models[f'kmeans-{n_clusters}'] = KMeans(n_clusters=n_clusters, random_state=random_state)

    eps = 0.5
    min_samples = 10
    models['dbscan'] = DBSCAN(eps=eps, min_samples=min_samples)

    #标准化+PCA降维 保留2个分量
    csv_path = "./Live_20210128.csv"
    data = GetPCAData(csv_path=csv_path, n_components=2)
    labels = Predict(models, data)  #data是(m,6)的数组，默认使用欧氏距离（要用其它距离可以先算出来
    Draw(data,labels,"test.jpg")
    wb = Workbook()
    ws = wb.active
    indexs = ["clusters","noise points","SC","CHI","DBI","all"]
    column = list(range(2,9))
    make_xlsx_header(ws,labels,indexs,column)
    for j in range(2,9):
        data=GetPCAData(csv_path=csv_path,n_components=j)
        for i,k in enumerate(labels.items()):
            model_name, res_label = k
            eval_data = Eval(data, res_label)
            for s in range(len(eval_data)):
                if not isinstance(eval_data[s],int):
                    value = "{:.2f}".format(eval_data[s])
                else:
                    value = eval_data[s]
                make_xlsx_body(ws,i*len(indexs)+2+s,j+1,value)
    wb.save('test.xlsx')
