# %%
# 相关库和函数导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 图形化
import seaborn as sns  # 热度图
from IPython.display import display  # 载入数据查看时需要使用的函数
from sklearn import decomposition  # PCA


# 数据预处理
def GetData(csv_path):
    dt = pd.read_csv(
        csv_path, dtype={'status_id': str}, encoding='gbk',
        engine='python')  # 导入数据，注意编码使用gbk，然后含有中文字符的地址要设定"engine=python"

    # ## 数据清洗
    dt_new = dt.iloc[:, 1:-4]  # 去除数据集最后4列和第一列
    dt_new.drop(['status_published'],axis=1,inplace=True)
    # 如果有重复值则去重？
    # 貌似不应该去重，同天发布的同类型商品反应数量相同是有可能的
    #dt_new = dt_new.drop_duplicates()  # 去除重复值

    # 时间转时间戳
    # dt_new['status_published'] = pd.to_datetime(dt_new['status_published'],
    #                                             format='%m/%d/%Y %H:%M',
    #                                             errors='coerce')
    # dt_new['status_published'] = dt_new['status_published'].astype(
    #     'int64') // 1e9

    # 类别数据编码，添加四列
    pf = pd.get_dummies(dt_new['status_type'])
    #dt_new = pd.concat([dt_new, pf], axis=1)
    dt_new.drop(['status_type'], axis=1, inplace=True)
    return dt_new, pf


def GetNorData(csv_path):
    dt_new, pf = GetData(csv_path)
    #标准化
    cols = dt_new.columns
    df = pd.DataFrame()
    for col in cols:
        df['S_' + col] = (dt_new[col] - dt_new[col].mean()) / dt_new[col].std()
    return df, pf


def GetPCAData(csv_path, n_components):
    df, pf = GetNorData(csv_path)
    # PCA降维
    pca = decomposition.PCA(n_components=n_components)
    df = pca.fit_transform(df)  #ndarray
    print(pca.explained_variance_ratio_)
    df = np.concatenate((df, pf.values), axis=1)
    return df


if __name__ == "__main__":
    # 全局设定
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width',
                  True)  # 修改对齐设定，使得数据输出时更好看
    pd.set_option('display.max_columns', None)  # 设置数据显示时显示所有的列

    # %% [markdown]
    # # 数据导入
    csv_path = "./Live_20210128.csv"
    # %%
    dt = pd.read_csv(
        csv_path, dtype={'status_id': str}, encoding='gbk',
        engine='python')  # 导入数据，注意编码使用gbk，然后含有中文字符的地址要设定"engine=python"

    # %% [markdown]
    # # 数据查看

    # %%
    display(dt.describe())  # 对数据集进行描述统计分析

    # %%
    display(dt.dtypes)  # 查看数据集中各变量的类型

    # %% [markdown]
    display(dt.shape)  # 查看数据集的行数与列数

    # %%
    display(dt.columns)  # 查看数据集的列名

    # %%
    display(dt.iloc[0:5, 1:12])  # 查看数据集的前5行数据，注意下标从0开始

    # %% [markdown]
    # # 数据预处理

    # %% [markdown]
    # ## 数据清洗

    # %%
    dt_new = dt.iloc[:, 0:12]  # 去除数据集最后4列

    display(dt_new.isnull().sum())  # 检查数据集是否有缺失值

    # %%
    display(dt_new.duplicated().sum())  # 检查数据是否有重复值

    # %%
    # 如果有重复值则去重
    loc = pd.DataFrame(dt_new[dt_new.duplicated(
        keep=False) == True].index.tolist()).transpose(
        )  # 找到重复值的数据地址，并转置显示，设定keep=False用于输出所有重复值的位置（如果不加，只输出重复值最后出现的位置）
    display(loc)  # 获取重复值位置
    dt_new = dt_new.drop_duplicates()  # 去除重复值
    display(dt_new.duplicated().sum())  # 查看现在是否还有重复值

    display(dt_new.shape)  # 查看数据清洗后的数据维度

    # %%

    # 时间转时间戳
    dt_new['status_published'] = pd.to_datetime(dt_new['status_published'],
                                                format='%m/%d/%Y %H:%M',
                                                errors='coerce')
    dt_new['status_published'] = dt_new['status_published'].astype(
        'int64') // 1e9

    #%%

    # 各类别各num变量条形图
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_reactions')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_comments')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_shares')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_likes')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_loves')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_wows')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_hahas')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_sads')
    # dt_new.groupby('status_type').sum().plot(kind='bar', y='num_angrys')

    # 类别数据编码
    from sklearn.preprocessing import OneHotEncoder

    pf = pd.get_dummies(dt_new['status_type'])
    dt_new = pd.concat([dt_new, pf], axis=1)
    dt_new.drop(['status_type'], axis=1, inplace=True)

    # 关系矩阵热度图
    d = dt_new.iloc[:, 1:]
    corr = d.corr()
    print(corr)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap='Oranges')
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()

    display(dt_new.iloc[0:5])  # 查看数据集的前5行数据，注意下标从0开始

    # %% [markdown]
    # ## 特征选择

    # # %%
    #标准化
    cols = dt_new.columns

    df = pd.DataFrame()
    # df.insert(0, 'status_id', dt_new['status_id'].values)
    for col in cols:
        if col == 'status_id':
            continue
        df['S_' + col] = (dt_new[col] - dt_new[col].mean()) / dt_new[col].std()
    display(df[0:5])  # 查看数据集的前5行数据

    # PCA降维
    pca = decomposition.PCA(n_components=8)
    df = pca.fit_transform(df)  #ndarray
    display(df[0:5])  # 查看数据集的前5行数据

    #降维后热度图
    df = pd.DataFrame(data=df[0:, 0:], columns=range(df.shape[1]))
    corr = df.corr()
    print(corr)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap='Oranges')
    b, t = plt.ylim()
    plt.ylim(b + 0.5, t - 0.5)
    plt.title("Feature Correlation Heatmap")
    plt.show()
