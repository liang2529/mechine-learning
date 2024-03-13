import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # 降维
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score  # 轮廓系数

plt.switch_backend('TkAgg')
# 设置字体为楷体
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
# 获取数据
X, y = datasets.load_iris(return_X_y=1)
# 降维
pca = PCA(n_components=2)
X = pca.fit_transform(X)

scores = []
for k in range(2, 4):
    kmeans = KMeans(n_clusters=k)  # 建模
    kmeans.fit(X)
    y_ = kmeans.predict(X)  # 预测
    # y_ = kmeans.labels_
    score = silhouette_score(X, y_)  # 轮廓系数
    print(score)
    scores.append(score)
# 画图
plt.plot(range(2, 4), scores, color='green')
index = np.argmax(scores)
# 画点
plt.scatter(range(2, 4)[index], scores[index], color='red', s=50)
print('轮廓系数最大的K值是:', range(2, 4)[index])
plt.xlabel('K值')
plt.ylabel('轮廓系数', c='red')
plt.show()
