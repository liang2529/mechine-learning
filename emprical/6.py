import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA  # 降维

# 获取数据
X, y = datasets.load_iris(return_X_y=1)
# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(X[:5])
print(X_pca[:5])
