# 导入相关库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib

plt.switch_backend('TkAgg')
# 设置字体为楷体
matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
# 加载鸢尾花数据集
iris = load_iris()

# 将数据集转换为DataFrame格式，方便进行数据预处理
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# 对数据进行标准化处理（归一化）
scaler = StandardScaler()
iris_norm = scaler.fit_transform(iris_df)

# 进行主成分分析(PCA)，将4维降到2维
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(iris_norm)

# 将降维后的数据构造成DataFrame格式，方便进行可视化
iris_pca_df = pd.DataFrame(data=iris_pca, columns=['PC1', 'PC2'])

# 将降维前后的数据进行可视化
plt.figure(figsize=(8, 6))
plt.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], alpha=0.8, label='original')
plt.scatter(iris_pca_df['PC1'], iris_pca_df['PC2'], alpha=0.8, label='PCA')
plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Iris Data')
plt.show()