# 导包
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer  # (美国威斯康辛州乳腺癌数据集)
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=1)
# 标准化
stand = StandardScaler()
X = stand.fit_transform(X)
# 划分数据
X_train, X_text, y_train, y_text = train_test_split(X, y, test_size=0.2)
# 不使用科学计数法
np.set_printoptions(suppress=True)
# 建模
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_text)
print('真实值:', y_text)
print('预测值:', y_pred)
print('算法预测准确率:', model.score(X_text, y_text))

# 概率
probe_ = model.predict_proba(X_text)
print('预测概率:', probe_[:5])
