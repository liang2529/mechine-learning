# 导包
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston #波士顿房价
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取数据
X, y = load_boston(return_X_y=1)
# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 建模
model_1 = LinearRegression()
# 训练
model_1.fit(X_train, y_train)
# 预测
y_pred1 = model_1.predict(X_test)
# 计算均方误差
print(mean_squared_error(y_test, y_pred1))
# 模型得分
print(model_1.score(X_test, y_test))
#预测房价
print(y_pred1)

#梯度下降
#建模
model_2 = SGDRegressor(max_iter=1000, tol=1e-3)
# 训练
model_2.fit(X_train, y_train)
# 预测
y_pred2 = model_2.predict(X_test)
# 计算均方误差
print(mean_squared_error(y_test, y_pred2))
# 模型得分
print(model_2.score(X_test, y_test))
#预测值房价
print(y_pred2)

