import numpy as np
import pandas as pd

# 加载数据
data = {
    '年龄': ['青年', '青年', '青年', '青年', '青年', '中年', '中年', '中年', '中年', '中年', '老年', '老年', '老年', '老年', '老年'],
    '有工作': ['否', '否', '是', '是', '否', '否', '否', '是', '否', '否', '否', '否', '是', '是', '否'],
    '有房子': ['否', '否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否', '否', '否'],
    '信用': ['一般', '好', '好', '一般', '一般', '一般', '好', '好', '非常好', '非常好', '非常好', '好', '好', '非常好', '一般'],
    '类别': ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
}
X = pd.DataFrame(data)
# 数据转换
X['年龄'] = X['年龄'].map({'青年': 0, '中年': 1, '老年': 2})
X['有工作'] = X['有工作'].map({'否': 0, '是': 1})
X['有房子'] = X['有房子'].map({'否': 0, '是': 1})
X['信用'] = X['信用'].map({'一般': 0, '好': 1, '非常好': 2})

print(X)
# 计算未分类的信息熵
info = X['类别'].value_counts() / X['类别'].size
E = (info * np.log2(1 / info)).sum()  # 0.9709505944546687

gain_max = 0  # 最大的信息增益
best_spilt = {}  # 最佳列分条件
# 计算特征的信息熵
for feature in ['年龄', '有工作', '有房子', '信用']:
    x = X[feature].unique()
    x.sort()  # 排序
    for i in range(len(x) - 1):
        split = x[i:i + 2].mean()  # 分裂值
        # 筛选条件,将数据分开
        cond = X[feature] <= split
        # print(cond)

        # 计算左边概率，右边概率
        # print(split,cond.value_counts()/cond.size)
        p = cond.value_counts() / cond.size
        indexs = p.index
        print(indexs)
        entropy = 0
        for index in indexs:
            user = X[cond == index]['类别']  # 取出目标值y的数据
            p_user = user.value_counts() / user.size
            # 每个分支的信息熵
            entropy += (p_user * np.log2(1 / p_user)).sum() * p[index]
        if E - entropy > gain_max:
            gain_max = E - entropy
            best_spilt.clear()
            best_spilt[feature] = split
        print(f'分裂值:{split},{feature}的信息信息增益是,{E - entropy}')
print('最佳裂分条件是', best_spilt, '最大的信息增益是', gain_max)
