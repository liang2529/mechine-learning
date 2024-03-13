from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯分类器
import pandas as pd

# 加载数据集
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
                'Overcast', 'Overcast', 'Rain'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot',
             'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'High'],
    'Windy': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
              'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})
# 数据转换
data['Outlook'] = data['Outlook'].map({'Overcast': 0, 'Rain': 1, 'Sunny': 2})
data['Temp'] = data['Temp'].map({'Cool': 0, 'Hot': 1, 'Mild': 2})
data['Humidity'] = data['Humidity'].map({'High': 0, 'Normal': 1})
data['Windy'] = data['Windy'].map({'Strong': 0, 'Weak': 1})
data['PlayTennis'] = data['PlayTennis'].map({'No': 0, 'Yes': 1})
print(data)
# 准备特征和标签
X = data.drop('PlayTennis', axis=1)
y = data['PlayTennis']

# 创建朴素贝叶斯分类器
clf = MultinomialNB()

# 训练分类器
clf.fit(X, y)

# 预测样本  {Outlook=Sunny:1，Temp=Mild:2，Humidity=Normal:0,Windy=Strong:1}
sample = [[1, 2, 0, 1]]
prediction = clf.predict(sample)

if prediction[0] == 'Yes':
    print('{Outlook=Sunny，Temp=Mild，Humidity=Normal，Windy=Strong}该样本会打球。')
else:
    print('{Outlook=Sunny，Temp=Mild，Humidity=Normal，Windy=Strong}该样本不会打球。')
