from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1.get Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2.observe data
# 看資料的屬性
train.info()
# 統計數值(只針對數字類型)
# print(train.describe())

# 3.draw chart
columns = { 0: 'Die', 1: 'Alive' }
sns.countplot(train.Embarked, hue=train.Survived)
sns.FacetGrid(train, hue='Sex').map(sns.kdeplot, "Age").set(xlim=(0, 80))
# plt.show()

# 4.fill in NaN Data
train.Age = train.Age.fillna(train.Age.mean())
test.Age = test.Age.fillna(test.Age.mean())
# print(train.Age)
# print(train.Age.mean())

train.Cabin = train.Cabin.fillna('Other')
test.Cabin = test.Cabin.fillna('Other')
# print(train.Cabin)

# 5.category attribute conver to number
modify_train = pd.get_dummies(train, columns=['Sex', 'Pclass'])
test_train = pd.get_dummies(test, columns=['Sex', 'Pclass'])
# print(modify_train.info())

# 6.Random Forest ML
from sklearn.ensemble import RandomForestClassifier
x_train = modify_train[['Age','Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3']]
y_train = modify_train['Survived']

x_test = test_train[['Age','Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3']]

model = RandomForestClassifier()
model.fit(x_train, y_train)

test['Survived'] = model.predict(x_test)
y_test = test['Survived']

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

# 7.Submit
from pandas import DataFrame as df
dict1 = {'PassengerId': test.PassengerId,
         'Survived': y_test.astype(int)}
submit = df(dict1)
submit.to_csv('result.csv', index=False)

