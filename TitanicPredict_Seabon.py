import pandas
from pandas import DataFrame
import numpy
from sklearn.linear_model import LogisticRegression

data = pandas.read_csv('train.csv')
# print(data)
# 看資料的屬性
data.info()
# 統計數值(只針對數字類型)
# print(data.describe())

# # 顯示生存或死亡的數量
# survived = data.Survived.value_counts().plot(kind='bar')


# draw data
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(data.Survived.value_counts())
plt.show()