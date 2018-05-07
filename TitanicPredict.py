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
import matplotlib.pyplot as plt # 也可以試試seaborn, bokeh
# 顯示生存或死亡的數量
fig = plt.figure()
plt.subplot2grid((2, 3), (0, 0))
plt.title('Survived Number')
plt.ylabel('numbers')
plt.xlabel('Die'  'Live')
data.Survived.value_counts().plot(kind='bar')

# 只顯示生存的人的艙等
plt.subplot2grid((2, 3), (0, 1))
plt.title('pClass Live')
plt.ylabel('numbers')
plt.xlabel('C0'  'C1'  'C2')
data.Pclass[data.Survived == 1].value_counts().sort_index().plot(kind='bar')

# 只顯示死亡的人的艙等
plt.subplot2grid((2, 3), (0, 2))
plt.title('pClass Die')
plt.ylabel('numbers')
plt.xlabel('C0'  'C1'  'C2')
data.Pclass[data.Survived == 0].value_counts().sort_index().plot(kind='bar')

# 年齡分佈
plt.subplot2grid((2, 3), (1, 0), colspan=2)
plt.title('Age')
plt.ylabel('numbers')
data.Age[data.Pclass == 1].value_counts().sort_index().plot(kind='kde')
data.Age[data.Pclass == 2].value_counts().sort_index().plot(kind='kde')
data.Age[data.Pclass == 3].value_counts().sort_index().plot(kind='kde')
plt.legend(('p1', 'p2', 'p3'), loc='best')

# 性別分佈
plt.subplot2grid((2, 3), (1, 2))
plt.title('Sex')
plt.ylabel('numbers')
data.Age[data.Sex == 'female'].value_counts().sort_index().plot(kind='kde')
data.Age[data.Sex == 'male'].value_counts().sort_index().plot(kind='kde')
plt.legend(('female', 'male'), loc='best')
# plt.show()

# ========================================
from  sklearn.ensemble import RandomForestRegressor
# age pre-processing
# data = data.Age[data.Age.notnull()]
# print(data)
# print(type(data))
# print(data.shape)

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr


data_train, rfr = set_missing_ages(data)
print(data_train)
print(rfr)

