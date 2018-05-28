import pandas as pd

def hello():
  print('Hello World! here is in main')

def getData():
  train = pd.read_csv('train.csv')
  test = pd.read_csv('test.csv')
  return train, test

def fillMissing(train, test):
  train.Age = train.Age.fillna(train.Age.mean())
  test.Age = test.Age.fillna(test.Age.mean())
  return train, test

def oneHotEncoding(train, test):
  modify_train = pd.get_dummies(train, columns=['Sex', 'Pclass'])
  modify_test = pd.get_dummies(test, columns=['Sex', 'Pclass'])
  return modify_train, modify_test

def prepareXY(train, test):
  x_train = train[['Age', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3']]
  y_train = train['Survived']
  x_test = test[['Age', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3']]
  test['Survived'] = 0
  y_test = test['Survived']
  return x_train, y_train, x_test, y_test

def showInfo(train, test):
  print(train.info())
  print(test.info())


if __name__ == '__main__':
  getData()
  fillMissing()