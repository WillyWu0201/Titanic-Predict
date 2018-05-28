from preprocess import *
from deeplearnmodel import *

train, test = getData()
train, test = fillMissing(train=train, test=test)
train, test = oneHotEncoding(train=train, test=test)
x_train, y_train, x_test, y_test = prepareXY(train=train, test=test)

showInfo(train=train, test=test)

deepNN(x_train, y_train, x_test, y_test)