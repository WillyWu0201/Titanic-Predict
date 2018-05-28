from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

def deepNN(x_train, y_train, x_test, y_test):
  model = Sequential()
  model.add(Dense(units=40, input_dim=6, kernel_initializer='uniform', activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  train_history = model.fit(x=np.array(x_train), y=np.array(y_train), validation_split=0.1, epochs=30,
                            batch_size=30, verbose=2)

  scores = model.evaluate(x=np.array(x_test), y=np.array(y_test))
  print('\nscore = ', scores )
  show_train_history(train_history, 'acc', 'val_acc')

def show_train_history(train_history, train, validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train History')
  plt.ylabel('Train')
  plt.xlabel('Epoch')
  plt.legend(['train', 'validation'], loc ='center right')
  plt.show()