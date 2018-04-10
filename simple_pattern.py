import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score, average_precision_score
import pandas as pd

from keras.layers import Input
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score, average_precision_score

# the super paramaters, it is not enough but ...
learing_ratio = 0.01 
batch_size = 5
train_turns = 8

def to_label_feature(tdf):
    label = np_utils.to_categorical(tdf['label'].values - 1)
    feature = tdf[tdf.columns[1:]].values
    return label, feature

# the evaluation by self define not the keras.
def evaluation(label, result):
    accuracy = 1.0*np.sum(np.argmax(label, axis = 1) == np.argmax(result, axis = 1)) / label.shape[0]
    loss = np.sum(-np.log(result) * label) / label.shape[0]
    print accuracy, loss
    for i in range(3):
        print 'result on class %d' % (i + 1)
        lbl = label[:, i]
        pred = result[:, i]
        print 'aupr is %f, auroc is %f' % (average_precision_score(lbl, pred), roc_auc_score(lbl, pred))

def make_model():
  inp = Input(shape = (13, ))
  oup = Dense(3, activation = 'softmax')(inp)
  model = Model(inputs = inp, outputs = oup)
  sgd = SGD(lr = learing_ratio)
  model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
  return model

def make_zeromean_value(df):
  for i in range(1, 14):
    column_name = df.columns[i]
    print column_name
    #make the df value to be 0 of summer.
    avg = df[column_name].mean()
    std = df[column_name].std()
    df[column_name] = (df[column_name] - avg) / std
  return df

def cutData_byRatio(df, ratio):
  #make labels and the indexs
  indexes = np.random.permutation(df.shape[0])
  pos = int(len(indexes) * 0.8)
  train_indexes = indexes[:pos]
  test_indexes = indexes[pos:]
  return df.iloc[train_indexes], df.iloc[test_indexes]

def main_start_training():
  # get the Data
  df = pd.read_csv('wine.data', sep = ',', header = None)
  df.columns = ['label',
              'Alcohol',
              'Malic acid',
              'Ash',
              'Alcalinity of ash',
              'Magnesium',
              'Total phenols',
              'Flavanoids',
              'Nonflavanoid phenols',
              'Proanthocyanins',
              'Color intensity',
              'Hue',
              'OD280/OD315 of diluted wines',
              'Proline']

  # set the everoments          
  model = make_model()
  df = make_zeromean_value(df)
  train_df, test_df = cutData_byRatio(df, 0.8)
  #get the dateframe frome the labels
  train_label, train_feature = to_label_feature(train_df)
  test_label, test_feature = to_label_feature(test_df)

  #fit is the ture training , evaluation is the result by self custarmer
  for _ in range(train_turns):
    model.fit(train_feature, train_label, batch_size = batch_size, epochs = 1, validation_data = (test_feature, test_label))
    train_result = model.predict(train_feature)
    evaluation(train_label, train_result)
    test_result = model.predict(test_feature)
    evaluation(test_label, test_result)

if __name__ == '__main__':
  main_start_training()
