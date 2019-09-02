import numpy as np
import random
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

seed = 1234
random.seed(seed)
np.random.seed(seed)

#Global values
x = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([0, 1, 1, 0])

def make_one_layer_pred(train_data,train_labels,test_data):
    clas = MLPClassifier(hidden_layer_sizes=(2,),learning_rate_init=0.3,activation='logistic',max_iter=10000)
    clas.fit(train_data,train_labels)
    print (clas.coefs_)
    predict = clas.predict(test_data)
    return predict

if __name__ == '__main__':

    start_time = datetime.datetime.now()  # Track learning starting time
    one_layer_result = make_one_layer_pred(x,y,x)
    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = (end_time - start_time).total_seconds()  # Track execution time
    print('time spend=%.4f sec & acc=%.2f'%(exection_time, accuracy_score(y,one_layer_result)) )
    print('results ',one_layer_result)

