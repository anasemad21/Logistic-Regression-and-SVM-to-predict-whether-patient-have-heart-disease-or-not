import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Anoosh
data=pd.read_csv('heart.csv')
columns=data.shape[1]        # data.shape(303,14)
data=data.sample(frac=1)
#print(columns)   14
x_training =data.iloc[:231,[5,9]]  #70%
x_training=(x_training-x_training.min())/ (x_training.max()-x_training.min())#normalization
x_training.insert(0,'ones',1)
y_training =data.iloc[ :231,columns-1:columns]
x_training=np.matrix(x_training.values)
y_training=np.matrix(y_training.values)
#print(len(x_training))
#print(y_training)
#print(x_training)
#------------------------------------> 4 ,7
x_testing=data.iloc[231:303,[5,9]]  #30%
x_testing=(x_testing-x_testing.min())/ (x_testing.max()-x_testing.min())
x_testing.insert(0,'ones',1)
y_testing=data.iloc[231:303,columns-1:columns]
x_testing=np.matrix(x_testing.values)
y_testing=np.matrix(y_testing.values)
w=np.zeros(x_training.shape[1])
w=np.matrix(w)
#print(w.shape)  #(1,3)
def svm(x_training,y_training,w):
    iterate=500
    alpha=0.1
    lam=1/iterate
    for i in range(iterate):
        for j in range(len(x_training)):
            if (y_training[j] * np.dot(w, x_training[j].T)) < 1:   #y*f(x)
              w = w + alpha * (np.dot(y_training[j] ,x_training[j]) - (2 * (lam * w)))
            else:
              w = w - alpha * (2 * lam * w)
    return w

def accurcy(w,x_testing,y_testing):
    counter=0
    for i in range(len(x_testing)):
        if (y_testing[i] * np.dot(w, x_testing[i].T)) >= 1:
            counter+=1
        else:
            counter+=0
    return (counter/len(x_testing))*100

w_best=svm(x_training,y_training,w)
print("accurcy={0}%".format(accurcy(w_best,x_testing,y_testing)))