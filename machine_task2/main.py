import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('heart.csv')
data=data.sample(frac=1)
#print(data)
columns=data.shape[1] #1*4 ht5tar 4
x_training =data.iloc[:231,[3,4,7,9]]
x_training=(x_training-x_training.min())/ (x_training.max()-x_training.min())#normalization
x_training.insert(0,'ones',1)
y_training =data.iloc[ :231,columns-1:columns]
x_training =np.matrix(x_training.values)
y_training =np.matrix(y_training.values)
theta=np.zeros(5)
theta=np.matrix(theta)
x_testing=data.iloc[231:303,[3,4,7,9]]
x_testing=(x_testing-x_testing.min())/ (x_testing.max()-x_testing.min())
x_testing.insert(0,'ones',1)
y_testing=data.iloc[231:303,columns-1:columns] # this is Y values
x_testing=np.matrix(x_testing.values) # to convert to matrix
y_testing=np.matrix(y_testing.values)  # to convert to matrix

def cost(xg,yg,thetag):
    first=np.multiply(-yg,np.log(sigmuied(xg*thetag.T))) #y=1
    second=np.multiply((1-yg),np.log(1-sigmuied(xg*thetag.T))) #y=0
    return np.sum(first-second)/len(xg)
def sigmuied(z):
    return 1/(1+np.exp(-z))
def gradient_descent(thetag,xg,yg,alpha,iteration) :
    costs = np.zeros(iteration)
    t=np.matrix(np.zeros(thetag.shape))
    #theta_min=np.matrix(np.zeros(thetag.shape))
    parameters=int(thetag.ravel().shape[1])
    #grade=np.zeros(parameters)
    #min_cost=1000000
    for j in range(iteration):
        error = sigmuied(xg * thetag.T) - yg
        for i in range(parameters):
            term = np.multiply(error, xg[:,i])
            t[0, i] = t[0, i] - ((alpha / len(xg))  * np.sum(term))
        thetag= t
        costs[j] = cost(xg, yg, t)
    return thetag,costs
def predict(theta,x):
    prob=sigmuied(x * theta.T)
    return [1 if x>=0.5 else 0 for x in prob]
num_of_iteration=5000
theeta_min,costs=gradient_descent(theta,x_training,y_training,0.1,num_of_iteration)
prediction=predict(theeta_min,x_testing)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
           for ( a,b) in zip(prediction, y_testing)]
accuracy = (sum(map(int, correct)) / len(correct) *100)
print('accuracy={0} %'.format(accuracy))
print("new predction \n",prediction)
print(theeta_min)
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(np.arange(num_of_iteration),costs,'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Data')
plt.show()