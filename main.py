import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load file
colnames = ['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','Dilute','Proline']
datafile = pd.read_csv('wine.data', names = colnames, index_col = False)

#Plot relations between params
relationship = sns.pairplot(datafile, vars=['Alcohol','Malic acid','Ash'], hue='Class')
#plt.show(relationship)

#Separate input and output values from the datafile
Y = datafile['Class'].copy()
X = datafile.drop(columns=['Class'])

#Normalize X data
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(X)
X[:] = scaler.transform(X)

#Transform Y values into matrix
Y = pd.get_dummies(Y)

#Split data into train and test sections
ratio = 0.2 # <--- Set the ratio of train and test rows
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ratio)

#Define sigmoid and sigmoid derivative functions
def sigmoid(x, deriv=False):
    return 1 / (1 + np.exp(-x))

def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Sizes of input and output
def layer_sizes(X,Y):
    n_x = X.shape[1]
    n_y = Y.shape[1]
    return (n_x, n_y)

#Initialize weights and biases for hidden layer and output layer
def init_params(n_x, n_h, n_y):
    return {
    "W1" : np.random.randn(n_x, n_h) * 0.01,
    "W2" : np.random.randn(n_h, n_y) * 0.01,
    }

#Calculate accuracy of the neural network
def accuracy(prediction, test):    
    predicted_oneHot = np.argmax(prediction, axis=1)
    test_oneHot = np.argmax(test.to_numpy(), axis=1)
    equals = np.equal(predicted_oneHot, test_oneHot)
    return np.mean(equals) 

#Forward propagate
def f_propagate(X, params):   
    #First layer
    Z1 = np.dot(X, params['W1'])
    A1 = sigmoid(Z1)
    #Second layer
    Z2 = np.dot(A1, params['W2'])
    A2 = sigmoid(Z2)

    cache = {
        'Z1' : Z1,
        'A1' : A1,
        'Z2' : Z2,
        'A2' : A2
    }
    return A2, cache

#Compute cost
def compute_cost(A2, Y):
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / len(logprobs)
    return cost

#Back propagate
def b_propagate(params, cache, X, Y):
    
    #Second layer
    dZ2 = cache['A2'] - Y
    dW2 = np.dot(cache['A1'].T, dZ2)
    #First layer
    dZ1 = np.dot(dZ2, params['W2'].T) * dSigmoid(cache['Z1'])
    dW1 = np.dot(X.T, dZ1)

    return {
        'dW2' : dW2,
        'dW1' : dW1
        }

#Update weights and biases on neurons
def update_params(params, grads, l_rate):    
    W1 = params['W1'] - l_rate * grads['dW1']
    W2 = params['W2'] - l_rate * grads['dW2']
    return {
        'W1' : W1,
        'W2' : W2
    }


def train_nn(X, Y, n_h, n_epochs, showCost = False):
    
    np.random.seed(6)
    
    #Get sizes of input and output layers
    n_x, n_y = layer_sizes(X, Y)
    
    #Initialize parameters (weights and biases)
    params = init_params(n_x, n_h, n_y)
    for i in range(0, n_epochs):

        #Forward propagate
        A2, cache = f_propagate(X, params)

        #Compute difference between our predictions and actual values
        cost = compute_cost(A2, Y)
        #Backpropagation
        grads = b_propagate(params, cache, X, Y)

        #Update weights and biases 
        l_rate = 0.02 #linear learning rate
        params = update_params(params, grads, l_rate)

        #Print cost each 10 epochs
        if((showCost == True) & (i % 10 == 0)):
            avgCost = cost.mean()
            print('Cost after iteration %i: %f' %(i, avgCost))
    return params

params = train_nn(X_train, Y_train, 15, 50, showCost=True)
A2, cache = f_propagate(X_test, params)



print(accuracy(A2, Y_test))