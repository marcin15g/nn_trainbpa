import neural_network as nn
import plots as my_plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Load file
colnames = ['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','Dilute','Proline']
datafile = pd.read_csv('wine.data', names = colnames, index_col = False)

#Separate input and output values from the datafile
Y = datafile['Class'].copy()
X = datafile.drop(columns=['Class'])

#Normalize X data
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(X)
X[:] = scaler.transform(X)

#Transform Y values into matrix
Y = pd.get_dummies(Y)

### === MAIN === ###

#Train neural network
test_ratio = 0.2 
n_hiddenLayers = 15
n_epochs = 2000
showCost = False
const_lRate = 0.01
adaptive_lRate = {
    'InitialRate' : 0.01,
    'DecrementVar': 0.7,
    'IncrementVar' : 1.05,
    'ErrorRate' : 1.04 
}

#Split data into train and test sections
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio)

#Train the network on train data
params = nn.train_nn(X_train, Y_train, n_hiddenLayers, n_epochs, adaptive_lRate, showCost=showCost)

#Test neural network on test data
A2, cache = nn.f_propagate(X_test, params)
acc = "{:.4f}".format(nn.accuracy(A2, Y_test))
print('Accuracy on test data: ' + acc + '%')

#Test neural network on train data
A2, cache = nn.f_propagate(X_train, params)
acc = "{:.4f}".format(nn.accuracy(A2, Y_train))
print('Accuracy on train data: ' + acc + '%')

#Test neural network on all data
A2, cache = nn.f_propagate(X, params)
acc = "{:.4f}".format(nn.accuracy(A2, Y))
print('Accuracy on all available data: ' + acc + '%')
