import neural_network as nn
import numpy as np
import plots as my_plt
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from random import randint

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


# === Test function definitions ===
def test_on_parameters(n_hiddenLayers, n_epochs, showCost):        
    # Split data into train and test sections
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)

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

    # #Test neural network on all data
    A2, cache = nn.f_propagate(X, params)
    acc = "{:.4f}".format(nn.accuracy(A2, Y))
    print('Accuracy on all available data: ' + acc + '%')


def test_range__hidden_and_epochs(hidden_start, hidden_end, hidden_step, epochs_start, epochs_end, epochs_step, iterate_n):

    plotX = plotY = plotZ =[]
    for hidden in range(hidden_start, hidden_end+1,hidden_step):
        for epoch in range(epochs_start, epochs_end+1,epochs_step):
            acc_arr = []
            for i in range(1,iterate_n+1):
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=randint(1,100))        
                params = nn.train_nn(X_train, Y_train, hidden, epoch, const_lRate, showCost=False)

                pred_test, pred_cache = nn.f_propagate(X_test, params)
                acc_test = nn.accuracy(pred_test, Y_test) 

                pred_train, train_cache = nn.f_propagate(X_train, params)
                acc_train = nn.accuracy(pred_train, Y_train)

                pred_all, all_cache = nn.f_propagate(X, params)
                acc_all = nn.accuracy(pred_all, Y)
                print('Test: %f, Train: %f, All: %f' %(acc_test, acc_train, acc_all))
                acc_arr.append((acc_test + acc_train + acc_all)/3)
            plotX.append(hidden)
            plotY.append(epoch)
            plotZ.append(np.mean(acc_arr)) #plotZ.append(acc_test)
            print('Hidden: %i - Epoch: %i - Average Acc: %f' %(hidden, epoch, np.mean(acc_arr)))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Hidden layers')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Accuracy')
    surf = ax.plot_trisurf(plotX, plotY, plotZ, linewidth=0.1, cmap='winter')
    plt.savefig('./plots/plot.png')
    plt.show()

def test_params(n_hidden, n_epochs, n_iterations):
    acc_arr = []
    lowest_test = lowest_train = lowest_all = 100

    for i in range(1, n_iterations + 1):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=i)        
        params = nn.train_nn(X_train, Y_train, n_hidden, n_epochs, adaptive_lRate, showCost=False)

        pred_test, pred_cache = nn.f_propagate(X_test, params)
        acc_test = nn.accuracy(pred_test, Y_test) 

        pred_train, train_cache = nn.f_propagate(X_train, params)
        acc_train = nn.accuracy(pred_train, Y_train)

        pred_all, all_cache = nn.f_propagate(X, params)
        acc_all = nn.accuracy(pred_all, Y) 

        if(acc_test < lowest_test): lowest_test = acc_test
        if(acc_train < lowest_train): lowest_train = acc_train
        if(acc_all < lowest_all): lowest_all = acc_all

        acc_arr.append((acc_test + acc_train + acc_all)/3)
        print('Iteration: %i, Test: %f, Train: %f, All: %f' %(i, acc_test, acc_train, acc_all))

    print('================================================')
    print('Average accuracy for all iterations: %f' %(np.mean(acc_arr)))
    print('Lowest accuracies >>> Test: %f, Train: %f, All: %f' %(lowest_test, lowest_train, lowest_all))




############## === MAIN === ##############

#Global params
test_ratio = 0.2
const_lRate = 0.01
adaptive_lRate = {
    'InitialRate' : 0.01,
    'DecrementVar': 0.7,
    'IncrementVar' : 1.05,
    'ErrorRate' : 1.04 
}

### TEST ON PREDEFINED PARAMETERS ###
# n_hiddenLayers = 14
# n_epochs = 500
# showCost = True

# test_on_parameters(n_hiddenLayers, n_epochs, showCost)


### TEST RANGE FOR HIDDEN LAYERS AND EPOCHS ###
# hidden_start = 8
# hidden_end = 8
# hidden_step = 11
# epochs_start = 500
# epochs_end = 500
# epochs_step = 50
# iterate_n = 1000

# test_range__hidden_and_epochs(hidden_start, hidden_end, hidden_step, epochs_start, epochs_end, epochs_step, iterate_n)


### TEST SPECIFIC PARAMETERS FOR n INTERATIONS ###
# n_hidden = 8
# n_epochs = 500
# n_iterations = 20

# test_params(n_hidden, n_epochs, n_iterations)


