import neural_network as nn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

#Present input and output data on plots
# sns.pairplot(datafile, vars=['Alcohol', 'Ash', 'Magnesium'], hue='Class')
# plt.show()



# === Test function definitions ===
def test_on_parameters(n_hiddenLayers, n_epochs, showCost):   

    # Split data into train and test sections
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)
    print(X.shape, X_train.shape, X_test.shape)
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

    plotX = []
    plotY = []
    plotZ = []
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

                acc_arr.append((acc_test + acc_train + acc_all)/3)
            plotX.append(hidden)
            plotY.append(epoch)
            plotZ.append(np.mean(acc_arr)) #plotZ.append(acc_test)
            print('Hidden: %i - Epoch: %i - Average Acc: %f' %(hidden, epoch, np.mean(acc_arr)))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Neurons in hidden layer')
    ax.set_ylabel('Epochs')
    ax.set_zlabel('Accuracy')
    surf = ax.plot_trisurf(plotX, plotY, plotZ, linewidth=0.1, cmap='winter')
    plt.savefig('./plots/plot.png')
    plt.show()

def test_inc_and_dec():

    plotX = []
    plotY = []
    plotZ = []
    size = 10

    incArr = np.linspace(1, 1.1, size)
    decArr = np.linspace(0.5, 1, size)

    for inc in range(0, size):
        for dec in range(0, size):
            acc_arr = []
            adaptive_lRate = {
                'InitialRate' : 0.01,
                'DecrementVar': decArr[dec],
                'IncrementVar' : incArr[inc],
                'ErrorRatio' : 1.04
            }
            for i in range(1,4):

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=randint(1,100))        
                params = nn.train_nn(X_train, Y_train, 8, 500, adaptive_lRate, showCost=False)

                pred_test, pred_cache = nn.f_propagate(X_test, params)
                acc_test = nn.accuracy(pred_test, Y_test) 

                pred_train, train_cache = nn.f_propagate(X_train, params)
                acc_train = nn.accuracy(pred_train, Y_train)

                pred_all, all_cache = nn.f_propagate(X, params)
                acc_all = nn.accuracy(pred_all, Y)

                acc_arr.append((acc_test + acc_train + acc_all)/3)

            plotX.append(incArr[inc])
            plotY.append(decArr[dec])
            plotZ.append(np.mean(acc_arr)) #plotZ.append(acc_test)
            print('Inc: %f - Dec: %f - Average Acc: %f' %(incArr[inc], decArr[dec], np.mean(acc_arr)))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('Increment')
    ax.set_ylabel('Decrement')
    ax.set_zlabel('Accuracy')
    surf = ax.plot_trisurf(plotX, plotY, plotZ, linewidth=0.1, cmap='summer')
    plt.savefig('./plots/plot.png')
    plt.show()

def test_params(n_hidden, n_epochs, n_iterations):
    acc_arr = []
    lowest_test = lowest_train = lowest_all = 100
    incidents = 0

    for i in range(1, n_iterations + 1):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=i)        
        params = nn.train_nn(X_train, Y_train, n_hidden, n_epochs, const_lRate, showCost=False)

        pred_test, pred_cache = nn.f_propagate(X_test, params)
        acc_test = nn.accuracy(pred_test, Y_test) 

        pred_train, train_cache = nn.f_propagate(X_train, params)
        acc_train = nn.accuracy(pred_train, Y_train)

        pred_all, all_cache = nn.f_propagate(X, params)
        acc_all = nn.accuracy(pred_all, Y) 

        if(acc_test < lowest_test): lowest_test = acc_test
        if(acc_train < lowest_train): lowest_train = acc_train
        if(acc_all < lowest_all): lowest_all = acc_all
        if(acc_all < 90 or acc_train < 90 or acc_test < 90): incidents += 1
        acc_arr.append((acc_test + acc_train + acc_all)/3)
        print('Iteration: %i, Test: %f, Train: %f, All: %f' %(i, acc_test, acc_train, acc_all))

    print('================================================')
    print('Average accuracy for all iterations: %f, Number of incidents (<90 acc): %i' %(np.mean(acc_arr), incidents))
    print('Lowest accuracies >>> Test: %f, Train: %f, All: %f' %(lowest_test, lowest_train, lowest_all))

def test_const_lRate(l_start, l_end):
    rateAcc = []
    singleAcc = []
    ratesArr = np.linspace(l_start,l_end, 100)
    for i in range(0, 100):
        for j in range(1,5):

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=i*j)        
            params = nn.train_nn(X_train, Y_train, 8, 500, ratesArr[i], showCost=False)

            pred_test, pred_cache = nn.f_propagate(X_test, params)
            acc_test = nn.accuracy(pred_test, Y_test) 

            pred_train, train_cache = nn.f_propagate(X_train, params)
            acc_train = nn.accuracy(pred_train, Y_train)

            pred_all, all_cache = nn.f_propagate(X, params)
            acc_all = nn.accuracy(pred_all, Y)

            singleAcc.append(np.mean([acc_test, acc_train, acc_all]))
        rateAcc.append(np.mean(singleAcc))
        print('Learning_rate: %f, Acc: %f' %(ratesArr[i], np.mean(singleAcc)))
        singleAcc = []

    plt.plot(ratesArr, rateAcc)
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')

    plt.savefig('./plots/plot.png')
    plt.show()

def test_adaptive_lRate():
    rateAcc = []
    singleAcc = []
    ratesArr = np.linspace(0.001,0.2, 50)
    for i in range(0, 50):
        for j in range(1,3):

            adaptive_lRate = {
                'InitialRate' : ratesArr[i],
                'DecrementVar': 0.7,
                'IncrementVar' : 1.05,
                'ErrorRatio' : 1.04
            }

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=i*j )        
            params = nn.train_nn(X_train, Y_train, 8, 500, adaptive_lRate, showCost=False)

            pred_test, pred_cache = nn.f_propagate(X_test, params)
            acc_test = nn.accuracy(pred_test, Y_test) 

            pred_train, train_cache = nn.f_propagate(X_train, params)
            acc_train = nn.accuracy(pred_train, Y_train)

            pred_all, all_cache = nn.f_propagate(X, params)
            acc_all = nn.accuracy(pred_all, Y)

            singleAcc.append(np.mean([acc_test, acc_train, acc_all]))
        rateAcc.append(np.mean(singleAcc))
        print('Error rate: %f, Acc: %f' %(ratesArr[i], np.mean(singleAcc)))
        singleAcc = []

    plt.plot(ratesArr, rateAcc)
    plt.xlabel('Initial Rate')
    plt.ylabel('Accuracy')
    plt.savefig('./plots/plot.png')
    plt.show()

def compare_rates():

    plotX = []
    plotConst = []
    plotAdapt = []

    for epoch in range(0, 400, 10):
        for j in range(1,5):

            singleConst = []
            singleAdapt = []

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=randint(1,100) )

            params_const = nn.train_nn(X_train, Y_train, 8, epoch, const_lRate, showCost=False)
            params_adapt = nn.train_nn(X_train, Y_train, 8, epoch, adaptive_lRate, showCost=False)

            pred_test_const, pred_cache = nn.f_propagate(X_test, params_const)
            acc_test_const = nn.accuracy(pred_test_const, Y_test) 
            pred_train_const, train_cache = nn.f_propagate(X_train, params_const)
            acc_train_const = nn.accuracy(pred_train_const, Y_train)
            pred_all_const, all_cache = nn.f_propagate(X, params_const)
            acc_all_const = nn.accuracy(pred_all_const, Y)

            pred_test_adapt, pred_cache = nn.f_propagate(X_test, params_adapt)
            acc_test_adapt = nn.accuracy(pred_test_adapt, Y_test) 
            pred_train_adapt, train_cache = nn.f_propagate(X_train, params_adapt)
            acc_train_adapt = nn.accuracy(pred_train_adapt, Y_train)
            pred_all_adapt, all_cache = nn.f_propagate(X, params_adapt)
            acc_all_adapt = nn.accuracy(pred_all_adapt, Y)
            
            singleConst.append(np.mean([acc_test_const, acc_train_const, acc_all_const]))
            singleAdapt.append(np.mean([acc_test_adapt, acc_train_adapt, acc_all_adapt]))

        plotX.append(epoch)
        plotConst.append(np.mean(singleConst))
        plotAdapt.append(np.mean(singleAdapt))
        print('Epoch: %f, Const acc: %f, Adaptive acc: %f' %(epoch, np.mean(singleConst), np.mean(singleAdapt)))

    plt.plot(plotX, plotConst, label='Constant Learning Rate')
    plt.plot(plotX, plotAdapt, label='Adaptive Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('./plots/plot.png')
    plt.show()


############## === MAIN === ##############

#Global params
test_ratio = 0.8
const_lRate = 0.01

adaptive_lRate = {
    'InitialRate' : 0.01,
    'DecrementVar': 0.7,
    'IncrementVar' : 1.05,
    'ErrorRatio' : 1.04 
}

### TEST ON PREDEFINED PARAMETERS ###
# n_hiddenLayers = 8
# n_epochs = 500
# showCost = True

# test_on_parameters(n_hiddenLayers, n_epochs, showCost)


### TEST RANGE FOR HIDDEN LAYERS AND EPOCHS ###
# test_ratio = 0.8

# hidden_start = 4
# hidden_end = 16
# hidden_step = 1
# epochs_start = 600
# epochs_end = 1000
# epochs_step = 50
# iterate_n = 3

# test_range__hidden_and_epochs(hidden_start, hidden_end, hidden_step, epochs_start, epochs_end, epochs_step, iterate_n)


### TEST SPECIFIC PARAMETERS FOR n INTERATIONS ###
test_ratio = 0.8
n_hidden = 14
n_epochs = 850
n_iterations = 30

test_params(n_hidden, n_epochs, n_iterations)


### TEST RANGE FOR CONSTANT LEARNING RATE ###
# l_start = 0.001
# l_end = 0.3

# test_const_lRate(l_start, l_end)

### TEST ADAPTIVE LEARNING RATE
# test_adaptive_lRate()

### TEST INCREMENT AND DECREMENT VARS 
# test_inc_and_dec()

### COMPARE CONST AND ADAPTIVE RATE
# compare_rates()
