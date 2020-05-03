import neural_network as nn
import matplotlib.pyplot as plt
import seaborn as sns

def data_relations(file, columns):
    return sns.pairplot(file, vars=columns, hue='Class')


# TODO
# def range_HiddenLayers(n_h, range, X, Y, epochs, l_rate):

#     for i in range(1,n_h+1):
#         params = nn.train_nn(X, Y, i, epochs, l_rate)
#         A2, cache = nn.f_propagate(X_train, params)
#         train_acc.append(accuracy(A2, Y_train))

#         A2, cache = f_propagate(X_test, params)
#         test_acc.append(accuracy(A2, Y_test))

#         A2, cache = f_propagate(X, params)
#         all_acc.append(accuracy(A2, Y))    
        
#         print('ranged data for %i hidden_layers' %(i))