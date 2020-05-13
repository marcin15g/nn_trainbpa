import neural_network as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

def data_relations(file, columns):
    return sns.pairplot(file, vars=columns, hue='Class')
