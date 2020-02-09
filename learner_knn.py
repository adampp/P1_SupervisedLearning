from sklearn import neighbors
import numpy as np
from learner_base import *

class KNN(BaseLearner):
    plotTitle = "KNN"
    plotFileName = "KNN"
    paramGrid = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'weights': ['uniform', 'distance'], 'p': [1.0, 2.0, 4.0]}]
    paramNames = ['n_neighbors', 'p']
    
    def __init__(self, n_neighbors, weights, algorithm, p):
        self.learner = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, p=p)
        self.paramRanges = [np.ceil(np.linspace(n_neighbors*self.paramLB, n_neighbors*self.paramUB, self.numLinSpace)).astype(int),
            np.linspace(p*self.paramLB, p*self.paramUB, self.numLinSpace)]