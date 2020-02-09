from sklearn import svm
import numpy as np
from learner_base import *
from math import *

class SVM(BaseLearner):
    plotTitle = "SVM"
    plotFileName = "SVM"
    paramGrid = [{'C': [10.0, 50.0, 100.0, 500.0], 'kernel': ['poly', 'rbf'], 'degree': [2, 3], 'coef0': [0.0, 0.1, 1.0]}]
    paramNames = ['C', 'degree']
    
    def __init__(self, C, kernel, degree, gamma, coef0):
        self.learner = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
        if coef0 == 0.0:
            temp = np.linspace(0.0, 0.5, 10)
        else:
            temp = np.linspace(coef0*self.paramLB, coef0*self.paramUB, self.numLinSpace)
        self.paramRanges = [np.linspace(C*self.paramLB, C*self.paramUB, self.numLinSpace),
            np.linspace(ceil(degree*self.paramLB), degree*self.paramUB, self.numLinSpace)]