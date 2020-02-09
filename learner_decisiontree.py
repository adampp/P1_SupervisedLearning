from sklearn import tree
import numpy as np
from learner_base import *


class DecisionTree(BaseLearner):
    plotTitle = "Decision Tree"
    plotFileName = "DecisionTree"
    # paramGrid = [{'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 3, 4, 5, 6]}]
    paramGrid = [{'criterion': ['gini', 'entropy'], 'max_depth': [None], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    paramNames = ['max_depth', 'min_samples_split']
    
    def __init__(self, criterion, splitter, max_depth, min_samples_split, xlim, ccp_alpha):
        self.learner = tree.DecisionTreeClassifier(criterion = criterion, splitter = splitter, max_depth = max_depth, min_samples_split = min_samples_split, ccp_alpha = ccp_alpha)
        self.xlim = xlim
        
        if max_depth is None:
            max_depth = 3
        self.paramRanges = [np.ceil(np.linspace(max_depth*self.paramLB, max_depth*self.paramUB, self.numLinSpace)).astype(int),
            np.ceil(np.linspace(min_samples_split*self.paramLB, min_samples_split*self.paramUB, self.numLinSpace)).astype(int)]
            
        for i in range(self.numLinSpace):
            if self.paramRanges[1][i] <= 1:
                self.paramRanges[1][i] = 2