from sklearn import tree
from sklearn import ensemble
import numpy as np
from learner_base import *


class AdaBoost(BaseLearner):
    plotTitle = "AdaBoost"
    plotFileName = "AdaBoost"
    paramGrid = [{'n_estimators': [1, 10, 20, 40, 80, 160, 300], 'learning_rate': [0.1, 0.2, 0.4, 0.8]},]
    paramNames = ['n_estimators', 'learning_rate']
    
    def __init__(self, childLearner, n_estimators, learning_rate, random_state):
        self.learner = ensemble.AdaBoostClassifier(base_estimator=childLearner, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        self.paramRanges = [np.ceil(np.linspace(n_estimators*self.paramLB, n_estimators*self.paramUB, self.numLinSpace)).astype(int),
            np.linspace(learning_rate*self.paramLB, learning_rate*self.paramUB, self.numLinSpace)]