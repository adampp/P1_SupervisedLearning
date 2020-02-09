import warnings

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
from sklearn import utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_learning_curve import *
from plot_model_complexity import *
from plot_cost_complexity_pruning import *
from plot_nn_learning_curve import *
from data_adult import *
from data_occupancy import *
from data_news import *
from data_seismic import *
from data_steel import *
from data_wall import *
from data_baby import *
from data_pulsar import *
from data_heart import *
from learner_decisiontree import *
from learner_neuralnet import *
from learner_svm import *
from learner_knn import *
from learner_adaboost import *

randomState = 1
testSize = 0.15
kfolds = 7

def learningCurve(dataSet, learner, stepNum, updateTxt):
    
    data = dataSet.load()
    
    X = data.iloc[:,0:-1]
    Y = data.iloc[:,-1]

    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
        random_state = randomState)
    # xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(xTrain, yTrain,
        # test_size = testSize, random_state = randomState)
    # cv = model_selection.ShuffleSplit(n_splits=20, test_size=testSize,
        # random_state=randomState)
    
    print("====================================================")
    print(f"LearningCurve-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:")
    
    samplePercent = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    title = f"{learner.plotTitle} Learning Curve on {dataSet.plotTitle}"
    plt, resultStr = plot_learning_curve(learner.learner, title, xTrain, yTrain, cv=kfolds,
        train_sizes=samplePercent, ylim=(0.60, 1.01), n_jobs = 4)
    plt.savefig(f"plots/{dataSet.plotFileName}_{learner.plotFileName}_{stepNum}_LearningCurve.png",
        format='png')
    
    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"LearningCurve-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:"+"\n")
            file.write(resultStr+"\n")
    
    if learner.plotFileName == "NeuralNet":
        epochs = np.linspace(20, 500, 20)
        plot_nn_learning_curve(learner.learner, title, xTrain, yTrain, cv=kfolds, train_epochs=epochs,
            testSize=testSize, ylim=None)
        plt.savefig(f"plots/{dataSet.plotFileName}_{learner.plotFileName}_{stepNum}_NNLearningCurve.png",
            format='png')
    
def modelComplexity(dataSet, learner, stepNum, updateTxt):
    data = dataSet.load()
    
    X = data.iloc[:,0:-1]
    Y = data.iloc[:,-1]

    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
        random_state = randomState)
    # xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(xTrain, yTrain,
        # test_size = testSize, random_state = randomState)
    # cv = model_selection.ShuffleSplit(n_splits=20, test_size=testSize, random_state=randomState)
    
    print("====================================================")
    print(f"ModelComplexity-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:")
    
    title = f"{learner.plotTitle} Model Complexity Curve on {dataSet.plotTitle}"
    
    xlabel = learner.paramNames
    param_name = learner.paramNames
    param_range = learner.paramRanges
    plt, resultStr = plot_model_complexity(learner.learner, X, Y, title, xlabel, param_name,
        param_range, cv=kfolds)
    
    plt.savefig(f"plots/{dataSet.plotFileName}_{learner.plotFileName}_{stepNum}_ModelComplexity.png",
        format='png')
    
    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"ModelComplexity-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:"+"\n")
            file.write(resultStr+"\n")
    
def dtPruning(dataSet, learner, stepNum, updateTxt):
    data = dataSet.load()
    
    X = data.iloc[:,0:-1]
    Y = data.iloc[:,-1]

    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
        random_state = randomState)
    # xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(xTrain, yTrain,
        # test_size = testSize, random_state = randomState)
    # cv = model_selection.ShuffleSplit(n_splits=20, test_size=testSize, random_state=randomState)
    
    print("====================================================")
    print(f"DTPruning-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:")
    
    title = f"{learner.plotTitle} Accuracy vs. alpha on {dataSet.plotTitle}"
    
    xlabel = learner.paramNames
    param_name = learner.paramNames
    param_range = learner.paramRanges
    plt, resultStr = plot_cost_complexity_pruning(learner.learner, X, Y, testSize, title, xlim=learner.xlim)
    
    plt.savefig(f"plots/{dataSet.plotFileName}_{learner.plotFileName}_{stepNum}_DTPruning.png", format='png')
    
    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"DTPruning-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}"+"\n")
            file.write(resultStr+"\n")

def gridSearch(dataSet, learner, stepNum, updateTxt):
    
    data = dataSet.load()
    
    X = data.iloc[:,0:-1]
    Y = data.iloc[:,-1]

    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
        random_state = randomState)
    # xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(xTrain, yTrain,
        # test_size = testSize, random_state = randomState)
    # cv = model_selection.ShuffleSplit(n_splits=5, test_size=testSize, random_state=randomState)
    
    gridSearch = model_selection.GridSearchCV(learner.learner, param_grid = learner.paramGrid, cv=kfolds)
    gridSearch.fit(xTrain, yTrain)
    print("====================================================")
    print(f"GridSearch-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:")
    print(f"Score={gridSearch.best_score_}, params={gridSearch.best_params_}")
    
    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"GridSearch-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:"+"\n")
            file.write(f"Score={gridSearch.best_score_}, params={gridSearch.best_params_}"+"\n")

def testData(dataSet, learner, stepNum, updateTxt):
    
    data = dataSet.load()
    
    X = data.iloc[:,0:-1]
    Y = data.iloc[:,-1]

    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
        random_state = randomState)
    # xTrain, xValidate, yTrain, yValidate = model_selection.train_test_split(xTrain, yTrain,
        # test_size = testSize, random_state = randomState)
    # cv = model_selection.ShuffleSplit(n_splits=5, test_size=testSize, random_state=randomState)
    
    learner.learner.fit(xTrain, yTrain)
    
    train_results = learner.learner.score(xTrain,yTrain)
    test_results = learner.learner.score(xTest,yTest)
    print("====================================================")
    print(f"FinalTest-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:")
    print(f"Train={train_results}, Test={test_results}")
    
    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"FinalTest-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:"+"\n")
            file.write(f"Train={train_results}, Test={test_results}"+"\n")
    
def testSwitch(dataSet, learner, switch, stepNum, updateTxt):
    if learner.plotFileName == "NeuralNet":
        warnings.catch_warnings()
        warnings.simplefilter("ignore")
            
    if switch == 'gridsearch':
        gridSearch(dataSet, learner, stepNum, updateTxt)
    elif switch == 'learningcurve':
        learningCurve(dataSet, learner, stepNum, updateTxt)
    elif switch == 'modelcomplexity':
        modelComplexity(dataSet, learner, stepNum, updateTxt)
    elif switch == 'pruning':
        dtPruning(dataSet, learner, stepNum, updateTxt)
    elif switch == 'testdata':
        testData(dataSet, learner, stepNum, updateTxt)
    else:
        print(f"ERROR - test {switch} not implemented")

if __name__ == "__main__":

    #################################
    algorithm = 'svm'
    dataset = 'adult'
    test = 'testdata'
    stepNum = 7
    updateTxt = True
    #################################
    
    if dataset == 'baby':
        data = BabyData()
    elif dataset == 'adult':
        data = AdultData()
        
    
    if algorithm == 'dt':
        learner = DecisionTree(criterion='gini', splitter='best', max_depth=None, min_samples_split=4,
            xlim=(0.0, 0.025), ccp_alpha=0.0)
        
        if dataset == 'baby':
            pass
            learner = DecisionTree(criterion='gini', splitter='best', max_depth=None, min_samples_split=9,
                xlim=(0.0, 0.025), ccp_alpha=0.004380914244641486)
        elif dataset == 'adult':
            pass
            learner = DecisionTree(criterion='gini', splitter='best', max_depth=None, min_samples_split=9,
                xlim=(0.0, 0.0025), ccp_alpha=0.0006805460013573766)
        
    elif algorithm == 'knn':
        learner = KNN(n_neighbors=5, weights = 'uniform', algorithm='auto', p=2)
        
        if dataset == 'baby':
            pass
            learner = KNN(n_neighbors=10, weights = 'distance', algorithm='auto', p=1.0)
        elif dataset == 'adult':
            pass
            learner = KNN(n_neighbors=18, weights = 'uniform', algorithm='auto', p=1.3)
        
    elif algorithm == 'nn':
        learner = NeuralNetwork(hidden_layer_sizes=(20,), activation='relu', solver='sgd', alpha=1e-4,
            learning_rate='constant', learning_rate_init=0.001, max_iter=500, random_state=randomState)
        
        if dataset == 'baby':
            pass
            learner = NeuralNetwork(hidden_layer_sizes=(5,), activation='relu', solver='sgd', alpha=1e-4,
                learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
        elif dataset == 'adult':
            pass
            learner = NeuralNetwork(hidden_layer_sizes=(54,), activation='tanh', solver='adam', alpha=1e-4,
                learning_rate='constant', learning_rate_init=0.00044642857, max_iter=500, random_state=randomState)
        
    elif algorithm == 'svm':
        learner = SVM(C=1.0, kernel='rbf', degree=2, gamma='scale', coef0=0.0)
        
        if dataset == 'baby':
            pass
            learner = SVM(C=10.0, kernel='poly', degree=3, gamma='scale', coef0=1.0)
        elif dataset == 'adult':
            pass
            learner = SVM(C=506.0, kernel='poly', degree=3.0, gamma='scale', coef0=1.0)
    
    elif algorithm == 'ab':
        dt = DecisionTree(criterion='gini', splitter='best', max_depth=5, min_samples_split=4,
            xlim=(0.0, 0.025), ccp_alpha=0.0)
        learner = AdaBoost(deepcopy(dt.learner), n_estimators=50, learning_rate=0.7, random_state=randomState)
        
        if dataset == 'baby':
            pass
            learner = AdaBoost(deepcopy(dt.learner), n_estimators=300, learning_rate=0.8,
                random_state=randomState)
        elif dataset == 'adult':
            pass
            learner = AdaBoost(deepcopy(dt.learner), n_estimators=125, learning_rate=0.17273469,
                random_state=randomState)
    
    testSwitch(data, learner, test, stepNum, updateTxt)
    