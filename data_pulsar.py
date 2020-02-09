import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import random
from math import *

class PulsarData:
    def __init__(self):
        self.filepath = "data/pulsar.csv"
        self.delimiter = ','
        self.outputMap = {0: -1, 1: 1}
        self.labels=[]
        self.numerics=[]
        self.plotTitle = "Pulsar Dataset"
        self.plotFileName = "Pulsar"
        array = [1]
        array[2]
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.columns = data.columns.str.strip()
        
        data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        a = pd.RangeIndex(len(data))
        Y = data.iloc[:,-1]
        
        posIdxs = a[Y == 1]
        negIdxs = a[Y == -1]
        
        newNegIdxs = random.sample_without_replacement(n_population=len(negIdxs), n_samples=ceil(len(negIdxs)*0.2), random_state = 1)
        
        idxs = posIdxs.union(negIdxs[newNegIdxs])
        
        data = data.iloc[idxs, :]
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(8):
            X.iloc[:,i] = numericScaler.fit_transform(pd.DataFrame(X.iloc[:,i]))
        
        data.iloc[:,0:-1] = X
        
        data = data.drop(columns='Mean of the DM-SNR curve')
        
        return data