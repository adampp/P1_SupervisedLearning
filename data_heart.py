import pandas as pd
import numpy as np
from sklearn import preprocessing

class HeartData:
    def __init__(self):
        self.filepath = "data/heart.csv"
        self.delimiter = ','
        self.outputMap = {0: -1, 1: 1}
        self.labels=[]
        self.numerics=[]
        self.plotTitle = "Heart Dataset"
        self.plotFileName = "Heart"
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.columns = data.columns.str.strip()
        
        # data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)
        
        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(13):
            X.iloc[:,i] = numericScaler.fit_transform(pd.DataFrame(X.iloc[:,i]))
        
        data.iloc[:,0:-1] = X
        
        return data