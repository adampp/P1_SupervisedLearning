from math import *
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import random

class NewsData:
    def __init__(self):
        self.filepath = "data/news.csv"
        self.delimiter = ','
        # self.outputMap = {"<=50K": -1, "<=50K.": -1, ">50K": 1, ">50K.": 1}
        self.labels=[]
        self.numerics=[]
        self.plotTitle = "News Dataset"
        self.plotFileName = "News"
        array = [1]
        array[2]
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.columns = data.columns.str.strip()
        
        idxs = random.sample_without_replacement(n_population=len(data), n_samples=ceil(len(data)*0.2), random_state = 1)
        data = data.iloc[idxs, :]
        
        threshold = 9000
        data.iloc[:,-1].values[data.iloc[:,-1].values < threshold] = -1
        data.iloc[:,-1].values[data.iloc[:,-1].values >= threshold] = 1
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(58):
            X.iloc[:,i] = numericScaler.fit_transform(pd.DataFrame(X.iloc[:,i]))
        
        data.iloc[:,0:-1] = X
        
        return data