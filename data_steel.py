import pandas as pd
import numpy as np
from sklearn import preprocessing

class SteelData:
    def __init__(self):
        self.filepath = "data/steel.csv"
        self.delimiter = ','
        self.outputMap = {0: -1, 1: 1}
        self.labels=[]
        self.numerics=[]
        self.plotTitle = "Steel Dataset"
        self.plotFileName = "Steel"
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.columns = data.columns.str.strip()
        
        all = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

        drop = ["Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
        data = data.drop(columns = drop)
        data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        # fault = data["Pastry"].values
        # fault = fault + data["Z_Scratch"].values * 2
        # fault = fault + data["K_Scatch"].values * 3
        # fault = fault + data["Stains"].values * 4
        # fault = fault + data["Dirtiness"].values * 5
        # fault = fault + data["Bumps"].values * 6
        # fault = fault + data["Other_Faults"].values * 7
        # data.drop(columns = all)
        # data['class'] = fault
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(27):
            X.iloc[:,i] = numericScaler.fit_transform(pd.DataFrame(X.iloc[:,i]))
        
        data.iloc[:,0:-1] = X
        
        return data