import pandas as pd
from sklearn import preprocessing

class SeismicData:
    def __init__(self):
        self.filepath = "data/seismic.csv"
        self.delimiter = ','
        self.outputMap = {0: -1, 1: 1}
        self.labels=["seismic", "seismoacoustic", "shift", "ghazard"]
        self.numerics=["genergy", "gpuls", "gdenergy", "gdpuls", "nbumps1", "nbumps2", "nbumps3", "nbumps4", "nbumps5", "nbumps6", "nbumps7", "nbumps89", "energy", "maxenergy"]
        self.plotTitle = "Seismic Dataset"
        self.plotFileName = "Seismic"
        array = [1]
        array[2]
    
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(len(self.numerics)):
            X[self.numerics[i]] = numericScaler.fit_transform(pd.DataFrame(X[self.numerics[i]]))
        
        data.iloc[:,0:-1] = X
        
        return data