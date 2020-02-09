import pandas as pd
from sklearn import preprocessing

class OccupancyData:
    def __init__(self):
        self.filepath = "data/occupancy.data"
        self.delimiter = ','
        self.outputMap = {0: -1, 1: 1}
        self.labels=[]
        self.numerics=["temperature", "humidity", "light", "co2", "humidity-ratio"]
        self.plotTitle = "Occupancy Dataset"
        self.plotFileName = "Occupancy"
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