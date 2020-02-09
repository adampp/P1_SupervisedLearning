import pandas as pd
from sklearn import preprocessing

class WallData:
    def __init__(self):
        self.filepath = "data/wallfollowing.data"
        self.delimiter = ','
        self.outputMap = {"Move-Forward": 1, "Slight-Right-Turn": 2, "Sharp-Right-Turn": 3, "Slight-Left-Turn": 4}
        self.labels=[]
        self.numerics=["sd-front", "sd-left", "sd-right", "sd-back"]
        self.plotTitle = "Wall Following Dataset"
        self.plotFileName = "Wall"
    
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