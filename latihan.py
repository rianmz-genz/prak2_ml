import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

class Latihan():
    def __init__(self, df, df2, X, X_train, X_test):
        self.df = df
        self.df2 = df2
        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        
    def lat_one(self):
        print('\n latihan 1')
        #lat 1:
        #ubah nilai nan menjadi median
        self.df2['Age'] = self.df2['Age'].fillna(self.df['Age'].median())
        self.df2['Salary'] = self.df2['Salary'].fillna(self.df['Salary'].median())
        print(self.df2)
    
    def lat_two(self):
        print('\n latihan 2')
        #ubah nilai missing menjadi median
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        imputer.fit(self.X[:, 1:3])
        self.X[:, 1:3] = imputer.transform(self.X[:, 1:3])
        print(self.X)

    def lat_three(self):
        print('\n latihan 3')
        minmax_scaler = MinMaxScaler()
        # Standarisasi variabel X_train
        self.X_train[:, 1:] = minmax_scaler.fit_transform(self.X_train[:, 1:])
        # Standarisasi variabel X_test
        self.X_test[:, 1:] = minmax_scaler.transform(self.X_test[:, 1:])
        print('X_train')
        print(self.X_train)
        print('X_test')
        print(self.X_test)