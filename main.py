#import libraries
import numpy as np
from dataset import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = Dataset("Data.csv")
df = dataset.df

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(df.info())
mv=df.isna().sum()
print('\nJumlah missing value tiap kolom: \n', mv)
df1=df.copy()
#ukuran data awal
print("Sebelum: ", df1.shape)
#hapus baris yang memiliki missing value
df1.dropna(inplace=True)
#ukuran data akhir
print("Setelah: ", df1.shape)  
df2 = df.copy()
df2['Age'] = df2['Age'].fillna(df['Age'].mean())
df2['Salary'] = df2['Salary'].fillna(df['Salary'].mean())
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
label_encoder_x= LabelEncoder()
X[:, 0]= label_encoder_x.fit_transform(X[:, 0])
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
st_x= StandardScaler()
X_train[:, 1:]= st_x.fit_transform(X_train[:, 1:])
X_test[:, 1:]= st_x.transform(X_test[:, 1:])


from latihan import Latihan

latihan = Latihan(df=df, df2=df2, X=X, X_train=X_train, X_test=X_test)

latihan.lat_one()
latihan.lat_two()
latihan.lat_three()