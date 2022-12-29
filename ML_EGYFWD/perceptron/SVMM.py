from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

data=pd.read_csv('svm_data.csv',header=None)

x_values=data.iloc[:,:-1]
y_values=data.iloc[:,-1]

model=SVC(kernel='rbf',C=25,degree=5)
model.fit(x_values,y_values)
y_pred=model.predict(x_values)
print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
acc = accuracy_score(y_values,y_pred)
print(acc)
