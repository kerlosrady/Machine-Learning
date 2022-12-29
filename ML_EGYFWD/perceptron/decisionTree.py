from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

df=pd.read_csv('DT.csv')
X=df.iloc[:,:-1]
Y=df.iloc[:,-1:]
print(Y)
model=DecisionTreeClassifier()
model.fit(X,Y)
print(model.predict([ [0.2, 0.8], [0.5, 0.4] ]))
y_pred=model.predict(X)
print(accuracy_score(Y,y_pred))