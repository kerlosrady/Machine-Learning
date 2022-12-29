from sklearn.linear_model import LinearRegression,Lasso
import pandas as pd

df=pd.read_csv('data4.csv')
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
lasso_reg=Lasso()
lasso_reg.fit(X,Y)
print(lasso_reg.coef_)