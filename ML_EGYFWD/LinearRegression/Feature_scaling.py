from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import pandas as pd

df=pd.read_csv('data5.csv')
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

lasso_reg=Lasso()

lasso_reg.fit(X_scaled,Y)

reg_coef=lasso_reg.coef_
print(reg_coef)