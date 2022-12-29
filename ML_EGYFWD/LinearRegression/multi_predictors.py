from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
df=pd.read_csv('data3.csv')
X=df[['Var_X']][1:]
Y=df[['Var_Y']][1:]
model=LinearRegression(fit_intercept=False)
poly_feat=PolynomialFeatures(degree=4)
X_poly=poly_feat.fit_transform(X)
model.fit(X_poly,Y)
