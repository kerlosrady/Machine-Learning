from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
df=pd.read_csv('data2.csv')
# bmi_life_data=df['BMI']
# bmi_life_data.reshape(-1,1)
# y=df['Life expectancy']
model=LinearRegression()
model.fit(df[['BMI']],df[['Life expectancy']])
print(model.predict([[21.07931]]))


