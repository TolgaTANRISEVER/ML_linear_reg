# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("linear_regression_dataset.csv",sep=";")
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% sklearn

from sklearn.linear_model import LinearRegression

linear_reg=LinearRegression()

x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)


#%% preidction

b0=linear_reg.predict([[0]])

print("b0: ",b0)

b0_=linear_reg.intercept_#keşisim noktası x-y
print(("b0_:"),b0_)

b1=linear_reg.coef_
print("b1:", b1)# eğim

maas_yeni =b0+b1*11
print(maas_yeni)

import numpy as np
print(linear_reg.predict([[11]]))

array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)

plt.scatter(x,y)
plt.show
y_head = linear_reg.predict(array)
plt.plot(array,y_head,color="red")














