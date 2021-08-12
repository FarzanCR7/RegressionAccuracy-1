# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 02:40:15 2021

@author: FARZAN
"""

#Importing the libraries for the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
#Importing the Dataset
dataset = pd.read_csv("Level_DataSet.csv")
x = dataset.iloc[:,0:1].values
y= dataset.iloc[:,1].values
#%%
#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
print(lin_reg.score(x,y)*100)
#%%
# Fitting the Polynomial Regression to the Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)
print(lin_reg_2.score(x_poly,y)*100)
#%%
# Visualising the linear regression results
plt.scatter(x, y, color= "Black")
plt.plot(x, lin_reg.predict(x), color = "red")
plt.title("Linear Regression")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
#%%
#Visualing the Polynomail regression result
plt.scatter(x, y, color= "Grey")
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title("(Polynomial Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
