# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:37:16 2019

@author: sebas
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import burrito taco data
tacos = pd.read_csv('tacosburritos.csv')

# Define variables of interest from for household survey
varshusa = ['RT', 'SERIALNO', 'DIVISION', 'PUMA', 'REGION', 'ST', 'NP', 'VALP',
        'RMSP','TEN', 'FES', 'HINCP', 'HUPAC', 'HHL', 'HHT', 'R18', 'R60', 
        'R65', 'SSMC', 'WORKSTAT']
# Import census data
husa = pd.read_csv('psam_husa.csv', dtype=object, usecols=varshusa)
husb = pd.read_csv('psam_husb.csv', dtype=object, usecols=varshusa)
# Merge datasets
hus = pd.concat([husa, husb])
# Restyle numerical variables
hus['NP'] = pd.to_numeric(hus['NP'])
hus['VALP'] = pd.to_numeric(hus['VALP'])
hus['RMSP'] = pd.to_numeric(hus['RMSP'])
hus['HINCP'] = pd.to_numeric(hus['HINCP'])

#Create median data by PUMA
medianhus = hus.groupby('PUMA').median()
medianhus.to_csv('medianhus.csv')
medianhus = pd.read_csv('medianhus.csv',dtype = {'PUMA': str,'NP': float,
           'RMSP': float,'VALP': float})

# Match to ZIP code
pumazip = pd.read_csv('geocorr3.csv', encoding='latin-1')
pumazip = pumazip.iloc[1:]
pumazip['pop10'] = pd.to_numeric(pumazip['pop10'])
pumazip['afact'] = pd.to_numeric(pumazip['afact'])
pumazip.rename(columns={'puma12':'PUMA'}, inplace=True)
censusdata = pd.merge(medianhus, pumazip, on='PUMA')
censusdata.rename(columns={'zcta5':'postalCode'}, inplace=True)
a = censusdata[['postalCode','pop10']]
a = a.groupby('postalCode').sum()
a.rename(columns={'pop10':'popsum'}, inplace=True)
a.to_csv('a.csv')
a = pd.read_csv('a.csv',dtype = {'postalCode': str,'pop10': float})
census = pd.merge(censusdata, a, on='postalCode')
census.to_csv('census10.csv')
census = census[['postalCode','afact', 'popsum', 'NP','RMSP', 'VALP', 'HINCP']]
census['popsum'] = census['afact']*census['popsum'] 
census['NP'] = census['afact']*census['NP'] 
census['RMSP'] = census['afact']*census['RMSP'] 
census['VALP'] = census['afact']*census['VALP'] 
census['HINCP'] = census['afact']*census['HINCP'] 
census = census.groupby('postalCode').sum()
census.to_csv('b.csv')
census = pd.read_csv('b.csv',dtype = {'postalCode': str})
datasetA = tacos.groupby('postalCode')['id'].nunique()
datasetA.to_csv('c.csv')
datasetA = pd.read_csv('c.csv',dtype = {'postalCode': str})
datasetA = datasetA.iloc[11:]
datasetA.columns = ['postalCode', 'RestCount'] 


# Create dataset for analysis
dataset = pd.merge(datasetA, census, on='postalCode')
dataset.to_csv('dataset10.csv')

#Visualizing Data for determining which model to use (linear)
datasetviz=dataset[['RestCount','HINCP']]

plt.scatter(datasetviz['HINCP'],datasetviz['RestCount'])
plt.title('Restaurant Count vs. Median Household Income by ZIP Code')
plt.show()

# Split for regression fit

X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

#Splitting dataset into training and test sets
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#Fitting MLR to training set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train.reshape(-1,1), y_train)

#Fitting polynomial reg
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X.reshape(-1,1))
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



#Predicting test set results
y_pred_lin = lin_reg.predict(X_test.reshape(-1,1))
y_pred_pol = lin_reg_2.predict(X_poly)


# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X.reshape(-1,1)), color = 'blue')
plt.title('Restaurant Count vs. Median Household Income by ZIP Code (Linear Regression)')
plt.xlabel('Median Household Income')
plt.ylabel('Restaurant Count')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X.reshape(-1,1))), color = 'blue')
plt.title('Restaurant Count vs. Median Household Income by ZIP Code (Polynomial Regression)')
plt.xlabel('Median Household Income')
plt.ylabel('Restaurant Count')
plt.show()


