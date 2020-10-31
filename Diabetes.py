#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:56:24 2020

@author: shehnazislam
"""

import numpy as np # mathematical operations and algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt # visualization library
import seaborn as sns # Fancier visualizations
import statistics # fundamental stats package
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import scipy.stats as stats # to calculate chi-square test stat


# To view full dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

# read data and add the column names
rawData = pd.read_excel('diabetes.xlsx', names=['Age','Gender', 
                    'Polyuria', 'Polydipsia', 'SuddenWeightLoss', 'Weakness', 'Polyphagia', 'GenitalThrush', 
                    'VisualBlurring', 'Itching', 'Irritability', 'DelayedHealing', 'PartialParesis', 'MuscleStiffness', 
                    'Alopecia', 'Obesity', 'Class'])

rawData.info()

# view the first 5 rows, "head" is the function that return the first "n" rows for the object based on position, by default, "5"
# Notes: index start from "0"
rawData.head()

# last 5 rows
rawData.tail()

# show all
print(rawData)


diabetes = rawData
# gerneral info
print('---------------- Count ----------------------\n')
print(diabetes.count()) # no missing values(blank/NA)

# for numerical variables
# use dataframe.function separately
print('\n---------------- Age  ----------------------')
print('Mean')
print(diabetes['Age'].mean())
print('\nMedian')
print(diabetes['Age'].median())
print('\nMax')
print(diabetes['Age'].max())
print('\nMin')
print(diabetes['Age'].min())
print('\nStandard Deviation')
print(diabetes['Age'].std())

# check the datatype
diabetes.dtypes


# but attribute "Age" are numerical variables?
print('\n--------Age------------')
print(diabetes['Age'].unique())
print('\n')
print(diabetes['Age'].value_counts())
diabetes['Age'].value_counts().plot(kind='bar')
plt.show()
diabetes['Age'].hist(figsize=(15, 5), bins=10) # histgram indicate that there may be a "problem" 
plt.show()


diabetes.info()


# re-do five number summary using above functions

# or use describe function
diabetes.describe().unstack()
# you may include parameter " include='all' "": means include all numerical and categorical variables
# german.describe(include='all').unstack() 


# numuric attributes: 
diabetes.plot(kind='box', subplots=True, layout=(1,3), sharex=False, sharey=False, figsize=(20,6))
plt.show()
# what do you observe? 2 Outliers, slighly dense towards lower values so positively skewed

# centre tendency, Median = 47.5
print(np.quantile(diabetes["Age"], 0.5))

# spread,  Q1 = 39 , Q3= 57, IQR= 18
age_Q1=np.quantile(diabetes["Age"], 0.25)
print(np.quantile(diabetes["Age"], 0.25))

age_Q3=np.quantile(diabetes["Age"], 0.75)
print(np.quantile(diabetes["Age"], 0.75))

age_IQR= age_Q3-age_Q1
print(age_IQR)

# skewness
# outlier, we see two outliers at the higher age values, roughly 80 and 90 years old

# numuric attributes: 
diabetes.hist(layout=(1,3), sharex=False, sharey=False, figsize=(15, 5), bins=10)
plt.show()



#225 in German Data set

# note: A symmetrical distribution will have a skewness of 0.
# Skewness is a measure of symmetry, or more precisely, the lack of symmetry
print('-----------Skewness-------------')
print(diabetes.skew(axis = 0, skipna = True))  # Va;ue of skewness only 0.33 which is low and so we can say the distribution of age is symettric.

# Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution. 
# That is, data sets with high kurtosis tend to have heavy tails, or outliers. Data sets with low kurtosis tend to have light tails, or lack of outliers. 
print('\n-----------Kurtosis-------------')
print(diabetes.kurtosis(skipna = True)) # Va;ue of skewness only -0.19 which is also low and so we can say the distribution does not have heavy tails or outliers.



# summary of non-numerical variables
diabetes.describe(exclude=[np.number]).unstack()

#Bar chart and Pie chart for Gender non-numeric attributes

print('\n---------- Gender ------------')
print(diabetes['Gender'].unique())
print('\n')
print(diabetes['Gender'].value_counts())
counts = diabetes['Gender'].value_counts()
percentage = diabetes['Gender'].value_counts(normalize=True)
print('\n')
print(pd.concat([counts,percentage], axis=1, keys=['counts', '%']))

# draw bar bar
diabetes['Gender'].value_counts().plot(kind='bar')
plt.show()
# or pie char
diabetes['Gender'].value_counts().plot(kind='pie')
plt.show()


#Bar chart and Pie chart for all 16 non-numeric attributes using for loop

list=['Gender','Polyuria', 'Polydipsia', 'SuddenWeightLoss', 'Weakness', 'Polyphagia', 'GenitalThrush', 
                    'VisualBlurring', 'Itching', 'Irritability', 'DelayedHealing', 'PartialParesis', 'MuscleStiffness', 
                    'Alopecia', 'Obesity', 'Class']

for x in list:

    print('\n---------- ' + x + '------------')
    print(diabetes[x].unique())
    print('\n')
    print(diabetes[x].value_counts())
    counts = diabetes[x].value_counts()
    percentage = diabetes[x].value_counts(normalize='true')
    print('\n')
    print(pd.concat([counts,percentage], axis=1, keys=['counts', '%']))
    
    # draw bar bar
    diabetes[x].value_counts().plot(kind='bar')
    plt.show()
    # or pie char
    diabetes[x].value_counts().plot(kind='pie')
    plt.show()


#Line 686 # NoDependents 

# Crosstab or contingency or frquency table between class atribute 'Gender' 

print('--------------------------------------Gender---------------------------------\n')
print(pd.crosstab(diabetes['Class'], diabetes['Gender'])) 
print('\n')
print(stats.chi2_contingency(pd.crosstab(diabetes['Class'], diabetes['Gender'])))


p_value_list= []
# Crosstab or contingency or frquency table between class atribute 'class' and all other 16 non-numeric attribtess
for x in list:
    print('--------------------------------------'+ x + '---------------------------------\n')
    print(pd.crosstab(diabetes['Class'], diabetes[x])) 
    print('\n')
    print(stats.chi2_contingency(pd.crosstab(diabetes['Class'], diabetes[x])))
    t_test_results= stats.chi2_contingency(pd.crosstab(diabetes['Class'], diabetes[x]))
    print(t_test_results[1])
    
    p_value_list.append(t_test_results[1])

print(p_value_list)   

    
for x, y in zip(list, p_value_list):
    print(x, y, sep='\t\t\t')
    
    
    
# 5% significant level 
# Gender			3.289703730553317e-24
# Polyuria			1.7409117803442155e-51
# Polydipsia			6.1870096408863144e-49
# SuddenWeightLoss			5.969166262549937e-23
# Weakness			4.869843446585542e-08
# Polyphagia			1.1651584346409174e-14
# GenitalThrush			0.016097902991938178
# VisualBlurring			1.7015036753241226e-08
# Irritability			1.7714831493959365e-11
# PartialParesis			1.565289071056334e-22
# MuscleStiffness			0.006939095697923978
# Alopecia			1.9092794963634e-09


# Attributes not correlated to class

# Itching			0.8297483959485009
# DelayedHealing			0.32665993771439944
# Obesity			0.12710799319896815



