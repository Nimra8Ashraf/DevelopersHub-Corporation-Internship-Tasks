import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset

df=pd.read_csv("Iris.csv",encoding="latin1")
print(df)

# shape of dataset

shape=np.shape(df)
print("Shape of Dataset : ",shape)

# print columns

print(df.columns)

# .head()

print(df.head())

# info

print(df.info())

# describe

print(df.describe())

# scatter plot to show relationship b/w features of dataset

sns.pairplot(df, hue='Species',kind='scatter')
plt.show()

# box plot to identify outliers
sns.boxplot(df)
plt.show()

# droping column species

df2 = df.drop(['Species','Id'], axis=1)
print(df2)

# histogram to show value distributions

sns.histplot(df2,bins=15)
plt.show()

