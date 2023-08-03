import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
df = pd.read_csv('C:/Users/91959/Downloads/archive (1)/Housing.csv)
print(df.shape)
df.sample(5)

print(df.columns)  

pd.options.display.max_columns=None
df.info()

df.plot('area', 'price', kind='scatter')

Category_features = ['mainroad','guestroom', 'basement', 'hotwaterheating', 
            'airconditioning', 'prefarea', 'furnishingstatus']
Numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

for feature in df.columns:
    print(feature, " : ", df[feature].unique())

    fig, axes = plt.subplots(2, 4, figsize=(20,10))
plt.title("Category_feature")
for k in range(len(Category_features)):
  num = []
  for t in df[Category_features[k]].unique():
    num.append(df[Category_features[k]].tolist().count(t))
  axes[k//4][k%4].pie(num, labels=df[Category_features[k]].unique(), autopct="%.2f%%", labeldistance=1.15, 
            wedgeprops = {'linewidth':1, 'edgecolor':'white'}, textprops={'color':'lightgreen', 'fontsize':15}, 
            colors=sns.color_palette('Blues_r'))
  axes[k//4][k%4].set_title(Category_features[k])
axes[-1][-1].axis('off')
plt.tight_layout()
plt.show()

import random

random_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(5)]

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)
for k in range(len(Numeric_features)-1):
  ax=fig.add_subplot(gs[k//3, k%3])
  sns.countplot(ax=ax, data=df, x=Numeric_features[k+1], palette=sns.color_palette('pastel'))
k+=1
sns.countplot(ax=fig.add_subplot(gs[k//3, k%3:]), data=df, x=Numeric_features[0], palette=sns.color_palette('pastel'))
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)
for k in range(len(Numeric_features)-1):
  ax=fig.add_subplot(gs[k//3, k%3])
  sns.countplot(ax=ax, data=df, x=Numeric_features[k+1], palette=sns.color_palette('pastel'))
  for label in ax.containers:
    ax.bar_label(label)
k+=1
ax = sns.countplot(ax=fig.add_subplot(gs[k//3, k%3:]), data=df, x=Numeric_features[0], palette=sns.color_palette('pastel'))
for label in ax.containers:
    ax.bar_label(label)
plt.tight_layout()

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(nrows=2, ncols=3)
for k in range(len(Numeric_features)-1):
  ax=fig.add_subplot(gs[k//3, k%3])
  sns.histplot(ax=ax, data=df, x=Numeric_features[k+1], discrete=True, stat="percent")
k+=1
ax0 = fig.add_subplot(gs[k//3, k%3:])
sns.histplot(ax=ax0, data=df, x=Numeric_features[0], discrete=True, stat="percent")
ax0.set_ylim(0,1.75)
plt.tight_layout()
plt.show()

sns.set_style("darkgrid")

plt.figure(figsize=(5,8))
plt.boxplot(x=df['price'], notch=True)
plt.ylabel('price')

fig, axes = plt.subplots(1, 3, figsize=(5,7))
sns.boxplot(ax=axes[0], data=df, y="price")
sns.boxplot(ax=axes[1], data=df, y="price", showcaps=False, 
        whiskerprops={"linestyle": 'dashed', "lw":4})
sns.boxplot(ax=axes[2], data=df, y="price", notch=True, showmeans=True, meanline=True, 
        meanprops={"color": "r", "lw":2}, medianprops={"color": "c", "lw":3})
plt.tight_layout()
plt.show()

print(df.shape)
r = 3
z_score = (df['area'] - df['area'].mean())/df['area'].std()
df = df[ (z_score > (-1)*r) & (z_score < r) ]
print(df.shape)

r = 3
lower_limit = df['area'].mean() - r*df['area'].std()
upper_limit = df['area'].mean() + r*df['area'].std()  
print(df.shape)
df = df[ (df['area'] >= lower_limit) & (df['area'] <= upper_limit) ]
print(df.shape)

import numpy as np
import pandas as pd

ex_df = pd.DataFrame({
    'feature_name' : ['square', np.nan, 'oval', 'square', 'circle', np.nan, 'triangle'],
    'feature_name2' : [1, np.nan, 3, 4, 5, np.nan, 7],
    'feature_name3' : ['squares', 'triangles', np.nan, 'circles', 'ovals', np.nan, 'squares'],
})
ex_df

ex_df.isnull().sum()

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

Models=[
    ["Linear Regression", LinearRegression()],
    ["Decision Tree Regressor", DecisionTreeRegressor()],
    ["RandomForestRegressor", RandomForestRegressor()],
    ["Gradient Boosting Regressor", GradientBoostingRegressor()],
]

print(ex_df)
