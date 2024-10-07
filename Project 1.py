import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("Project_1_Data.csv")
df = df.dropna()
df = df.reset_index(drop=True)

X=df.loc[:,"X"]
Y=df.loc[:,"Y"]
Z=df.loc[:,"Z"]
step=df.loc[:,"Step"]
#print(step)

ax = plt.figure().add_subplot(projection='3d')
p=ax.scatter(X, Y, Z, c=step)

ax.set(
    xlabel='X [km]',
    ylabel='Y [km]',
    zlabel='Z [m]',
)
plt.colorbar(p)
plt.show()
