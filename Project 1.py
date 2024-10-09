import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("Project_1_Data.csv")
#print(df.info())
# Question 1
X=df.loc[:,"X"]
Y=df.loc[:,"Y"]
Z=df.loc[:,"Z"]
step=df.loc[:,"Step"]
#print(step)

ax = plt.figure().add_subplot(projection='3d')
p=ax.scatter(X, Y, Z, c=step)

ax.set(
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
)
plt.colorbar(p)
#plt.show() #rememeber to remove this
print("hellow")

# Question 2

splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
#strat_df_train = strat_df_train.drop(columns=["Step"], axis = 1)
#strat_df_test = strat_df_test.drop(columns=["Step"], axis = 1)
#print(strat_df_train)

f_train = strat_df_train.drop("Step", axis = 1)
t_train = strat_df_train['Step']
f_test = strat_df_test.drop("Step", axis = 1)
t_test = strat_df_test['Step']
#print(f_train)

f=df.corr()
print(np.abs(f))
#sns.heatmap(f) ##Not working
#plt.show()

#Question 3
#Linear Regression
linear_reg = LinearRegression()
param_grid_lr = {}  # No hyperparameters to tune for plain linear regression, but you still apply GridSearchCV.
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(f_train, t_train)
best_model_lr = grid_search_lr.best_estimator_
print("Best Linear Regression Model:", best_model_lr)

# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(f_train, t_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)

# Random Forest
random_forest = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_rf.fit(f_train, t_train)
best_model_rf = grid_search_rf.best_estimator_
print("Best Random Forest Model:", best_model_rf)