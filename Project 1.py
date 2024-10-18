import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, StackingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix
import joblib as job

df = pd.read_csv("Project_1_Data.csv")
#print(df.info())

# Question 1
X=df.loc[:,"X"]
Y=df.loc[:,"Y"]
Z=df.loc[:,"Z"]
step=df.loc[:,"Step"]


ax = plt.figure().add_subplot(projection='3d')
p=ax.scatter(X, Y, Z, c=step)

ax.set(
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
)
plt.colorbar(p, label="Step")
#plt.show() #rememeber to remove this
plt.clf()

# Question 2

splitter = StratifiedShuffleSplit(n_splits = 1,
                               test_size = 0.2,
                               random_state = 42)
for train_index, test_index in splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
#print(strat_df_train)

f_train = strat_df_train.drop("Step", axis = 1)
t_train = strat_df_train['Step']
f_test = strat_df_test.drop("Step", axis = 1)
t_test = strat_df_test['Step']

f = np.abs(strat_df_train.corr())
print(f)
sns.heatmap(f, annot=True)
plt.show()

#Question 3

#Linear Regression
linear_reg = LinearRegression()
param_grid_lr = {}
grid_search_lr = GridSearchCV(linear_reg, param_grid_lr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lr.fit(f_train, t_train)
best_model_lr = grid_search_lr.best_estimator_

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

# Random Forest
random_forest = RandomForestRegressor(random_state=48)
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


# Random Forest + randomized
random_forest_rand = RandomForestRegressor(random_state=64)
param_grid_rf_rand = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rf_rand = RandomizedSearchCV(random_forest_rand, param_grid_rf_rand, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_rf_rand.fit(f_train, t_train)
best_model_rf_rand = grid_search_rf_rand.best_estimator_



# Question 4 + 5

# Linear Regression
t_test_pred_lr = np.round(best_model_lr.predict(f_test))
f1_lr = sklearn.metrics.f1_score(t_test_pred_lr, t_test, average='macro')
p_s_lr = sklearn.metrics.precision_score(t_test_pred_lr, t_test, average='macro')
a_s_lr = sklearn.metrics.accuracy_score(t_test_pred_lr, t_test)
con_mat_lr = sklearn.metrics.confusion_matrix(t_test_pred_lr, t_test)
print('Linear Regression')
print('F1 Score:', f1_lr)
print('Precision:', p_s_lr)
print('Accuracy:', a_s_lr)
#print(con_mat_lr)

# Decision Tree
t_test_pred_dt = np.round(best_model_dt.predict(f_test))
f1_dt = sklearn.metrics.f1_score(t_test_pred_dt, t_test, average='macro')
p_s_dt = sklearn.metrics.precision_score(t_test_pred_dt, t_test, average='macro')
a_s_dt = sklearn.metrics.accuracy_score(t_test_pred_dt, t_test)
con_mat_dt = sklearn.metrics.confusion_matrix(t_test_pred_dt, t_test)
print('Decision Tree')
print('F1 Score:', f1_dt)
print('Precision:', p_s_dt)
print('Accuracy:',a_s_dt)
#print(con_mat_dt)

# Random Forest
t_test_pred_rf = np.round(best_model_rf.predict(f_test))
f1_rf = sklearn.metrics.f1_score(t_test_pred_rf, t_test, average='macro')
p_s_rf = sklearn.metrics.precision_score(t_test_pred_rf, t_test, average='macro')
a_s_rf = sklearn.metrics.accuracy_score(t_test_pred_rf, t_test)
con_mat_rf = sklearn.metrics.confusion_matrix(t_test_pred_rf, t_test,normalize=None)
print('Random Forest')
print('F1 Score:', f1_rf)
print('Precision:', p_s_rf)
print('Accuracy:',a_s_rf)
#print(con_mat_rf)

# Random Forest +rand
t_test_pred_rf_rand = np.round(best_model_rf_rand.predict(f_test))
f1_rf_rand = sklearn.metrics.f1_score(t_test_pred_rf_rand, t_test, average='macro')
p_s_rf_rand = sklearn.metrics.precision_score(t_test_pred_rf_rand, t_test, average='macro')
a_s_rf_rand = sklearn.metrics.accuracy_score(t_test_pred_rf_rand, t_test)
con_mat_rf_rand = sklearn.metrics.confusion_matrix(t_test_pred_rf_rand, t_test,normalize=None)
print('Random Forest & RandomCV')
print('F1 Score:', f1_rf)
print('Precision:', p_s_rf)
print('Accuracy:',a_s_rf_rand)
#print(con_mat_rf_rand)

# Question 6
est = [('dt',decision_tree), ('rf_rand',random_forest)]
stack = StackingClassifier(estimators = est, final_estimator= LogisticRegression(max_iter=1000))
stack.fit(f_train,t_train)
t_test_pred_stack = stack.predict(f_test)
f1_stack = sklearn.metrics.f1_score(t_test_pred_stack, t_test, average='macro')
p_s_stack = sklearn.metrics.precision_score(t_test_pred_stack, t_test, average='macro')
a_s_stack = sklearn.metrics.accuracy_score(t_test_pred_stack, t_test)
con_mat_stack = sklearn.metrics.confusion_matrix(t_test_pred_stack, t_test)
print('Stacking Classifier')
print('F1 Score:', f1_stack)
print('Precision:', p_s_stack)
print('Accuracy:', a_s_stack)
#print(con_mat_stack)

# Question 7
rand_set1= [[9.375, 3.0625,1.51],
            [6.995,5.125,0.3875],
            [0,3.0625,1.93],
            [9.4,3,1.8],
            [9.4,3,1.3]]
job.dump(best_model_dt,'./models/decision_tree.pkl')
rt_model = job.load('./models/decision_tree.pkl')
test1 = rt_model.predict(rand_set1)
print(test1)



