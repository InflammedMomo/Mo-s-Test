import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bokeh.palettes import RdBu3
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("Project_1_Data.csv")
print(df.info())
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
sns.heatmap(f)
#plt.show()

