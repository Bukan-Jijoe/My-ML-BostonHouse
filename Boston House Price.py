#%%
#Import
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.datasets import load_diabetes 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# %%
def cramers_corrected_stat(confusion_matrix):
    """"calculate Cramers V statistic for categorical-categorical data association"""
    chi2=ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0,phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k- ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1),(rcorr-1)))
#%%
os.chdir

CSV_PATH = os.path.join(os.getcwd(),'Dataset','housing.csv')

Model_path = os.path.join(os.getcwd(),'Model','model.pkl')

# %%
column_name = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B 1000','LSTAT','MEDV']
df = pd.read_csv(CSV_PATH,header=None, delimiter="\s+",names=column_name)

# %%
df.head(20)
# %%
df.describe() #finding the outliers
# %%
df.info() 
# %%
df
# %%

# %%
print((df["ZN"] == 0).sum())
print((df["CHAS"] == 0).sum())
# %%
cat = ["CHAS"]
con = df.drop(labels=cat, axis=1).columns

# %% replacing the zero value with null value
for x in con:
    df[x] = df[x].replace(0,np.nan)

# %%
for x in con:
    plt.figure()
    sns.histplot(df[x])
    plt.show
# %%
for x in cat:
    plt.figure()
    sns.countplot(df[x])
    plt.show
# %%
df.boxplot(figsize=(8,6), fontsize=8.0,rot=90)
#to find the outlies based on boxplot
#%%  plot continuous data
for x in con:
    plt.figure()
    sns.displot(df[x], kde=True)
    plt.show

#%% Data Cleaning
import missingno as msno
msno.matrix(df)
msno.bar(df)

#%%  Fill in the missing value
df['ZN'] = df['ZN'].fillna(df['ZN'].median())

# %% Feature Selection
# to find the Correlation between continuous data and continuous target
# thus use regression analysis
for x in con:
    print(x)
    lr = LinearRegression()
    lr.fit(np.expand_dims(df[x], axis=-1), df['MEDV'])
    print(lr.score(np.expand_dims(df[x], axis=-1), df['MEDV']))

# %%
# to find the correlation between categorical data and continuous target
for x in cat:
    print(x)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df['MEDV'], axis=-1),df[x])
    print(lr.score(np.expand_dims(df['MEDV'], axis=-1), df[x],))


#%% Data Preprocessing

X = df.loc[:, df.drop(['MEDV'],axis=1).columns.to_list()]
y = df['MEDV']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42 )

#%% MODEL DEVELOPMENT -----> Pipeline

#Linear Regression

pipeline_mms_linr = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Linear_Regression", LinearRegression())
]) # Pipeline Steps

pipeline_ss_linr = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Linear_Regression", LinearRegression())
]) # Pipeline Steps

# Decision Tree Regressor
pipeline_mms_dt = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Decision_Tree", DecisionTreeRegressor())
]) # Pipeline Steps

pipeline_ss_dt = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Decision_Tree", DecisionTreeRegressor())
]) # Pipeline Steps

#Random Forest Regressor

pipeline_mms_rf = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Random_Forest", RandomForestRegressor())
]) # Pipeline Steps

pipeline_ss_rf = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Random_Forest", RandomForestRegressor())
]) # Pipeline Steps

# #GradientBoost Regressor
pipeline_mms_gb = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Gradient_Boost", GradientBoostingRegressor())
]) # Pipeline Steps

pipeline_ss_gb = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Gradient_Boost", GradientBoostingRegressor())
]) # Pipeline Steps

#Ridge
pipeline_mms_rd = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Ridge", Ridge())
]) # Pipeline Steps

pipeline_ss_rd = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Ridge", Ridge())
]) # Pipeline Steps

#Lasso
pipeline_mms_ls = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Lasso", Lasso())
]) # Pipeline Steps

pipeline_ss_ls = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("Lasso", Lasso())
]) # Pipeline Steps

#SVR
pipeline_mms_svr = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("SVR", SVR())
]) # Pipeline Steps

pipeline_ss_svr = Pipeline([
    ("Standard_Scaler", StandardScaler()),
    ("SVR", SVR())
]) # Pipeline Steps

#a list to store all the pipelines

pipelines = [pipeline_mms_linr,pipeline_ss_linr,pipeline_mms_rd,pipeline_ss_rd, 
             pipeline_mms_ls,pipeline_ss_ls,pipeline_ss_svr,pipeline_mms_svr,
             pipeline_mms_dt,pipeline_ss_dt,pipeline_ss_rf,pipeline_mms_rf,
             pipeline_mms_gb,pipeline_ss_gb]
#%%
for pipe in pipelines:
    pipe.fit(X_train,y_train)

pipe_scores = []

for i,pipe in enumerate(pipelines):
    y_pred = pipe.predict(X_test)
    print(pipe)
    print('RMSE:{}'.format(root_mean_squared_error(y_test,y_pred)))
    print('R2:{}'.format(r2_score(y_test,y_pred)))
    print('MAE:{}'.format(mean_absolute_error(y_test,y_pred)))
    print('\n')
    score = pipe.score(X_test, y_test)
    pipe_scores.append(score)



pipe_score = max(pipe_scores)
print(f"The best score is: {pipe_score}")
pipe_scalar = pipelines[pipe_scores.index(pipe_score)]
print(f"The best pipeline combination is: {(pipe_scalar)}")
print(type(pipe_scalar))

# %%
# %% Hyperparameter Tuning & Grid Search CV

pipeline_mms_rf = Pipeline([
    ("Min_Max_Scaler", MinMaxScaler()),
    ("Gradient_Boost", GradientBoostingRegressor())
]) # Pipeline Steps

# Hyperparameter Tuning
grid_param = [{'Gradient_Boost__n_estimators':[100,200],
               'Gradient_Boost__learning_rate':[0.1],
               'Gradient_Boost__min_samples_split':[5],
               'Gradient_Boost__min_samples_leaf':[2],
               'Gradient_Boost__alpha':[0.9]
               }]

gridsearch = GridSearchCV(pipeline_mms_rf, grid_param, cv = 10, verbose = 1, n_jobs= -1)
grid = gridsearch.fit(X_train,y_train)
gridsearch.score(X_test, y_test)
print(grid.best_params_)
print(grid.best_estimator_)
print(type(grid.best_estimator_))
print(grid.best_score_)


# in this case, without the hyperparameter tuning, the model have better accuracy compared tuned with this criteria
best_model = pipe_scalar

# %%
import pickle
with open (Model_path,"wb") as file:
    pickle.dump(best_model,file)
# %%