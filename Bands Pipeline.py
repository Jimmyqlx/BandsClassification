import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
#Read Data
df = pd.read_csv('bands.data',header=None)
#Column indices for categorical data
cat_ind=list(range(1, 20))
#Column indices for numerical data
num_ind=list(range(20, 39))
#fill '?' missing data with NaN
df[cat_ind]=df[cat_ind].replace(to_replace ='?', value=np.nan)
df[num_ind]=df[num_ind].replace(to_replace ='?', value=np.nan)

#convert data type to float and category
le = preprocessing.LabelEncoder()
df[39]= df[39].astype('category')
df[cat_ind]= df[cat_ind].astype('category')
df[num_ind]= df[num_ind].astype('float')

#convert band label to 0 and 1
df[39]=le.fit_transform(df[39])

#get x train data and y label
train_ind=list(range(0,39))
X=df.iloc[:,train_ind]
y=df.iloc[:,39]

#numerical data pipeline
numeric_features = num_ind
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

#categorical data pipeline
categorical_features = cat_ind
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#data transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Append classifier to preprocessing pipeline.
# full prediction pipeline.
#GradientBoosting
clf = Pipeline(steps=[('preprocessor', preprocessor),
('selector', SelectKBest(mutual_info_classif, k=5)),
                      ('classifier', GradientBoostingClassifier(random_state=0,learning_rate=0.05,max_depth=4))])
l=cross_val_score(clf, X, y, cv=10)
print(np.mean(l))
print(l)


#Logistic Regression
clf = Pipeline(steps=[('preprocessor', preprocessor),
('selector', SelectKBest(mutual_info_classif, k=5)),
                      ('classifier', LogisticRegression(random_state=0,C=0.001,max_iter=1000))])
l=cross_val_score(clf, X, y, cv=10)
print(np.mean(l))
print(l)

#Randomforest
clf = Pipeline(steps=[('preprocessor', preprocessor),
('selector', SelectKBest(mutual_info_classif, k=5)),
                      ('classifier', RandomForestClassifier(random_state=0))])
l=cross_val_score(clf, X, y, cv=10)
print(np.mean(l))
print(l)

#DecisionTree
clf = Pipeline(steps=[('preprocessor', preprocessor),
('selector', SelectKBest(mutual_info_classif, k=5)),
                      ('classifier', DecisionTreeClassifier(random_state=0,max_depth=4))])
l=cross_val_score(clf, X, y, cv=10)
print(np.mean(l))
print(l)