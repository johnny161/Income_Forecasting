# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
         'native-country', 'income']
test_data = pd.read_csv("data/adult_test.csv", skiprows=1, names=names)
train_data = pd.read_csv("data/adult_train.csv", names=names)
test_data['income'][(test_data['income']==' <=50K') | (test_data['income']==' <=50K.')] = 0
test_data['income'][(test_data['income']==' >50K') | (test_data['income']==' >50K.')] = 1
train_data['income'][(train_data['income']==' <=50K') | (train_data['income']==' <=50K.')] = 0
train_data['income'][(train_data['income']==' >50K') | (train_data['income']==' >50K.')] = 1
test_data.replace(to_replace=" ?", value=np.NaN, inplace=True)
train_data.replace(to_replace=" ?", value=np.NaN, inplace=True)

#test_data = test_data.fillna(0)
#train_data = train_data.fillna(0)
train_data["workclass"] = train_data["workclass"].fillna("Private")
train_data["occupation"] = train_data["occupation"].fillna("Prof-specialty")
train_data["native-country"] = train_data["native-country"].fillna("United-States")
test_data["workclass"] = test_data["workclass"].fillna("Private")
test_data["occupation"] = test_data["occupation"].fillna("Prof-specialty")
test_data["native-country"] = test_data["native-country"].fillna("United-States")

X_train, X_validation, y_train, y_validation = train_test_split(train_data.iloc[:,:-1],train_data.iloc[:,-1],test_size=0.3,random_state=1234)
X_test, y_test = test_data.iloc[:,:-1], test_data.iloc[:,-1]
# print(train_data.info())

categorical_features_indices = np.where(X_train.dtypes == np.object)[0]
model = CatBoostClassifier(cat_features=categorical_features_indices)
model.fit(X_train,y_train,eval_set=(X_validation, y_validation),plot=False)

fea_ = model.feature_importances_
fea_name = model.feature_names_
plt.figure(figsize=(12, 10))
plt.barh(fea_name,fea_,height =0.5)
plt.show()

print("accuracy on the training subset:{:.4f}".format(model.score(X_train,y_train)))
print("accuracy on the test subset:{:.4f}".format(model.score(X_test,y_test)))

# fillna 0 | mode (is same)
'''
bestTest = 0.2871485281
bestIteration = 409

Shrink model to first 410 iterations.
accuracy on the training subset:0.8940
accuracy on the test subset:0.8728
'''

