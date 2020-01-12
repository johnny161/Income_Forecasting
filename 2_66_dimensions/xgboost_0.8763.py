import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from data_processing import prepare

if __name__=="__main__":
    train, test, train_label, test_label = prepare()

    #param_test1 = {
    # 'max_depth':list(range(3,7,2)),
    # 'min_child_weight':list(range(2,4,1)),
    # 'colsample_bytree':[0.6, 0.7, 0.8, 0.9],
    #}
    #gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=20, max_depth=5,
    # min_child_weight=1, gamma=0, subsample=0.8,
    # objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
    # param_grid = param_test1, scoring='accuracy',n_jobs=4,iid=False, cv=5)
    #gsearch1.fit(train,train_label)
    #print(gsearch1.best_params_, gsearch1.best_score_)
    
    xgb = XGBClassifier(learning_rate=0.13,
                        max_depth=6,
                        min_child_weight=3,
                        colsample_bytree=0.6,
                        #subsample=0.8,
                        nthread=4,
                        gamma=0.02
                        )
    xgb.fit(train, train_label)
    y_pred = xgb.predict(test)
    accuracy = accuracy_score(y_pred, test_label)                 
    print(accuracy)