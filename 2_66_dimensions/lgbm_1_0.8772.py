import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

from data_processing import prepare


if __name__=="__main__":
    train, test, train_label, test_label = prepare()

    ##### ==step1====================================================
    params = {    
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric': 'auc',
          'nthread':4,
          'learning_rate':0.1,
          'num_leaves':30, 
          'max_depth': 5,   
          'subsample': 0.8, 
          'colsample_bytree': 0.8,
    }

    #data_train = lgb.Dataset(train, train_label)
    #cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',early_stopping_rounds=50,seed=0)
    #print('best n_estimators:', len(cv_results['auc-mean']))
    #print('best cv score:', pd.Series(cv_results['auc-mean']).max())

    ##### best n_estimators: 190
    ##### best cv score: 0.9290793974831517
    ##### ==step1====================================================



    ##### ==step2====================================================确定max_depth和num_leaves
    #params_test1={'max_depth': range(3,8,1), 'num_leaves':range(5, 100, 5)}     
    #    
    #gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=190, max_depth=6, bagging_fraction = 0.8,feature_fraction = 0.8), 
    #                       param_grid = params_test1, scoring='roc_auc',cv=5,n_jobs=4)
    #gsearch1.fit(train,train_label)
    #print(gsearch1.best_params_, gsearch1.best_score_)

    ##### {'max_depth': 6, 'num_leaves': 20} 0.9286963291603303
    ##### ==step2====================================================



    ##### ==step3====================================================确定min_data_in_leaf和max_bin in
    #params_test2={'max_bin': range(5,256,10), 'min_data_in_leaf':range(1,102,10)}
              
    #gsearch2 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=190, max_depth=6, num_leaves=20,bagging_fraction = 0.8,feature_fraction = 0.8), 
    #                       param_grid = params_test2, scoring='roc_auc',cv=5,n_jobs=4)
    #gsearch2.fit(train,train_label)
    #print(gsearch2.best_params_, gsearch2.best_score_)

    ##### {'max_bin': 225, 'min_data_in_leaf': 11} 0.9294057258756219
    ##### ==step3====================================================



    ##### ==step4====================================================确定feature_fraction、bagging_fraction、bagging_freq
    #params_test3={'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
    #          'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
    #          'bagging_freq': range(0,81,10)
    #}
              
    #gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=190, max_depth=6, num_leaves=20,max_bin=225,min_data_in_leaf=11), 
    #                       param_grid = params_test3, scoring='roc_auc',cv=5,n_jobs=4)
    #gsearch3.fit(train,train_label)
    #print(gsearch3.best_params_, gsearch3.best_score_)

    ##### {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.6} 0.9298165445431701
    ##### ==step4====================================================



    ##### ==step5====================================================确定lambda_l1和lambda_l2
    #params_test4={'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
    #          'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
    #}
              
    #gsearch4 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=190, max_depth=6, num_leaves=20,max_bin=225,min_data_in_leaf=11,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.6), 
    #                       param_grid = params_test4, scoring='roc_auc',cv=5,n_jobs=4)
    #gsearch4.fit(train,train_label)
    #print(gsearch4.best_params_, gsearch4.best_score_)

    ##### {'lambda_l1': 0.001, 'lambda_l2': 0.3} 0.9299647609906201
    ##### ==step5====================================================



    ##### ==step6====================================================确定 min_split_gain 
    #params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
              
    #gsearch5 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=190, max_depth=6, num_leaves=20,max_bin=225,min_data_in_leaf=11,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.6,
    #lambda_l1=0.001,lambda_l2=0.3), 
    #                       param_grid = params_test5, scoring='roc_auc',cv=5,n_jobs=4)
    #gsearch5.fit(train,train_label)
    #print(gsearch5.best_params_, gsearch5.best_score_)

    ##### {'min_split_gain': 0.0} 0.9299647609906201
    ##### ==step6====================================================



    ##### ==step7====================================================降低学习率，增加迭代次数，验证模型
    model=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=6, num_leaves=20,max_bin=225,min_data_in_leaf=11,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.6,
    lambda_l1=0.001,lambda_l2=0.3,min_split_gain=0)
    model.fit(train,train_label)
    y_pre=model.predict(test)
    print("acc:",accuracy_score(test_label,y_pre))

    # plot confusion_matrix & classification_report
    print(confusion_matrix(test_label,y_pre))
    print(classification_report(test_label,y_pre))

    ##### acc: 0.8771574227627296
    ##### ==step7====================================================