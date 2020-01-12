# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang
@Reference:https://github.com/yanqiangmiffy/income-forecast/blob/master/utils.py
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from data_processing import prepare


def get_models():
    """
    生成机器学习库,此处没调参数                                   训练集验证精确度
    """
    nb = GaussianNB()                                           # 0.5986606780768458
    svc = SVC(C=100, probability=True)
    ln_svc = LinearSVC()
    knn = KNeighborsClassifier()                                # 0.8317927412987294
    lr = LogisticRegression()                                   # 0.8462886125311275
    nn = MLPClassifier(solver='sgd', alpha=0.01, batch_size=64) # 0.8542736724622951
    ab = AdaBoostClassifier()                                   # 0.8602009975512969
    gb = GradientBoostingClassifier()                           # 0.8652069797279378
    rf = RandomForestClassifier()                               # 0.8541507185968265
    dc=DecisionTreeClassifier()                                 # 0.8123830914998582
    xgb = XGBClassifier()                                       # 0.8632413919090565
    lgb = LGBMClassifier()                                      # 0.8723319992032567
    models = {
        'naive bayes': nb,
        # 'svm': svc,
        # 'linear_svm': ln_svc,
        'knn': knn,
        'logistic': lr,
        # 'mlp-nn': nn,
        'ada boost':ab,
        'random forest': rf,
        'gradient boost': gb,
        'dc':dc,
        'xgb':xgb,
        'lgb':lgb
    }
    return models


def score_models(models, X, y):
    """Score model in prediction DF"""
    print("评价每个模型.")
    for name,model in models.items():
        score = cross_val_score(model,X,y,scoring='accuracy',cv=5)
        mean_score=np.mean(score)
        print("{}: {}" .format(name, mean_score))
    print("Done.\n")



models = get_models()
train, test, train_label, test_label = prepare()
score_models(models, train, train_label)