# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_oof(clf,X,y,test_X):
    oof_train=np.zeros((n_train,))
    oof_test_mean=np.zeros((n_test,))
    oof_test_single=np.empty((5,n_test))
    for i, (train_index,val_index) in enumerate(kf.split(X,y)):
        kf_X_train=X[train_index]
        kf_y_train=y[train_index]
        kf_X_val=X[val_index]
        
        clf.fit(kf_X_train,kf_y_train)
        
        oof_train[val_index]=clf.predict(kf_X_val)
        oof_test_single[i,:]=clf.predict(test_X)
    oof_test_mean=oof_test_single.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)

#∂‘stackingÀ„∑®
if __name__ == "__main__":
    train, test, train_label, test_label = prepare()

    n_train=train.shape[0]
    n_test=test.shape[0]
    kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)

    LR_train,LR_test=get_oof(LogisticRegression(C=1),train,train_label,test)
    SVM_train,SVM_test=get_oof(SVC(C=1, gamma=0.1),train,train_label,test)
    GBDT_train,GBDT_test=get_oof(GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=5),train,train_label,test)

    X_stack=np.concatenate((LR_train,SVM_train,GBDT_train),axis=1)
    y_stack=train_label
    X_test_stack=np.concatenate((LR_test,SVM_test,GBDT_test),axis=1)

    stack_score=cross_val_score(RandomForestClassifier(n_estimators=500,max_depth=19),X_stack,y_stack,cv=5)
    print(stack_score.mean(),stack_score)

    rf=RandomForestClassifier(n_estimators=500,max_depth=19)
    rf.fit(X_stack,y_stack)
    print(rf.score(X_test_stack,test_label))

    ##### max_depth=19  n_estimators=500=============================================
    ##### 0.8755604692586451 ========================================================



    ##### ======= {'max_depth': 12, 'max_features': 20, 'n_estimators': 500} 0.864198147403551=====================
    #param_grid ={'n_estimators':[600, 700, 800, 1000],'max_depth':[19], "max_features":[8, 20, 66]}
    #grid_search=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    #print("train")
    #print("param_grid ={'n_estimators':[600, 700, 800, 1000],'max_depth':[19]}")
    ##### ==========================================================================================================

    # max_feature_params = ['auto', 'sqrt', 'log2', .01, .5, .99]