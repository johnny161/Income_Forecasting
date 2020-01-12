# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    train, test, train_label, test_label = prepare()
    
    ##### first round grid search ========================================
    ##### {'C': 1, 'gamma': 0.1} 0.8553178796442269
    #param_grid={'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}
    #grid_search=GridSearchCV(SVC(),param_grid,cv=5)
    #grid_search.fit(train, train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    svm = SVC(C=1, gamma=0.1)
    svm.fit(train, train_label)
    accuracy = svm.score(test, test_label)
    print("accuracy of SVM is: {}".format(accuracy))
    ##### accuracy of SVM is: 0.8558442356120631