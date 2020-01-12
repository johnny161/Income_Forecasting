# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train, test, train_label, test_label = prepare()
    
    ##### first round grid search ==============={'C': 10} 0.8464421799002636
    #param_grid = {'C':[0.01,0.1,1,10]}
    #grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    #grid_search.fit(train, train_label) 
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    ##### second round grid search =============={'C': 10} 0.8464421799002636
    #param_grid = {'C':[8,10,12,15]}
    #grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    #grid_search.fit(train, train_label) 
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    clf = LogisticRegression(C=10)
    clf.fit(train, train_label)
    test_predict = clf.predict(test)
    accuracy = accuracy_score(test_label, test_predict)
    print("accuracy of LogisticRegression is: {}".format(accuracy))
    ##### accuracy of LogisticRegression is: 0.848965051286776