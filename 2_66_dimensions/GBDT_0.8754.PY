# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

#Gradient Boosting Decision Tree
if __name__ == "__main__":
    train, test, train_label, test_label = prepare()
    
    ##### first round grid search =========================================
    ##### {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200} 0.8730384722899693
    #param_grid ={'n_estimators':[120, 200],'learning_rate':[0.1],'max_depth':[5]}
    #grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    ##### second round grid search =========================================
    ##### {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200} 0.8729770519441178
    #param_grid ={'n_estimators':[200],'learning_rate':[0.05, 0.1, 0.2],'max_depth':[5]}
    #grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    ##### third round grid search ========================================
    ##### {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200} 0.8730691848206819
    #param_grid ={'n_estimators':[200],'learning_rate':[0.1],'max_depth':[5, 6, 7]}
    #grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    gbt = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=5)
    gbt.fit(train, train_label)
    accuracy = gbt.score(test, test_label)
    print("accuracy of GBDT is: {}".format(accuracy))
    ##### accuracy of GBDT is: 0.8753762053927892
