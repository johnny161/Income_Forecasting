# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#使用随机森林  也是Bagging的一种
if __name__ == "__main__":
    train, test, train_label, test_label = prepare()
    
    ##### first round grid search ========================================
    ##### {'max_depth': 11, 'n_estimators': 300} 0.8598324801917615
    #param_grid ={'n_estimators':[200,300,400],'max_depth':[7,9,11]}
    #grid_search=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    ##### second round grid search ========================================
    ##### {'max_depth': 13, 'n_estimators': 400} 0.8612144780557953
    #param_grid ={'n_estimators':[400, 500],'max_depth':[11, 13]}
    #grid_search=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    ##### third round grid search ========================================
    ##### {'max_depth': 15, 'n_estimators': 400} 0.8616137456706319
    #param_grid ={'n_estimators':[300, 400],'max_depth':[13, 15]}
    #grid_search=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
    #grid_search.fit(train,train_label)
    #print(grid_search.best_params_,grid_search.best_score_)
    ##### =================================================================

    rf = RandomForestClassifier(n_estimators=400,max_depth=19)
    rf.fit(train,train_label)
    accuracy = rf.score(test,test_label)
    print(accuracy)
    ##### 3iterations: 0.8163503470302806
    ##### 9iterations: 0.8585467723112831
    ##### 15iterations: 0.8630305263804434
    ##### 17iterations: 0.863767581843867
    ##### 18iterations: 0.8649345863276211
    ##### 19iterations: 0.8653645353479515
    ##### 20iterations: 0.8637061605552484
    ##### 30iterations: 0.8562741846323936
    ##### 50iterations: 0.8525274860266568