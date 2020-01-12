# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train, test, train_label, test_label = prepare()

    ada = AdaBoostClassifier(base_estimator=LogisticRegression(C=10),n_estimators=150)
    ada.fit(train,train_label)
    
    accuracy = ada.score(test, test_label)
    print("accuracy of AdaBoost of LR is: {}".format(accuracy))
    ##### accuracy of AdaBoost of LR is: 0.8489036299981574