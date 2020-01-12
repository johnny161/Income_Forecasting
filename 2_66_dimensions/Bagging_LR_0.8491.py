# -*- coding: utf-8 -*-
from data_processing import prepare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train, test, train_label, test_label = prepare()

    bagging = BaggingClassifier(LogisticRegression(C=10),n_estimators=150)
    bagging.fit(train,train_label)
    
    test_predict = bagging.predict(test)
    accuracy = accuracy_score(test_label, test_predict)
    print("accuracy of Bagging of LR is: {}".format(accuracy))
    ##### accuracy of Bagging of LR is: 0.8490878938640133