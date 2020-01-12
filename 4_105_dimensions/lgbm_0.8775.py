import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

    ##### ==step====================================================采用之前的训练模型
    model=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=6, num_leaves=20,max_bin=225,min_data_in_leaf=11,bagging_fraction=0.6,bagging_freq= 0, feature_fraction= 0.6,
    lambda_l1=0.001,lambda_l2=0.3,min_split_gain=0)
    model.fit(train,train_label)
    y_pre=model.predict(test)
    print("acc:",accuracy_score(test_label,y_pre))

    ##### acc: 0.8775259504944414
    ##### ==step====================================================
    
    ##### ==step====================================================打印混淆矩阵_ROC
    model_1_predict_test_proba = model.predict_proba(test)[:,1]
    
    fpr, tpr, thresholds  = roc_curve(test_label,model_1_predict_test_proba )
    
    confusion_matrix_test = confusion_matrix(test_label,y_pre)
    
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    sns.heatmap(confusion_matrix_test,annot=True,cmap="YlGnBu")
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted",fontsize=14)
    plt.ylabel("Actual",fontsize=14)

    # Plot ROC curve
    plt.subplot(1,2,2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('ROC Curve',fontsize=15)
    plt.show()
    print("\n","\n",'AUC Değeri : ', roc_auc_score(test_label,model_1_predict_test_proba ))
    ##### ==step====================================================