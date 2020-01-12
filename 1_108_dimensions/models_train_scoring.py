# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang
@Reference:https://github.com/yanqiangmiffy/income-forecast/blob/master/utils.py
"""
import matplotlib.pyplot as plt
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
import warnings
warnings.filterwarnings('ignore')

output_dir = "output/"
fea_columns = []

def dataProcess_X(rawData):
    if "income" in rawData.columns:
        Data = rawData.drop(['income'], axis=1)
    else:
        Data = rawData
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjectColumn = [x for x in list(Data) if x not in listObjectColumn] #读取数字的column
    
    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjectColumn]

    ObjectData = pd.get_dummies(ObjectData)
    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data.astype("float64")

    #去除序号列
    #Data.drop("fnlwgt", axis=1, inplace=True)

    #Data = (Data - Data.mean(axis=0)) / Data.std(axis=0)
    return Data

def dataProcess_Y(rawData):
    y = rawData['income']
    Data_y = pd.DataFrame((y==' >50K').astype("float64"), columns=["income"])
    return Data_y

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def get_models():
    """
    生成机器学习库,此处没调参数                                训练集验证精确度    只删去序号列   只去除归一化
    """
    nb = GaussianNB()                                           # 0.4600           0.4600         0.7952
    svc = SVC(C=100, probability=True)
    ln_svc = LinearSVC()
    knn = KNeighborsClassifier()                                # 0.8239           0.8250         0.7766
    lr = LogisticRegression()                                   # 0.8515           0.8512         0.7974
    nn = MLPClassifier(solver='sgd', alpha=0.01, batch_size=64) # 0.8482           0.8492         0.7600
    ab = AdaBoostClassifier()                                   # 0.8604           0.8600         0.8604
    gb = GradientBoostingClassifier()                           # 0.8652           0.8657         0.8651
    rf = RandomForestClassifier()                               # 0.8542           0.8426         0.8453
    dc=DecisionTreeClassifier()                                 # 0.8137           0.8196         0.8132
    xgb = XGBClassifier()                                       # 0.8630           0.8636         0.8630
    lgb = LGBMClassifier()                                      # 0.8730           0.8732         0.8737
    models = {
        'naive bayes': nb,
        # 'svm': svc,
        # 'linear_svm': ln_svc,
        'knn': knn,
        'logistic': lr,
        'mlp-nn': nn,
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

# 特征重要性
def plot_GBDT_fea_importance(train, train_label):
    model = GradientBoostingClassifier()
    model.fit(train, train_label)
    fi = pd.DataFrame({'importance':model.feature_importances_},index=fea_columns)
    print(fi.sort_values('importance',ascending=False)[:20])
    fi.sort_values('importance',ascending=False)[:20].plot.bar(figsize=(11,7))
    plt.xticks(rotation=90)
    plt.title('Feature Importance Top20',size='x-large')
    plt.show()

if __name__ == "__main__":
    trainData = pd.read_csv("data/adult_train.csv")
    testData = pd.read_csv("data/x_test.csv")
    ans = pd.read_csv("data/y_test.csv")
    x_train = dataProcess_X(trainData)
    y_train = dataProcess_Y(trainData).values
    x_test = dataProcess_X(testData)
    y_test = ans['label'].values
    fea_columns = list(x_train.iloc[:,:].columns)
    x_train_, x_test_ = x_train.align(x_test, join='left', fill_value=0, axis=1) #类似左外连接同时对齐属性
    fea_columns = list(x_train.columns)
    x_train = x_train_.values
    x_test = x_test_.values

    print(x_train.shape)

    models = get_models()
    score_models(models, x_train, y_train)

    #plot_GBDT_fea_importance(x_train, y_train)





