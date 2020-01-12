
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

output_dir = "output/"

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

    Data = (Data - Data.mean(axis=0)) / Data.std(axis=0)
    return Data

def dataProcess_Y(rawData):
    y = rawData['income']
    Data_y = pd.DataFrame((y==' >50K').astype("float64"), columns=["income"])
    return Data_y

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

if __name__ == "__main__":
    trainData = pd.read_csv("data/adult_train.csv")
    testData = pd.read_csv("data/x_test.csv")
    ans = pd.read_csv("data/y_test.csv")

    x_train = dataProcess_X(trainData)
    y_train = dataProcess_Y(trainData).values
    x_test = dataProcess_X(testData)
    y_test = ans['label'].values

    x_train_, x_test_ = x_train.align(x_test, join='left', fill_value=0, axis=1) #类似左外连接同时对齐属性
    x_train = x_train_.values
    x_test = x_test_.values

    train_rows = int(x_train.shape[0])
    test_rows = int(x_test.shape[0])
    cols = int(x_train.shape[1])

    gbt = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=5)
    gbt.fit(x_train, y_train)
    accuracy = gbt.score(x_test, y_test)
    print("accuracy of GBDT is: {}".format(accuracy))
    #accuracy of GBDT is: 0.8684355997788834




