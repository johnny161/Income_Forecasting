import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

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
    Data.astype("int64")

    Data = (Data - Data.mean(axis=0)) / Data.std(axis=0)
    return Data

def dataProcess_Y(rawData):
    y = rawData['income']
    Data_y = pd.DataFrame((y==' >50K').astype("int64"), columns=["income"])
    return Data_y

if __name__ == "__main__":
    trainData = pd.read_csv("../data/adult_train.csv")
    testData = pd.read_csv("../data/x_test.csv")
    ans = pd.read_csv("../data/y_test.csv")

    x_train = dataProcess_X(trainData).values
    y_train = dataProcess_Y(trainData).values
    x_test = dataProcess_X(testData).values
    y_test = ans['label'].values

    print(x_train.shape)
