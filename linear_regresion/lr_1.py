
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import scipy.optimize as opt

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

def sigmoid(z):
    res = 1.0 / (1 + np.exp(-z))
    return res

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    z = X * theta.T
    h = sigmoid(z)
    cost = np.multiply(y, np.log(h)) + np.multiply(1-y, np.log(1-h))
    
    return -np.mean(cost)

def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        grad[i] = np.mean(np.multiply(error, X[:, i]))

    return grad

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

if __name__ == "__main__":
    trainData = pd.read_csv("../data/adult_train.csv")
    testData = pd.read_csv("../data/x_test.csv")
    ans = pd.read_csv("../data/y_test.csv")

    x_train = dataProcess_X(trainData).values
    y_train = dataProcess_Y(trainData).values
    x_test = dataProcess_X(testData).values
    y_test = ans['label'].values

    train_rows = int(x_train.shape[0])
    test_rows = int(x_test.shape[0])
    x_train = np.insert(x_train, 0, np.ones(train_rows), 1)
    x_test = np.insert(x_test, 81, np.zeros(test_rows), 1)
    x_test = np.insert(x_test, 0, np.ones(test_rows), 1)

    cols = int(x_train.shape[1])
    theta = np.zeros(cols)

    result_train = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x_train, y_train))
    print("train finished!")

    theta_min = np.matrix(result_train[0])
    predictions = predict(theta_min, x_test)
    correct = [1 if a == b else 0 for (a, b) in zip(y_test, predictions)]
    accuracy = sum(map(int, correct)) / test_rows
    print('accuracy = {0}%'.format(accuracy))





