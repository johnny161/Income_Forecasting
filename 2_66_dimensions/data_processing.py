# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

fea_columns = []

def process_label(df):
    df['sex'][df['sex']==' Female'] = 0
    df['sex'][df['sex']==' Male'] = 1

    df['native-country'] = df['native-country'].apply(lambda el: 1 if el.strip() == 'United-States' else 0)

    # 数据预处理 类别编码
    cate_cols = ['workclass','education','marital-status','occupation','relationship','race']
    df = pd.get_dummies(df, columns=cate_cols)
    return df

def process_nums(df):
    # 数据预处理 数值型数据
    num_cols = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    for num in num_cols:
        df[num] = df[num].apply(lambda el: np.log(el + 1))

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].values)
    return df

def create_feature(df, train_len):
    global fea_columns
    # 数据预处理 类别编码
    new_df = process_label(df)
    # 数据预处理 数值型数据
    new_df = process_nums(new_df)
    # 标签预处理
    new_df['income'][(new_df['income']==' <=50K') | (new_df['income']==' <=50K.')] = 0
    new_df['income'][(new_df['income']==' >50K') | (new_df['income']==' >50K.')] = 1

    new_df = new_df.astype('float64')
    new_df_label = new_df.pop('income')
    new_train, new_test = new_df[:train_len], new_df[train_len:]
    new_train_label, new_test_label = new_df_label[:train_len], new_df_label[train_len:]
    print(new_train.shape, new_test.shape, new_train_label.shape, new_test_label.shape)
    fea_columns = list(new_train.columns)
    #print(fea_columns)
    return new_train.values, new_test.values, new_train_label.values, new_test_label.values

def test5models(train, train_label):
    model = [LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),SVC()]
    names = ['Lr', 'Tree', 'RF', 'GDBT', 'SVM']
    print("5_cross_val_score of 'Lr', 'Tree', 'RF', 'GDBT', 'SVM':")
    for name,model in zip(names, model):
        score = cross_val_score(model, train, train_label,cv=5)
        print("{}:{},{}".format(name,score.mean(),score))

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

def prepare():
    # input data
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
         'native-country', 'income']
    test_data = pd.read_csv("data/adult_test.csv", skiprows=1, names=names)
    train_data = pd.read_csv("data/adult_train.csv", names=names)
    train_len = len(train_data)

    # remove ? all, here we can try only drop the train_data next time
    all_data = pd.concat((train_data, test_data), axis=0)
    all_data.replace(to_replace=" ?", value=np.NaN, inplace=True)

    # fill np.NaN with mode in ["workclass", "occupation", "native-country"
    all_data["workclass"] = all_data["workclass"].fillna("Private")
    all_data["occupation"] = all_data["occupation"].fillna("Prof-specialty")
    all_data["native-country"] = all_data["native-country"].fillna("United-States")

    #========================================================================
    return create_feature(all_data, train_len)

if __name__ == "__main__":
    train, test, train_label, test_label = prepare()
    #test5models(train, train_label)
    plot_GBDT_fea_importance(train, train_label)
    #plot_GBDT_fea_importance(test, test_label)