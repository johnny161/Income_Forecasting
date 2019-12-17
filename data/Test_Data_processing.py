import pandas as pd 
import numpy as np

names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
         'native-country', 'income']

adult_train = pd.read_csv("adult_train.csv", header=None, names=names)

adult_test = pd.read_csv("adult_test.csv", skiprows=1, header=None, names=names)
# print(adult_test.head())

adult_test_result = pd.DataFrame(adult_test['income'])
adult_test_result.insert(1, "label", (adult_test_result['income'] == ' >50K.').astype(np.int))
# print(adult_test.head())

# adult_train.to_csv('./adult_train.csv', index=False)
adult_test.to_csv('./x_test.csv', columns=names[:-1], index=False)
adult_test_result.to_csv('./y_test.csv', columns=['label'], index=False)