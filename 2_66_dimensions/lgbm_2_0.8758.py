# -*- coding: gbk -*-
""" 
@Reference:https://blog.csdn.net/u012735708/article/details/83749703
"""

import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score

from data_processing import prepare
 
train, test, train_label, test_label = prepare()
 
#### ����ת��
#print(u'����ת��')
#lgb_train = lgb.Dataset(train, train_label, free_raw_data=False)
#lgb_eval = lgb.Dataset(test, test_label, reference=lgb_train,free_raw_data=False)
 
#### ���ó�ʼ����--����������֤����
#print(u'���ò���')
#params = {
#          'boosting_type': 'gbdt',
#          'objective': 'binary',
#          'metric': 'auc',
#          'nthread':4,
#          'learning_rate':0.1
#          }
 
#### ������֤(����)
#print('������֤')
#max_auc = float('0')
#best_params = {}
 
## ׼ȷ��
#print("����1�����׼ȷ��")
#for num_leaves in range(5,100,5):
#    for max_depth in range(3,8,1):
#        params['num_leaves'] = num_leaves
#        params['max_depth'] = max_depth
 
#        cv_results = lgb.cv(
#                            params,
#                            lgb_train,
#                            seed=1,
#                            nfold=5,
#                            metrics=['auc'],
#                            early_stopping_rounds=10,
#                            verbose_eval=True
#                            )
            
#        mean_auc = pd.Series(cv_results['auc-mean']).max()
#        boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
            
#        if mean_auc >= max_auc:
#            max_auc = mean_auc
#            best_params['num_leaves'] = num_leaves
#            best_params['max_depth'] = max_depth
#if 'num_leaves' and 'max_depth' in best_params.keys():          
#    params['num_leaves'] = best_params['num_leaves']
#    params['max_depth'] = best_params['max_depth']
 
## �����
#print("����2�����͹����")
#for max_bin in range(5,256,10):
#    for min_data_in_leaf in range(1,102,10):
#            params['max_bin'] = max_bin
#            params['min_data_in_leaf'] = min_data_in_leaf
            
#            cv_results = lgb.cv(
#                                params,
#                                lgb_train,
#                                seed=1,
#                                nfold=5,
#                                metrics=['auc'],
#                                early_stopping_rounds=10,
#                                verbose_eval=True
#                                )
                    
#            mean_auc = pd.Series(cv_results['auc-mean']).max()
#            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
#            if mean_auc >= max_auc:
#                max_auc = mean_auc
#                best_params['max_bin']= max_bin
#                best_params['min_data_in_leaf'] = min_data_in_leaf
#if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
#    params['min_data_in_leaf'] = best_params['min_data_in_leaf']
#    params['max_bin'] = best_params['max_bin']
 
#print("����3�����͹����")
#for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
#    for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
#        for bagging_freq in range(0,50,5):
#            params['feature_fraction'] = feature_fraction
#            params['bagging_fraction'] = bagging_fraction
#            params['bagging_freq'] = bagging_freq
            
#            cv_results = lgb.cv(
#                                params,
#                                lgb_train,
#                                seed=1,
#                                nfold=5,
#                                metrics=['auc'],
#                                early_stopping_rounds=10,
#                                verbose_eval=True
#                                )
                    
#            mean_auc = pd.Series(cv_results['auc-mean']).max()
#            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
#            if mean_auc >= max_auc:
#                max_auc=mean_auc
#                best_params['feature_fraction'] = feature_fraction
#                best_params['bagging_fraction'] = bagging_fraction
#                best_params['bagging_freq'] = bagging_freq
 
#if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
#    params['feature_fraction'] = best_params['feature_fraction']
#    params['bagging_fraction'] = best_params['bagging_fraction']
#    params['bagging_freq'] = best_params['bagging_freq']
 
 
#print("����4�����͹����")
#for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
#    for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]:
#        params['lambda_l1'] = lambda_l1
#        params['lambda_l2'] = lambda_l2
#        cv_results = lgb.cv(
#                            params,
#                            lgb_train,
#                            seed=1,
#                            nfold=5,
#                            metrics=['auc'],
#                            early_stopping_rounds=10,
#                            verbose_eval=True
#                            )
                
#        mean_auc = pd.Series(cv_results['auc-mean']).max()
#        boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
#        if mean_auc >= max_auc:
#            max_auc=mean_auc
#            best_params['lambda_l1'] = lambda_l1
#            best_params['lambda_l2'] = lambda_l2
#if 'lambda_l1' and 'lambda_l2' in best_params.keys():
#    params['lambda_l1'] = best_params['lambda_l1']
#    params['lambda_l2'] = best_params['lambda_l2']
 
#print("����5�����͹����2")
#for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#    params['min_split_gain'] = min_split_gain
    
#    cv_results = lgb.cv(
#                        params,
#                        lgb_train,
#                        seed=1,
#                        nfold=5,
#                        metrics=['auc'],
#                        early_stopping_rounds=10,
#                        verbose_eval=True
#                        )
            
#    mean_auc = pd.Series(cv_results['auc-mean']).max()
#    boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
 
#    if mean_auc >= max_auc:
#        max_auc=mean_auc
        
#        best_params['min_split_gain'] = min_split_gain
#if 'min_split_gain' in best_params.keys():
#    params['min_split_gain'] = best_params['min_split_gain']
 
#print(best_params) #ֻ����



##### Ԥ�� #########################################################
model=lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.01, n_estimators=1000, max_depth=6, num_leaves=50,max_bin=255,min_data_in_leaf=11,bagging_fraction=0.9,bagging_freq= 40, feature_fraction= 0.6,
lambda_l1=0,lambda_l2=0,min_split_gain=0)
model.fit(train,train_label)
y_pre=model.predict(test)
print("acc:",accuracy_score(test_label,y_pre))