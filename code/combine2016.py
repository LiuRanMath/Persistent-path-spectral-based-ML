# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:15:14 2022

@author: ranran
"""

import numpy as np
import scipy as sp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

test_file = open('../data/2016/name/test_name.txt')
pre_test_data = test_file.readlines()
test_data = eval(pre_test_data[0])
test_file.close()

train_file = open('../data/2016/name/train_name.txt')
pre_train_data = train_file.readlines()
train_data = eval(pre_train_data[0])
train_file.close()

all_file = open('../data/2016/name/all_name.txt')
pre_all_data = all_file.readlines()
all_data = eval(pre_all_data[0])
all_file.close()



def combine_feature_of_euclid_and_charge(typ,hop):
    file1 = '../data/2016/euclid_feature/' + typ + '_mm_hop' + str(hop) + '.csv'
    file2 = '../data/2016/charge_feature/' + typ + '_mm_hop' + str(hop) + '.csv'
    feature1 = np.loadtxt(file1,delimiter = ',')
    feature2 = np.loadtxt(file2,delimiter = ',')
    
    com_feature = np.hstack((feature1,feature2))
    filename = '../data/2016/combine_euclid_and_charge_feature/' + typ + '_mm_hop' + str(hop) + '.csv'
    np.savetxt(filename,com_feature,delimiter=',')
    
    
    

    
def gradient_boosting(X_train,Y_train,X_test,Y_test):
    params={'n_estimators': 40000, 'max_depth': 6, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train,Y_train)
    pearson_coorelation = sp.stats.pearsonr(Y_test,regr.predict(X_test))
    mse = mean_squared_error(Y_test, regr.predict(X_test))
    rmse = pow(mse,0.5)
    return [pearson_coorelation[0],rmse]


def get_pearson_correlation(hop):
    feature_matrix_of_train = np.loadtxt( '../data/2016/combine_euclid_and_charge_feature/train_hop'+ str(hop) +'_mm.csv',delimiter=',' )
    target_matrix_of_train = np.loadtxt( '../data/2016/charge_feature/train_target.csv',delimiter=',' )
    feature_matrix_of_test = np.loadtxt( '../data/2016/combine_euclid_and_charge_feature/test_hop'+ str(hop) +'_mm.csv',delimiter=',' )
    target_matrix_of_test = np.loadtxt( '../data/2016/charge_feature/test_target.csv',delimiter=',' )
    number = 10
    P = np.zeros((number,1))
    M = np.zeros((number,1))
    print(feature_matrix_of_test.shape)
    print(feature_matrix_of_train.shape)
    print(target_matrix_of_test.shape,target_matrix_of_train.shape) 
    
    for i in range(number):
        [P[i][0],M[i][0]] = gradient_boosting(feature_matrix_of_train,target_matrix_of_train,feature_matrix_of_test,target_matrix_of_test)
        print(P[i])
    median_p = np.median(P)
    median_m = np.median(M)
    print('feature obtained by mean and median using hopping:' ,hop)
    print('for data 2016 , 10 results for euclid and charge distance-model are:')
    print(P)
    print('median pearson correlation values are')
    print(median_p)
    print('median root mean squared error values are')
    print(median_m)
    
    
    
    
def run_PDBbind_2016():
    ###############################################################################
    '''
    run this function, you can get the results for data2016 
    using mean and median attributes combined hopping 1,2,3
    with euclid and charge features
    '''
    ###############################################################################
    
    #combine feature
    for typ in ['test','train']:
        for hop in [1,2,3,12,123]:
            combine_feature_of_euclid_and_charge(typ, hop)
    
    # machine learning
    get_pearson_correlation(1)
    get_pearson_correlation(2)
    get_pearson_correlation(3)
    get_pearson_correlation(12)
    get_pearson_correlation(123)
            
    
run_PDBbind_2016()    
