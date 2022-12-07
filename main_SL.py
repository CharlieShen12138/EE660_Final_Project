# Import necessary packages 
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
np.seterr(divide='ignore', invalid='ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score

# Preprocessing Functions
def remove_constant(data):
    col_to_remove = []
    for col in data.columns:
        if col != 'ID' and col != 'target':
            if data[col].std() == 0:
                col_to_remove.append(col)
    return col_to_remove

def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups    
    # Covert data to dict like structure for better manipulate
    dups = []

    for t, v in groups.items(): 

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):            
        # Search dupilicated columns column by column
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break

    return dups   # Return duplicated columns
# Referance: https://www.kaggle.com/code/samratp/lightgbm-xgboost-catboost#XGB-Modeling

def drop_saprce(data):
    column_to_detect = [col for col in data.columns if col not in['ID','target']]
    counter = 0
    for col in column_to_detect:
        if len(np.unique(data[col])) < 2:
            data.drop(col, axis = 1, inplace = True)
            counter += 1
    print('The number of column has been droped is:', counter)
    return data

# Model Train & Evaluate functions
def Make_Score_Reg(Test_Y,Pred_Y):
    res_ = {}
    res_['RMSLE'] = mean_squared_log_error(Test_Y,Pred_Y)
    res_['R2'] = r2_score(Test_Y,Pred_Y)
    return res_

def Trival_System(Train_Y,Test_Y)-> dict:
    mean = np.mean(Train_Y)
    Pred_Y = [mean]*len(Test_Y)
    Pred_Y = np.expm1(Pred_Y)
    return Make_Score_Reg(Test_Y,Pred_Y)

def KNN_Regression(Train_X,Train_Y,Test_X,Test_Y)-> dict:
    KNN_modle = KNeighborsRegressor()
    Pred_Y = KNN_modle.fit(Train_X,Train_Y).predict(Test_X)
    Pred_Y = np.expm1(Pred_Y)
    return Make_Score_Reg(Test_Y,Pred_Y)

def Random_Froest_Regressor(Train_X , Train_Y , Test_X, Test_Y):
    Model_RF = RandomForestRegressor(n_estimators = 100, max_depth = None, criterion = 'squared_error', random_state = 42)
    # Train and Evaluate Model
    Model_RF.fit(Train_X, Train_Y)
    Pred_Y = Model_RF.predict(Test_X)
    Pred_Y = np.expm1(Pred_Y)
    return Make_Score_Reg(Test_Y,Pred_Y)

def AdaBoost_Regressor(Train_X,Train_Y,Test_X,Test_Y):
    Model_Ada = AdaBoostRegressor(learning_rate = 1, loss = 'exponential', n_estimators = 140, random_state = 42)
    Model_Ada.fit(Train_X, Train_Y)
    Pred_Y = Model_Ada.predict(Test_X)
    Pred_Y = np.expm1(Pred_Y)
    return Make_Score_Reg(Test_Y,Pred_Y)

def XGB_Train(Train_X,Train_Y,Test_X,Test_Y):
    XGB_Train_Data = xgb.DMatrix(Train_X,Train_Y)
    XGB_Test_Data =  xgb.DMatrix(Test_X)
    para = {'objective': 'reg:squarederror',
            'eval_metric':'rmse',
            'max_depth': 15,
            'learning_rate': 0.3,
            'alpha':0.001,
            'gamma': 0.0,
            'eta': 0.001,
            'min_child_weight': 1}
    model_XGB = xgb.train(para, XGB_Train_Data, 50, maximize=False, verbose_eval=100)
    Pred_Y = np.expm1(model_XGB.predict(XGB_Test_Data))
    return Make_Score_Reg(Test_Y,Pred_Y)

def LGBM_Train(Train_X,Train_Y,Test_X,Test_Y):
    LGBM_para = {'objective' : 'regression',
                 'metric':'rmse',
                 'bagging_fraction': 0.6,
                 'feature_fraction': 0.6,
                 'learning_rate': 0.1,
                 'max_depth': 15,
                 'num_leaves': 50,
                 'force_row_wise':True,
                 'verbose': -1}
    LGBM_Train_Data = lgb.Dataset(Train_X, label = Train_Y, silent = True, params={'verbose': -1}, free_raw_data=False)
    LGBM_Model = lgb.train(params = LGBM_para, train_set = LGBM_Train_Data)
    Pred_Y = np.expm1(LGBM_Model.predict(Test_X))
    return Make_Score_Reg(Test_Y,Pred_Y)

# Main Function
if __name__ == "__main__":   
    os.chdir('Data')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')
    from Data.qns3vm import *
    # Preprocessing Step
    customer_data = pd.read_csv('Data.csv')
    remove_constant_col = remove_constant(customer_data)
    customer_data.drop(remove_constant_col, axis = 1, inplace = True)
    remove_duplicate_col = duplicate_columns(customer_data)
    customer_data.drop(remove_duplicate_col, axis = 1, inplace = True)
    customer_data = drop_saprce(customer_data)
    customer_data_X = customer_data.drop(['ID', 'target'], axis=1)
    customer_data_Y = customer_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(customer_data_X, customer_data_Y, test_size = 0.2, random_state = 42)
    Y_train = np.log1p(Y_train)
    Perf_Trival = Trival_System(Y_train,Y_test)
    Perf_KNN = KNN_Regression(X_train,Y_train,X_test,Y_test)
    Perf_RF = Random_Froest_Regressor(X_train,Y_train,X_test,Y_test)
    Perf_Ada = AdaBoost_Regressor(X_train,Y_train,X_test,Y_test)
    Perf_XGB = XGB_Train(X_train,Y_train,X_test,Y_test)
    Perf_LGBM = LGBM_Train(X_train,Y_train,X_test,Y_test)
    System_Performance = {'Algorithm':['Trival','KNN','Random Forest',
                                   'AdaBoost','XGBoost', 'LightGBM'],
                      'RMSLE':[Perf_Trival['RMSLE'], Perf_KNN['RMSLE'], Perf_RF['RMSLE'],
                               Perf_Ada['RMSLE'], Perf_XGB['RMSLE'], Perf_LGBM['RMSLE']],
                      'R2':[Perf_Trival['R2'], Perf_KNN['R2'], Perf_RF['R2'],
                               Perf_Ada['R2'], Perf_XGB['R2'], Perf_LGBM['R2']]}
    System_Performance = pd.DataFrame(System_Performance)
    print(System_Performance)


