# Import necessary packages 
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from random import Random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from Data.qns3vm import *
np.seterr(divide='ignore', invalid='ignore')

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

def Customer_Class(Data_Y):
    res = []
    Y = list(Data_Y)
    # Define Intervals for different Classes
    # Indicates for Low Value Customer and High Value Customer
    for v in Y:
        if v >=0 and v < 0.5e7:
            res.append('-1')
        else:
            res.append('1')
    #print(collections.Counter(res))
    res = pd.DataFrame(res)
    return res[0]

def Semi_Dataset(Train_X, Train_Y, P = 0.15):
    Data = {}
    X_L, X_U, Y_L, Y_U = train_test_split(Train_X, Train_Y, test_size = (1-P), random_state = 42)
    Data['X_Labeled'] = X_L
    Data['X_Unlabeled'] = X_U
    Data['Y_Labeled'] = Y_L
    Data['Y_Unlabeled'] = Y_U
    return Data

def Class_Label_Self(Data_Y):
    res = []
    Y = list(Data_Y)
    # Define Intervals for different Classes
    # Indicates for Low Value Customer and High Value Customer
    for v in Y:
        if v >=0 and v < 0.5e7:
            res.append('0')      # Here we use 0 instead of 1 for Slef-Training
        else:
            res.append('1')
    #print(collections.Counter(res))
    res = pd.DataFrame(res)
    return res[0]

def Make_Score_Cls(Test_Y,Pred_Y):
    res_ = {}
    res_['Accuracy'] = accuracy_score(Test_Y,Pred_Y)
    res_['Confusion_Matrix'] = confusion_matrix(Test_Y,Pred_Y)
    return res_

# Model Training and Evaluation Functions
def Self_Training_KNN(X_Train, Y_Train, X_Test, Y_Test, U_p = 0.15):
    rng = np.random.RandomState(42)
    Random_Unlable = rng.rand(len(Y_Train)) < U_p
    # Using defalut setting: 15% labeled data
    Y_Tr_Temp = list(Y_Train)
    # Make Unlabled Data
    for i,lab in enumerate(Y_Tr_Temp):
        if Random_Unlable[i] == True:
            continue
        else:
            Y_Tr_Temp[i] = -1
    Y_Train = pd.Series(Y_Tr_Temp).ravel()
    # Traing and evaluate Performance
    Model_KNN = KNeighborsClassifier(n_neighbors=3)
    ST_KNN = SelfTrainingClassifier(Model_KNN)
    ST_KNN.fit(X_Train, Y_Train)
    Y_Pred = ST_KNN.predict(X_Test)
    return Make_Score_Cls(Y_Test, Y_Pred)

def Lable_Spreading(X_Train, Y_Train, U_p = 0.15):
    # Prepare for the Label Prediction
    rng = np.random.RandomState(42)
    Random_Unlable = rng.rand(len(Y_Train)) < U_p  # Unlabeled Portion
    # Using defalut setting: 15% labeled data
    Y_Tr_Temp = list(Y_Train)
    # Make Unlabled Data
    for i,lab in enumerate(Y_Tr_Temp):
        if Random_Unlable[i] == True:
            Y_Tr_Temp[i] = int(Y_Tr_Temp[i])
        else:
            Y_Tr_Temp[i] = -1
    Y_Train = pd.Series(Y_Tr_Temp).ravel()
    Label_Prop_Model = LabelSpreading()
    Label_Prop_Model.fit(X_Train, Y_Train)
    Y_Train_Pred = Label_Prop_Model.predict(X_Train)
    return Y_Train_Pred

def Lable_Propagation(X_Train, Y_Train , U_p = 0.15):
    # Prepare for the Label Prediction
    rng = np.random.RandomState(42)
    Random_Unlable = rng.rand(len(Y_Train)) < U_p  # Unlabeled Portion
    # Using defalut setting: 15% labeled data
    Y_Tr_Temp = list(Y_Train)
    # Make Unlabled Data
    for i,lab in enumerate(Y_Tr_Temp):
        if Random_Unlable[i] == True:
            Y_Tr_Temp[i] = int(Y_Tr_Temp[i])
        else:
            Y_Tr_Temp[i] = -1
    Y_Train = pd.Series(Y_Tr_Temp).ravel()
    Label_Prop_Model = LabelPropagation()
    Label_Prop_Model.fit(X_Train, Y_Train)
    Y_Train_Pred = Label_Prop_Model.predict(X_Train)
    return Y_Train_Pred

def Random_Froest_Classifier_LS(X_Train, Y_Train, X_Test, Y_Test):
    Model_RF = RandomForestClassifier(max_depth = 3, max_features = 'sqrt', n_estimators = 50, random_state = 42)
    Model_RF.fit(X_Train, Y_Train)
    Y_Pred = Model_RF.predict(X_Test)
    Y_Pred = [str(n) for n in Y_Pred]
    return Make_Score_Cls(Y_Test, Y_Pred)

def XGB_Classifier_LS(X_Train, Y_Train, X_Test, Y_Test):
    XGB_CLS = xgb.XGBClassifier(use_label_encoder = False,
                                gamma = 0.0,
                                learning_rate = 0.01,
                                max_depth = 3,
                                min_child_weight = 3)
    XGB_CLS.fit(X_Train,Y_Train)
    Y_Pred = XGB_CLS.predict(X_Test)
    Y_Pred = [str(n) for n in Y_Pred]
    return Make_Score_Cls(Y_Test, Y_Pred)

def LGBM_Classifier_LS(X_Train, Y_Train, X_Test, Y_Test):
    LGBM_para = {'objective' : 'multiclass',
                 'num_class': 2,
                 'bagging_fraction': 0.6,
                 'feature_fraction': 0.6,
                 'learning_rate': 0.001,
                 'max_depth': 3,
                 'min_data_in_leaf' : 50,
                 'num_leaves' : 20,
                 'force_row_wise':True,
                 'verbose': -1,
                 'seed': 42}
    LGBM_Train_Data = lgb.Dataset(X_Train, label = Y_Train, silent = True, params={'verbose': -1}, free_raw_data=False)
    LGBM_Model = lgb.train(params = LGBM_para, train_set = LGBM_Train_Data)
    Y_Pred = LGBM_Model.predict(X_Test)
    Y_Pred = np.argmax(Y_Pred, axis = 1)
    Y_Pred = [str(n) for n in Y_Pred]
    return Make_Score_Cls(Y_Test, Y_Pred)

def Random_Froest_Classifier_LP(X_Train, Y_Train, X_Test, Y_Test):
    Model_RF = RandomForestClassifier(max_depth = 3, max_features = 'sqrt', n_estimators = 50, random_state = 42)
    Model_RF.fit(X_Train, Y_Train)
    Y_Pred = Model_RF.predict(X_Test)
    Y_Pred = [str(n) for n in Y_Pred]
    return Make_Score_Cls(Y_Test, Y_Pred)

def XGB_Classifier_LP(X_Train, Y_Train, X_Test, Y_Test):
    XGB_CLS = xgb.XGBClassifier(use_label_encoder = False,
                                gamma = 0.0,
                                learning_rate = 0.1,
                                max_depth = 4,
                                min_child_weight = 5)
    XGB_CLS.fit(X_Train,Y_Train)
    Y_Pred = XGB_CLS.predict(X_Test)
    Y_Pred = [str(n) for n in Y_Pred]
    return Make_Score_Cls(Y_Test, Y_Pred)

def LGBM_Classifier_LP(X_Train, Y_Train, X_Test, Y_Test):
    LGBM_para = {'objective' : 'multiclass',
                 'num_class': 2,
                 'bagging_fraction': 0.6,
                 'feature_fraction': 0.6,
                 'learning_rate': 0.001,
                 'max_depth': 3,
                 'min_data_in_leaf' : 20,
                 'num_leaves' : 20,
                 'force_row_wise':True,
                 'verbose': -1,
                 'seed': 42}
    LGBM_Train_Data = lgb.Dataset(X_Train, label = Y_Train, silent = True, params={'verbose': -1}, free_raw_data=False)
    LGBM_Model = lgb.train(params = LGBM_para, train_set = LGBM_Train_Data)
    Y_Pred = LGBM_Model.predict(X_Test)
    Y_Pred = np.argmax(Y_Pred, axis = 1)
    Y_Pred = [str(n) for n in Y_Pred]
    return Make_Score_Cls(Y_Test, Y_Pred)

def S3VM_SSL(X_Train_L, Y_Train_L, X_Train_U, X_Test, Y_Test):
    Y_Train_L = [int(n) for n in Y_train_Labeled]
    Y_Test = [int(n) for n in Y_Test]
    X_Train_L = list(np.array(X_Train_L))
    Y_Train_L = list(np.array(Y_Train_L))
    X_Train_U = list(np.array(X_Train_U))
    X_Test = list(np.array(X_Test))
    random_gen = Random()
    random_gen.seed()
    Model_S3VM = QN_S3VM(X_Train_L, Y_Train_L, X_Train_U, random_gen, lam = 0.5, kernel_type = 'RBF')
    Model_S3VM.train()
    Y_Pred = Model_S3VM.getPredictions(X_Test)
    Y_Pred = pd.DataFrame(Y_Pred).fillna(1)
    Y_Pred = [Y_Pred.iloc[i][0] for i in range(len(Y_Test))]
    return Make_Score_Cls(Y_Test, Y_Pred)
# Referance: https://github.com/NekoYIQI/QNS3VM/blob/master/qns3vm.py

# Main function
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
    # Make Y as binary class for classification
    customer_data_Y = Customer_Class(customer_data_Y)
    X_train, X_test, Y_train, Y_test = train_test_split(customer_data_X, customer_data_Y, test_size = 0.2, random_state = 42)

    # This part of Data is Labeled with 1 and -1 #
    ###################################################################
    Semi_Supervise_Learning_Data_Set = Semi_Dataset(X_train, Y_train)
    X_train_Labeled = Semi_Supervise_Learning_Data_Set['X_Labeled']
    Y_train_Labeled = Semi_Supervise_Learning_Data_Set['Y_Labeled']
    X_train_Unlabeled = Semi_Supervise_Learning_Data_Set['X_Unlabeled']
    ###################################################################
    # This part of Data is Labeled with 1 and 0 #
    ###################################################################
    Self_Learning_Y = Class_Label_Self(customer_data['target'])
    X_train_SL, X_test_SL, Y_train_SL, Y_test_SL = train_test_split(customer_data.drop(['ID', 'target'], axis=1), 
                                                                Self_Learning_Y, 
                                                                test_size = 0.2, 
                                                                random_state = 42)
                                                            # Using Same random_state to get compareable result
    ###################################################################
    # Get prediction of Unlabed Data
    Y_Train_Label_Spreading = Lable_Spreading(X_Train = X_train_SL, Y_Train = Y_train_SL, U_p = 0.15)
    Y_Train_Label_Propagation = Lable_Propagation(X_Train = X_train_SL, Y_Train = Y_train_SL, U_p = 0.15)
    # Performance Evaluation (with fixed parameters)
    ## Baseline
    Perf_Slef_Trainging = Self_Training_KNN(X_Train = X_train_SL, Y_Train = Y_train_SL, X_Test = X_test_SL, Y_Test = Y_test_SL, U_p = 0.15)
    ## Label Spreading
    Perf_RF_LS = Random_Froest_Classifier_LS(X_Train = X_train_SL, Y_Train = Y_Train_Label_Spreading, X_Test = X_test_SL, Y_Test = Y_test_SL)
    Perf_XGB_LS = XGB_Classifier_LS(X_Train = X_train_SL, Y_Train = Y_Train_Label_Spreading, X_Test = X_test_SL, Y_Test = Y_test_SL)
    Perf_LGBM_LS = LGBM_Classifier_LS(X_Train = X_train_SL, Y_Train = Y_Train_Label_Spreading, X_Test = X_test_SL, Y_Test = Y_test_SL)
    ## Label Propagation
    Perf_RF_LP = Random_Froest_Classifier_LP(X_Train = X_train_SL, Y_Train = Y_Train_Label_Spreading, X_Test = X_test_SL, Y_Test = Y_test_SL)
    Perf_XGB_LP = XGB_Classifier_LP(X_Train = X_train_SL, Y_Train = Y_Train_Label_Spreading, X_Test = X_test_SL, Y_Test = Y_test_SL)
    Perf_LGBM_LP = LGBM_Classifier_LP(X_Train = X_train_SL, Y_Train = Y_Train_Label_Spreading, X_Test = X_test_SL, Y_Test = Y_test_SL)
    ## S3VM
    Perf_S3VM = S3VM_SSL(X_Train_L = X_train_Labeled, Y_Train_L = Y_train_Labeled, X_Train_U = X_train_Unlabeled, X_Test = X_test, Y_Test = Y_test)

    # Make charts about performane on different algorithms (Base on Accuracy)
    ## Baseline V.S. Label Spreading & Label Propagation
    System_Performance = {'Algorithm':['Self Learning','Random Forest','XGBoost', 'LightGBM'],
                      'Label Spreading':[Perf_Slef_Trainging['Accuracy'],Perf_RF_LS['Accuracy'],
                                         Perf_XGB_LS['Accuracy'], Perf_LGBM_LS['Accuracy']],
                      'Label Propagation':[Perf_Slef_Trainging['Accuracy'],Perf_RF_LP['Accuracy'],
                                         Perf_XGB_LP['Accuracy'], Perf_LGBM_LP['Accuracy']]}
    System_Performance = pd.DataFrame(System_Performance)
    ## Baseline V.S. S3VM
    S3VM_VS_BL = {'Algorithm':['Slef Learning', 'S3VM'],
              'Accuracy': [Perf_Slef_Trainging['Accuracy'], Perf_S3VM['Accuracy']]}
    S3VM_VS_BL = pd.DataFrame(S3VM_VS_BL)
    
    print(System_Performance)
    print(S3VM_VS_BL)






    

