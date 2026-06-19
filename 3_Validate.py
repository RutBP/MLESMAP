""" LOAD MODEL: VALIDATION
Input files:
	- Test database
	- Model and scalers from Train.py
-----------------------------------------------------------------------------------------
Copyright (c) . All rights reserved.
BSD 3-CLAUSE or MODIFIED BSD LICENSE

Copyright 2026 Rut Blanco Prieto, Marisol Monterrubio Velasco, Josep de la Puente.
"""


from dislib.trees import RandomForestRegressor
import dislib as ds
import numpy as np
from pycompss.api.api import compss_barrier, compss_wait_on
from dislib.preprocessing import MinMaxScaler
from dislib.model_selection import KFold
import pickle
from sklearn.metrics import *
from dislib.data.array import *
from dislib.utils import train_test_split
from dislib.data import load_txt_file
import pandas as pd
from datetime import datetime

periods = ['2']
MAX_DEPTH = 40          #from 1_GridSearch.py
N_EST = 20              #from 1_GridSearch.py
TRY_FEAT = "third"      #from 1_GridSearch.py
location = "Iceland"

path = "/path/to/save/Models/"
path2 = "/path/to/Test_Data.csv"

for period in periods:
    print("PERIOD: ", period)

    def _determination_coefficient(y_true, y_pred):
        u = np.sum(np.square(y_true - y_pred))
        v = np.sum(np.square(y_true - np.mean(y_true)))
        return 1 - u / v

    def make_load_model():
        ini_time = time.time()
        df = load_txt_file(path2 + "Test_"+str(period)+"s.csv", discard_first_row=True, col_of_index=True,block_size=(133334, 9))

        Data_Y_arr = df[:, 8:9]
        Data_X_arr = df[:, 0:8]
        scaler_X = MinMaxScaler(feature_range=(0, 8))
        scaler_X.load_model(path+'scaler_X_'+str(period)+'s_'+str(location)+'_est'+str(N_EST)+'_dep'+str(MAX_DEPTH)+'_'+str(TRY_FEAT)+'.json')
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_y.load_model(path+'scaler_Y_'+str(period)+'s_'+str(location)+'_est'+str(N_EST)+'_dep'+str(MAX_DEPTH)+'_'+str(TRY_FEAT)+'.json')
        x_ds_test = Data_X_arr.rechunk((133334,8))
        print(x_ds_test)
        y_ds_test = Data_Y_arr.rechunk((133334,1))
        x_test = scaler_X.transform(x_ds_test)
        print(x_test)
        y_test = scaler_y.transform(y_ds_test)
        load_time = time.time()
        print("Load time", load_time - ini_time)
        print("Fitting...")
        start_time = time.time()
        rf = RandomForestRegressor(max_depth=MAX_DEPTH,n_estimators=N_EST,try_features=TRY_FEAT,random_state=0)
        rf.load_model(path+str(location)+'_model_T'+str(period)+'s_depth'+str(MAX_DEPTH)+'_nestim'+str(N_EST)+'.save', load_format="pickle")    
        y_pred = scaler_y.inverse_transform(rf.predict(x_test))
        y_true = scaler_y.inverse_transform(y_test)
        y_pred = y_pred.collect()
        y_true = y_true.collect()
        score2 = _determination_coefficient(y_true, y_pred)
        fit_time = time.time()
        print("Fit time:", fit_time - start_time)
        score1 = rf.score(x_test, y_test, collect=True)
        score1_time = time.time()
        print("score1 time:", score1_time - fit_time)
        print("score1", score1)
        pred_time = time.time()
        print("Prediction time:", pred_time - score1_time)
        print("Total time:", pred_time - start_time)
        score2 = _determination_coefficient(y_true, y_pred)
        print("score2", score2)
        np.savetxt(path + 'y_pred_val_dislib_'+str(location)+'_T'+str(period)+'s_depth'+str(MAX_DEPTH)+'_nestim'+str(N_EST)+'.dat',y_pred)
        np.savetxt(path + 'y_true_val_dislib_'+str(location)+'_T'+str(period)+'s_depth'+str(MAX_DEPTH)+'_nestim'+str(N_EST)+'.dat',y_true) 
        mse = mean_squared_error(y_true, y_pred)
        print("mse=",mse) 
        evs = explained_variance_score(y_true,y_pred)
        print("evs=",evs)
        mae = mean_absolute_error(y_true,y_pred)
        print("mae=", mae)
        mape = 100*(abs(y_pred-y_true)/y_true)
        print("mape=", 100-np.mean(mape))
        print("coefPearson=", np.corrcoef(y_true,y_pred))
        

    if __name__ == '__main__':
        make_load_model()
