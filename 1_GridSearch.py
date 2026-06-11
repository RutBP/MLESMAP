from sklearn import datasets
import pandas as pd
import dislib as ds
import numpy as np
from dislib.regression import RandomForestRegressor
from _search import GridSearchCV
from sklearn.datasets import make_regression
#from dislib.preprocessing import MinMaxScaler
from dislib.utils import train_test_split
from dislib.data import load_txt_file
from pycompss.api.api import compss_wait_on, compss_barrier
from minmax_scaler import MinMaxScaler

periods = ['2']
location = "Iceland"


for period in periods:
    print("PERIOD", period)
    def demostracion_grid_search():
        df = load_txt_file("/path/to/Train_Data.csv",discard_first_row=True, col_of_index=True,block_size=(33334, 9))
        dfAll = load_txt_file("/path/to/All_Data.csv",discard_first_row=True, col_of_index=True,block_size=(33334, 9))

        Data_X_arr_All = dfAll[:, 0:8]
        Data_Y_arr_All = dfAll[:, 8:9]
        Data_X_arr = df[:, 0:8]
        Data_Y_arr = df[:, 8:9]
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_X.fit(Data_X_arr_All)
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_y.fit(Data_Y_arr_All)
        x_ds_train, x_ds_test, y_ds_train, y_ds_test = train_test_split(Data_X_arr, Data_Y_arr)
        x_ds_test = x_ds_test.rechunk((10000, 8))
        y_ds_test = y_ds_test.rechunk((10000, 1))
        x_train = scaler_X.transform(x_ds_train)
        x_test = scaler_X.transform(x_ds_test)
        y_train = scaler_y.transform(y_ds_train)
        y_test = scaler_y.transform(y_ds_test)
        parameters = {'max_depth':(10,15,20,25,30,35,40,45,50)}

        rf = RandomForestRegressor(n_estimators=20,try_features='sqrt')   # https://dislib.readthedocs.io/en/latest/dislib.trees.html#dislib.trees.RandomForestClassifier

        searcher = GridSearchCV(rf, parameters, cv=5)
        np.random.seed(0)
        searcher.fit(x_train, y_train)
        print(str(location))
        print(searcher.cv_results_['params'])
        print(searcher.cv_results_['mean_test_score'])
        pd_df = pd.DataFrame.from_dict(searcher.cv_results_)
        print(pd_df[['params', 'mean_test_score']])
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(pd_df)
        print('best_estimator')
        print(searcher.best_estimator_)
        print(searcher.best_score_)
        print(searcher.best_params_)
        print(searcher.best_index_)
        print(searcher.scorer_)
        print(searcher.n_splits_)


    if __name__ == "__main__":
        demostracion_grid_search()
