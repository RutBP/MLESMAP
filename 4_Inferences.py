from dislib.trees import RandomForestRegressor
import dislib as ds
import numpy as np
from pycompss.api.api import compss_barrier, compss_wait_on
#from sklearn.preprocessing import MinMaxScaler
from dislib.preprocessing import MinMaxScaler
#from minmax_scaler import MinMaxScaler
from dislib.model_selection import KFold
import pickle
from sklearn.metrics import *
from dislib.data.array import *
from dislib.utils import train_test_split
from dislib.data import load_txt_file
import time
import pandas as pd
from datetime import datetime
import xarray as xr
import geopandas as gpd

from geojson import Point, Feature
from turfpy import measurement
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from scipy.interpolate import griddata

import json

periods = ['2']
MAX_DEPTH = 40          #from 1_GridSearch.py
N_EST = 20              #from 1_GridSearch.py
TRY_FEAT = "third"      #from 1_GridSearch.py
Location = "Iceland"

event = "Earthquake to be inferred"
magnitude_event = 6.5
latitude_event = 63.97558791664313
longitude_event = -20.34016532721380
depth_event = 6.3

path = "/path/to/models/"
path2 = "/path/to/Data/"

model = RandomForestRegressor(max_depth=MAX_DEPTH,n_estimators=N_EST,try_features=TRY_FEAT,random_state=0)
model.load_model(path+str(location)+'_model_T'+str(period)+'s_depth'+str(MAX_DEPTH)+'_nestim'+str(N_EST)+'_8feat.save', load_format="pickle")
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_X.load_model(path+'scaler_X_'+str(period)+'s_'+str(location)+'_est'+str(N_EST)+'_dep'+str(MAX_DEPTH)+'_'+str(TRY_FEAT)+'.json')
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.load_model(path+'scaler_Y_'+str(period)+'s_'+str(location)+'_est'+str(N_EST)+'_dep'+str(MAX_DEPTH)+'_'+str(TRY_FEAT)+'.json')
completedata = pd.read_csv(path2+"AllData_"+str(period)+"s.csv")
realEQ= pd.read_csv("/path/to/Real_Data.csv")


latMax = 64.5
latMin = 63.5
lonMax = -19
lonMin = -23

rang_lat = float(latMax-latMin)/40
rang_lon = float(lonMax-lonMin)/40
dfStations = pd.DataFrame()

sitesLat = np.arange(latMin, latMax, rang_lat)
sitesLon = np.arange(lonMin, lonMax, rang_lon)

vecStations = np.array(np.meshgrid(sitesLat,sitesLon)).T.reshape(-1,2)
dfStations['Site_Lat'] = vecStations[:,0]
dfStations['Site_Lon'] = vecStations[:,1]
dfStations = pd.concat([
    dfStations[['Site_Lat', 'Site_Lon']].assign(source='regular'),
    realEQ[['Site_Lat', 'Site_Lon']].assign(source='realEQ')
], ignore_index=True)
print(len(vecStations))
print(len(dfStations))

realEQ_coords = realEQ[['Site_Lat', 'Site_Lon']].values
vecStations_all = np.vstack([vecStations, realEQ_coords])
print(len(vecStations))
print(len(realEQ_coords))
print(len(vecStations_all))

azim1 = np.zeros(len(vecStations_all))
euclDist = np.zeros(len(vecStations_all))
for rj in np.arange(len(vecStations_all)):
    start1 = Feature(geometry=Point((longitude_event, latitude_event))) #Lat-Lon Earthquake
    end1 = Feature(geometry=Point((dfStations['Site_Lon'].iloc[rj], dfStations['Site_Lat'].iloc[rj]))) #Lat-Lon Station
    d1 = measurement.distance(start1,end1,'km')
    azim1[rj] = measurement.bearing(start1,end1) #print(d1/1000)
    euclDist[rj] =  np.sqrt((d1)**2 + (depth_event)**2)

df_InputData_CS_sites = pd.DataFrame()
df_InputData_CS_sites["Site_Lat"] = dfStations["Site_Lat"].values
df_InputData_CS_sites["Site_Lon"] = dfStations["Site_Lon"].values
df_InputData_CS_sites["Magnitude"] = magnitude_event*np.ones(len(vecStations_all))
df_InputData_CS_sites["Hypocenter_Lat"] = latitude_event*np.ones(len(vecStations_all))
df_InputData_CS_sites["Hypocenter_Lon"] = longitude_event*np.ones(len(vecStations_all))
df_InputData_CS_sites["Hypocenter_Depth"] = depth_event*np.ones(len(vecStations_all))
df_InputData_CS_sites["Distance"] = euclDist
df_InputData_CS_sites["Azimuth"] = azim1
df_InputData_CS_sites["Intensity Value "+str(period)+"s"] = 0
df_InputData_CS_sites["y_pred"] = 0
print(df_InputData_CS_sites)
feature_cols = ["Site_Lat","Site_Lon","Magnitude","Hypocenter_Lat","Hypocenter_Lon","Hypocenter_Depth","Distance","Azimuth"]

scaler_X.clip = True
X_test_minmax = scaler_X.transform(df_InputData_CS_sites[feature_cols])
X_test_minmax

inferences = model.predict(X_test_minmax)
print('inferences',inferences)
#print("feature_importance", model.feature_importances_)
y_pred = scaler_y.inverse_transform(inferences.reshape(-1,1))
df_InputData_CS_sites['y_pred'] = y_pred
print(y_pred)
df_event = df_InputData_CS_sites

vecStations_all = pd.DataFrame(vecStations_all, columns=["Site_Lat", "Site_Lon"])
vecStations_all['Site_Lon'] = dfStations['Site_Lon'].values
vecStations_all['Site_Lat'] = dfStations['Site_Lat'].values

lat_true = df_event['Site_Lat'].values
lon_true = df_event['Site_Lon'].values
z_true = np.power(10, df_event['y_pred'].values)

LonMin = vecStations_all['Site_Lon'].min()
LonMax = vecStations_all['Site_Lon'].max()
LatMin = vecStations_all['Site_Lat'].min()
LatMax = vecStations_all['Site_Lat'].max()

llcrnrlon = LonMin - 0.3
urcrnrlon = LonMax + 0.3
llcrnrlat = LatMin - 0.3
urcrnrlat = LatMax + 0.3

region = [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]

size = 300
xi = np.linspace(llcrnrlon, urcrnrlon, size)
yi = np.linspace(llcrnrlat, urcrnrlat, size)
xgrid, ygrid = np.meshgrid(xi, yi)

points = np.column_stack((lon_true, lat_true))
grid_z = griddata(points, z_true, (xgrid, ygrid), method='linear', fill_value=0)

ds = xr.DataArray(grid_z, coords=[yi, xi], dims=["lat", "lon"])

fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection=ccrs.Mercator())
ax.set_extent(region, crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = gl.right_labels = False

im = ax.pcolormesh(xi, yi, ds.values, cmap='turbo', transform=ccrs.PlateCarree(), shading='auto', alpha=0.50)

ax.scatter(dfStations['Site_Lon'], dfStations['Site_Lat'],
           color='gray', edgecolor='black', s=20,
           transform=ccrs.PlateCarree(), label="Estaciones")

ax.scatter(realEQ['Site_Lon'], realEQ['Site_Lat'],
           color='blue', edgecolor='blue', s=20,
           transform=ccrs.PlateCarree(), label="Estaciones")

ax.plot(longitude_event, latitude_event, marker='*', markersize=15,
        color='red', transform=ccrs.PlateCarree(), label='Evento')


cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
cbar.set_label(f"PSA, T={period}s (cm/s²)", fontsize=16)

plt.title("Mapa interpolado de Intensidades", fontsize=24)
plt.legend()
plt.savefig("/path/to/save/plot.png", dpi=300, bbox_inches='tight')

