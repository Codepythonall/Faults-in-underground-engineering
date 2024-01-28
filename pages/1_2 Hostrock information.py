# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:15:25 2023

@author: Lifeng Xu, Bo Zhang, Rick Chalaturnyk
"""

import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
import json
from streamlit_echarts import Map
from streamlit_echarts import JsCode
from streamlit_echarts import st_echarts
import plotly.express as px
from streamlit_globe import streamlit_globe
import sklearn
import streamlit as st
import sys
import requests
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.rcParams['axes.unicode_minus']=False
from sklearn import datasets
from numpy import argsort
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns ## 设置绘图的主题
import os
sys.path.append(os.getcwd())
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer,StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV,StratifiedKFold
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso,LassoLars,ElasticNetCV,LogisticRegression,LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import r2_score, explained_variance_score as EVS, mean_squared_error as MSE
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, mean_absolute_error,classification_report
from sklearn.neural_network import MLPClassifier,MLPRegressor
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency
from pandas.plotting import parallel_coordinates
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import distance


from streamlit_echarts import st_echarts
import json
from streamlit_echarts import Map
from streamlit_echarts import JsCode
import sys,os
sys.path.append(os.getcwd())
from pathlib import Path
st.set_page_config(layout="wide")

a=pd.read_csv(Path(__file__).parent / "Data/GEoREST_Fault.csv")





fig1 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='project_type',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 11),
                        mapbox_style="carto-positron",
                        opacity=0.5, 
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig1)
st.write ("Figure 1. project_type")


fig4 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_formation',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 12),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig4)
st.write ("Figure 4. rock_formation")



fig5 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_strat',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 12),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig5)
st.write ("Figure 5. rock_strat")


fig7 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_fr_dens_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 12),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig7)
st.write ("Figure 7. rock_fr_dens_max")


fig9 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_dens_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(2450, 2700),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig9)
st.write ("Figure 9. rock_dens_max")


fig12 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_poro_mean',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 0.5),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig12)
st.write ("Figure 12. rock_poro_mean")



fig14 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_perm_mean',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 1e-12),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig14)
st.write ("Figure 14. rock_perm_mean")


st.write("Data is collected by HiQuake (https://inducedearthquakes.org/)")

a=pd.read_csv(Path(__file__).parent / "Data/GEoREST_Fault.csv")
st.write("Item type: rock_perm_mean, fault_perm_max, seism_events, moment_max...")
st.text_input("Type the interested item", key="columns")
if st.session_state.columns:
    r1=a.groupby(st.session_state.columns).size()
    
    options1 = {
        "color":'#ff4060',
        "tooltip": {
      "trigger": 'axis',
      "axisPointer": {
        "type": 'shadow'
      }
    },
        "xAxis": {
            "type": "category",
            "data": r1.index.tolist(),
            "axisTick": {"alignWithLabel": True},
        },
        "yAxis": {"type": "value"},
        "series": [
            {"data": r1.values.tolist(), "type": "bar"}
        ],
    }
    st_echarts(options=options1)




# Throught the selectbox to dreaw the echart
optionA1 = st.selectbox(
    'Which item do you like best?',
    a.columns.tolist())
r1=a.groupby(optionA1).size()

options1 = {
    "color":'#ff4060',
    "tooltip": {
  "trigger": 'axis',
  "axisPointer": {
    "type": 'shadow'
  }
},
    "xAxis": {
        "type": "category",
        "data": r1.index.tolist(),
        "axisTick": {"alignWithLabel": True},
    },
    "yAxis": {"type": "value"},
    "series": [
        {"data": r1.values.tolist(), "type": "bar"}
    ],
}
st_echarts(options=options1)




st.write("Advanced search")

optionB1 = st.selectbox(
    'Which energy class are you interested in?',
    a["project_type"].unique())
optionB2 = st.slider('latitude',-90.,a.latitude.max(),(-40.,80.))
optionB3 = st.slider('longitude',-180.,a.longitude.max(),(-150.,200.))
optionB4 = st.multiselect(
    'Please select Parameters you are interested in',
    a.columns,
    ["sub_class", "rock_perm_mean", "fault_perm_max", "seism_events", "moment_max", "inj_rate_max", "rock_fr_dens_max", "rock_ucs_max", "site_depth_bas_max", "site_sv_max", "site_shmax_max", "site_shmin_max", "fault_dip_min", "site_T_max"])
FinalSelect=a[(a["moment_max"]==optionB1)&(a.latitude>optionB2[0])&(a.latitude<optionB2[1])
              &(a.longitude>optionB3[0])&(a.longitude<optionB3[1])][optionB4]

FinalSelect







