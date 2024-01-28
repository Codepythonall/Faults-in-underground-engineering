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


b=pd.read_csv(Path(__file__).parent / "Data/Hostrock information.csv")
# Throught the selectbox to dreaw the echart
optionA1 = st.selectbox(
    'Which item do you like best?',
    b.columns.tolist())
r1=b.groupby(optionA1).size()

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




a=pd.read_csv(Path(__file__).parent / "Data/GEoREST_Fault.csv")
fig1 = px.scatter_mapbox(a, 
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
                        range_color=(0, 500),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig1)
st.write ("Figure 1. rock_formation")

fig2 = px.scatter_mapbox(a, 
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
                        range_color=(0, 600),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig2)
st.write ("Figure 2. rock_strat")

fig3 = px.scatter_mapbox(a, 
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
                        range_color=(0, 80),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig3)
st.write ("Figure 3. rock_fr_dens_max")


fig4 = px.scatter_mapbox(a, 
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
                        range_color=(2500, 3000),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig4)
st.write ("Figure 4. rock_dens_max")

fig5 = px.scatter_mapbox(a, 
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

st.plotly_chart(fig5)
st.write ("Figure 5. rock_poro_mean")


fig6 = px.scatter_mapbox(a, 
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
                        range_color=(0, 0.000000000006),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig6)
st.write ("Figure 6. rock_perm_mean")

fig7 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_E_mean',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 100),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig7)
st.write ("Figure 7. rock_E_meanx")


fig8 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_nu_mean',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 0.35),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig8)
st.write ("Figure 8. rock_nu_mean")



fig9 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_K_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 120),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig9)
st.write ("Figure 9. rock_K_max")

fig10 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_G_min',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 40),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig10)
st.write ("Figure 10. rock_G_min")


fig11 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_biot_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0.2, 1.1),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig11)
st.write ("Figure 11. rock_biot_max")

fig12 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_phi_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(20, 70),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig12)
st.write ("Figure 12. rock_phi_max")

fig13 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_c_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 100),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig13)
st.write ("Figure 13. rock_c_max")

fig14 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_ucs_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 420),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig14)
st.write ("Figure 14. rock_ucs_max")


fig15 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_T0_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 32),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig15)
st.write ("Figure 15. rock_T0_max")


fig16 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_lambda_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 5),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig16)
st.write ("Figure 16. rock_lambda_max")

fig17 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='rock_beta_min',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 0.00009),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig17)
st.write ("Figure 17. rock_beta_min")

