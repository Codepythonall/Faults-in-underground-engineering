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


b=pd.read_csv(Path(__file__).parent / "Data/Fault information.csv")
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
                        color='fault_strike_max',
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

st.plotly_chart(fig1)
st.write ("Figure 1. fault_strike_max")

fig2 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_dip_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(10, 90),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig2)
st.write ("Figure 2. fault_dip_max")

fig3 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_dip_dir_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(10, 90),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig3)
st.write ("Figure 3. fault_dip_dir_max")

fig4 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_name',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(10, 90),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig4)
st.write ("Figure 4. fault_name")


fig6 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_type',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(10, 90),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig6)
st.write ("Figure 6. fault_type")

fig7 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_thick_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 230),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig7)
st.write ("Figure 7. fault_thick_max")

fig8 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_dist_inj',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 1600),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig8)
st.write ("Figure 8. fault_dist_inj （m) ")


fig9 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_inj_depth_min',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 5000),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig9)
st.write ("Figure 9. fault_inj_depth_min")

fig10 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_dens_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 2700),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig10)
st.write ("Figure 10. fault_dens_max (kg/m3)")

fig11 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_poro_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 0.45),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig11)
st.write ("Figure 11. fault_poro_max")

fig12 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_perm_max',
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

st.plotly_chart(fig12)
st.write ("Figure 12. fault_perm_max")


fig13 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_Kn_max',
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

st.plotly_chart(fig13)
st.write ("Figure 13. fault_Kn_max")



fig14 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_Ks_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 160),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig14)
st.write ("Figure 14. fault_Ks_max")


fig15 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_psi_min',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 9),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig15)
st.write ("Figure 15. fault_psi_min")



fig16 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_E_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 10),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig16)
st.write ("Figure 16. fault_E_max")


fig17 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_nu_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 0.3),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig17)
st.write ("Figure 17. fault_nu_max")

fig18 = px.scatter_mapbox(a, 
                        lat='latitude', 
                        lon='longitude', 
                        color='fault_phi_max',
                        color_continuous_scale=[                
                            '#800080',
                            '#0000FF',
                            '#00FFFF',
                            '#008000',
                            '#FFFF00',
                            '#FFA500',
                            '#FF0000',
                        ],
                        range_color=(0, 45),
                        mapbox_style="carto-positron",
                        opacity=0.5,
                        labels={'diff_percentage':'Difference Percentage'},
                        center={"lat": 53, "lon": -113},
                        zoom=0.1)

st.plotly_chart(fig18)
st.write ("Figure 18. fault_phi_max")

