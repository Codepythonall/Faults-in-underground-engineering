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


data0=[]
for i in a.groupby("country").size().items():
    data0.append({"name":i[0],"value":i[1]})

def render_world():
    formatter = JsCode(
        "function (params) {"
        + "var value = (params.value + '').split('.');"
        + "value = value[0].replace(/(\d{1,3})(?=(?:\d{3})+(?!\d))/g, '$1,');"
        + "return params.seriesName + '<br/>' + params.name + ': ' + value;}"
    ).js_code

    with open(Path(__file__).parent / "Data/0World.json", "r",encoding="utf-8") as f:
        map = Map(
            "world",
            json.loads(f.read())
        )
    options = {
        "title": {
            "text": "Distribution of global underground engineering",
            "subtext": "Data from the University of Edinburgh",
            "sublink": "https://inducedearthquakes.org/",
            "left": "right",
        },
        "tooltip": {
            "trigger": "item",
            "showDelay": 0,
            "transitionDuration": 0.2,
            "formatter": formatter,
        },
        "visualMap": {
            "left": "right",
            "min": 1,
            "max": 42,
            "inRange": {
                "color": [                                                                       
                    '#800080',
                    '#0000FF',
                    '#00FFFF',
                    '#008000',
                    '#FFFF00',
                    '#FFA500',
                    '#FF0000',
                ]
            },
            "text": ["High", "Low"],
            "calculable": True,
        },
        "toolbox": {
            "show": True,
            "left": "left",
            "top": "top",
            "feature": {
                "dataView": {"readOnly": False},
                "restore": {},
                "saveAsImage": {},
            },
        },
        "series": [
            {
                "name": "World CCS project distribution",
                "type": "map",
                "roam": True,
                "map": "world",
                "emphasis": {"label": {"show": True}},
                "textFixed": {"Alaska": [20, -20]},
                "data": data0,
            }
        ],
    }
    st_echarts(options, map=map)


render_world()


a1=pd.read_csv(Path(__file__).parent / "Data/GEoREST_Fault.csv")

data1=[]
for i in a1.groupby("project_type").size().items():
    data1.append({"name":i[0],"value":i[1]})
def render_world():
    formatter = JsCode(
        "function (params) {"
        + "var value = (params.value + '').split('.');"
        + "value = value[0].replace(/(\d{1,3})(?=(?:\d{3})+(?!\d))/g, '$1,');"
        + "return params.seriesName + '<br/>' + params.name + ': ' + value;}"
    ).js_code

    with open(Path(__file__).parent / "Data/0World.json", "r",encoding="utf-8") as f:
        map = Map(
            "world",
            json.loads(f.read())
        )
    options1 = {
        "title": {
            "text": "World CCS project distribution",
            "subtext": "Data from www.globalccsinstitute.com",
            "sublink": "https://inducedearthquakes.org/",
            "left": "right",
        },
        "tooltip": {
            "trigger": "item",
            "showDelay": 0,
            "transitionDuration": 0.2,
            "formatter": formatter,
        },
        "visualMap": {
            "left": "right",
            "min": 0,
            "max": 50,
            "inRange": {
                "color": [                                                                       
                    '#E6E6FA',
                    '#8470FF',
                    '#7B68EE',
                    '#6A5ACD',                   
                    '#483D8B',
                    '#191970',                    
                ]
            },
            "text": ["High", "Low"],
            "calculable": True,
        },
        "toolbox": {
            "show": True,
            "left": "left",
            "top": "top",
            "feature": {
                "dataView": {"readOnly": False},
                "restore": {},
                "saveAsImage": {},
            },
        },
        "series": [
            {
                "name": "World CCS project distribution",
                "type": "map",
                "roam": True,
                "map": "world",
                "emphasis": {"label": {"show": True}},
                "textFixed": {"Alaska": [20, -20]},
                "data": data1,
            }
        ],
    }
    st_echarts(options1, map=map)


render_world()

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
