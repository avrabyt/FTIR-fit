import lmfit
import rampy as rp #Charles' libraries and functions

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
from Modules.custom_funx import *


# Loading all the data

dir_name = 'Data/Thylakoids/qg27/'
data = load_files(dir_name) # The output comes as Dataframe
# print(data) # To visualize all the data
# print(data.columns) # To check the columns

# To plot
# fig = px.line(data_frame=data, x=data.index, y=data['y_10'])
#  fig.show()

data_array = data.to_numpy() # converting all the datas to numpy 2D array for easy data handing and manipulation
x = data.index # Making the x axis from the dataframe indices 

# Visualize by potting
# plt.figure()
# plt.plot(x, data_array)
# plt.show()

# Choosing the region of interest
roi = np.array([(1347,1365),(1774,1800)])
roi[1,1]

y = data_array[:,9]

# Baseline Fitting and trimming the data
# For Base line correction for each array
# Base line correction
y_corr, y_base = rp.baseline(x,y,roi,'poly',polynomial_order=3)
# print(len(data_array))


    




