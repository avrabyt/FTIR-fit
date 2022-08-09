import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from Modules.custom_funx import *


# Loading all the data

dir_name = 'Data/'
data = load_files(dir_name) 
# print(data)
roi = np.array([(1347,1365),(1774,1800)])
trim_data,cor_spectra,bdata = correct_spectra(data,roi,pol_order = 3)
norm_spectra = norm_spectra(trim_data)

# print(norm_spectra.columns)

plt.figure(figsize=(16,12))
for i,cols in enumerate(norm_spectra.columns):
    if i == 0 : 
        i +=1      
    plt.subplot(5,5,i)
    plt.plot(data['x'],bdata[cols], label='Baseline')
    plt.plot(data['x'],cor_spectra[cols],'r-.' ,label='Corrected')
    plt.plot(data['x'],data[cols],'k:' ,label='Original')
    plt.xlim(roi[0,0],roi[1,1])
    plt.ylim(-0.1,0.4)
    plt.title(cols,loc = 'right', fontweight = 'bold')
    plt.hlines(y= 0,xmin= roi[0,0], xmax= roi[1,1], color='grey', linestyle ='dashed', linewidth = 1.5)
    plt.gca().invert_xaxis()
    plt.gca().sharex = True
    plt.gca().sharey = True
    plt.gca().set_xlabel = 'Frequency, cm$^{-1}$'
plt.figlegend(['Baseline', 'Corrected', 'Original'])  
plt.tight_layout()
# plt.supxlabel('Frequency, cm$^{-1}$')
# plt.supylabel('absorbance (OD)')
plt.show()



    



    




