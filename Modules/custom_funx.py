import rampy as rp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
import os
import numpy as np

def residual(pars, x, data=None, eps=None): #Function definition
    # unpack parameters, extract .value attribute for each parameter
    a1 = pars['a1'].value
    a2 = pars['a2'].value
    a3 = pars['a3'].value
    a4 = pars['a4'].value
    a5 = pars['a5'].value
    a6 = pars['a6'].value
    a7 = pars['a7'].value
    
    f1 = pars['f1'].value
    f2 = pars['f2'].value
    f3 = pars['f3'].value
    f4 = pars['f4'].value
    f5 = pars['f5'].value 
    f6 = pars['f6'].value
    f7 = pars['f7'].value
    
    l1 = pars['l1'].value
    l2 = pars['l2'].value
    l3 = pars['l3'].value
    l4 = pars['l4'].value
    l5 = pars['l5'].value
    l6 = pars['l6'].value
    l7 = pars['l7'].value
    
    
    # Using the Gaussian model function from rampy
    # peak1 = rp.lorentzian(x,a1,f1,l1)
    # peak2 = rp.lorentzian(x,a2,f2,l2)
    # peak3 = rp.lorentzian(x,a3,f3,l3)
    # peak4 = rp.lorentzian(x,a4,f4,l4)
    # peak5 = rp.lorentzian(x,a5,f5,l5)
    # peak6 = rp.lorentzian(x,a6,f6,l6)
    # peak7 = rp.lorentzian(x,a7,f7,l7)
    #  chi-square         = 0.23088034
    # reduced chi-square = 2.5123e-04
    # peak1 = rp.pseudovoigt(x,a1,f1,l1,0)
    # peak2 = rp.pseudovoigt(x,a2,f2,l2,0)
    # peak3 = rp.pseudovoigt(x,a3,f3,l3,0.2)
    # peak4 = rp.pseudovoigt(x,a4,f4,l4,0.2)
    # peak5 = rp.pseudovoigt(x,a5,f5,l5,0)
    # peak6 = rp.pseudovoigt(x,a6,f6,l6,0.1)
    # peak7 = rp.pseudovoigt(x,a7,f7,l7,0)

    # peak1 = rp.pseudovoigt(x,a1,f1,l1,0)
    # peak2 = rp.pseudovoigt(x,a2,f2,l2,0)
    # peak3 = rp.pseudovoigt(x,a3,f3,l3,0.2)
    # peak4 = rp.pseudovoigt(x,a4,f4,l4,0.2)
    # peak5 = rp.pseudovoigt(x,a5,f5,l5,0)
    # peak6 = rp.pseudovoigt(x,a6,f6,l6,0)
    # peak7 = rp.pseudovoigt(x,a7,f7,l7,0)
    
    peak1 = rp.pseudovoigt(x,a1,f1,l1,0)
    peak2 = rp.pseudovoigt(x,a2,f2,l2,0)
    peak3 = rp.pseudovoigt(x,a3,f3,l3,0.2)
    peak4 = rp.pseudovoigt(x,a4,f4,l4,0.2)
    peak5 = rp.pseudovoigt(x,a5,f5,l5,0)
    peak6 = rp.pseudovoigt(x,a6,f6,l6,0.1)
    peak7 = rp.pseudovoigt(x,a7,f7,l7,0)


    model = peak1 + peak2 + peak3 + peak4 + peak5 + peak6 + peak7 # The global model is the sum of the Gaussian peaks
    
    if data is None: # if we don't have data, the function only returns the direct calculation
        return model, peak1, peak2, peak3, peak4, peak5, peak6, peak7
    if eps is None: # without errors, no ponderation
        return (model - data)
    return (model - data)/eps # with errors, the difference is ponderated


def create_residuals_df(res_stack,data):
    '''
    Create the dataframe ready to plot
    Input :
        res_stack : numpy converted to dataframe (nd array)
        data : the raw data also added to the dataframe (numpy)
    output : 
        res_df : A dataframe with columns to plot
    '''
    # Create dataframe
    res_df = pd.DataFrame(res_stack)
    col_names = ['Fit']
    nCol = res_df.shape
    for k in range(0,nCol[1]):
        if k > 0:
            col_names.append('Peak '+ str(k))
    res_df.columns = col_names
    data_df = pd.DataFrame(data, columns = ['Data'])        
    res_df = pd.concat([data_df, res_df], axis=1)
    return res_df

def plot_fit(res_df,x_fit):
    '''
    Creates plot of the peaks from the residual dataframe created
    Input 
        res_df : Dataframe
        x_fit 
    '''
    sh = res_df.shape
    cols = res_df.columns
    plt.figure(figsize=(10, 6))  
    # plt.style.use('ggplot')
    for x in range(0,sh[1]):
        if cols[x]== 'Data':
            print([cols[x]])
            plt.plot(x_fit,res_df[cols[x]],'k:',label = cols[x], linewidth = 3.5)
        elif cols[x]== 'Fit':
            plt.plot(x_fit,res_df[cols[x]],'r', markerfacecolor="None",label = cols[x])
        else:   
            plt.fill_between(x_fit,res_df[cols[x]],alpha=0.7,label =cols[x])

    plt.xlabel("Frequency, cm$^{-1}$")
    plt.ylabel("Normalized absorbance (OD)")
    plt.legend(loc="upper left")
    plt.gca().invert_xaxis()
    # plt.title(filename)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    return plt
    # print(plt.style.available)
    
def plot_res(x_fit,residual):
    plt.figure(figsize=(10, 6)) 
    plt.plot(x_fit,residual,'k')
    plt.gca().invert_xaxis()
    plt.title("Residuals")
    plt.hlines(y= 0,xmin= 1300, xmax= 1800, color='grey', linestyle ='dashed', linewidth = 1.5)
    plt.xlabel("Frequency, cm$^{-1}$")
    plt.ylabel("Residuals")
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    return plt

def cal_peak(peak1,peak2, peak3, peak4, x_fit):
    '''
    Protein to lipid ratio based on the area under the peak
    '''
    a_p1 = trapezoid(peak1,x_fit)
    a_p2 = trapezoid(peak2,x_fit)
    sum_lipid = a_p1+a_p2
    a_p3 = trapezoid(peak3,x_fit)
    a_p4 = trapezoid(peak4,x_fit)
    sum_protein = a_p3+a_p4
    ratio = sum_protein/sum_lipid

    return ratio 

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def load_files(dir_name):
    '''
    Load all files from the directory path
    The dataframe output and column naming is based on the temparature.
    With index as the x axis or Frequency
    '''
    init_temp = 10
    c = 0
    all_data = pd.DataFrame()
    for file in os.listdir(dir_name):
        filename = file
        if '_XY' in filename:
            full_file = os.path.join(dir_name,filename)
            # print(file_num)
            data_temp = init_temp+c
            # print(filename + ' : '+str(data_temp))
            c += 3 
            colx = "x_"+str(data_temp)
            coly = "y_"+str(data_temp)
            col_nam = [colx , coly]
            data = pd.read_csv(full_file, delimiter = "\t", names = col_nam)
            all_data = pd.concat([all_data,data[coly]], axis=1)
            
    all_data = pd.concat([data[colx], all_data], axis=1)    
    all_data = all_data.rename(columns = {colx:'x'})
    return all_data

def correct_spectra(data,roi,pol_order = 3):
    '''
    Performs background correction - baseline subtraction and data trimming.
    Input   - data : as Dataframe
            - roi : Region of interest as np array -> Usage : roi = np.array([(1347,1365),(1774,1800)])
            - pol_order : uses polynomial function for baseline correction
    output  - data_corr_trim : Trimmed data as Dataframe
            - data_corr : Corrected data, as DataFrame
            - data_base : Baseline data, as Dataframe
    
    '''
    
    x = data['x'].to_numpy()
    data_corr_trim = pd.DataFrame()
    data_base = pd.DataFrame()
    data_corr = pd.DataFrame()
    data_corr = pd.concat([data_corr,data['x']], axis = 1)
    data_base = pd.concat([data_base,data['x']], axis = 1)
    for col in data.columns:
        if col != 'x':
            y = data[col].to_numpy()            
            y_corr, y_base = rp.baseline(x,y,roi,'poly',polynomial_order = pol_order)        
            data_corr = pd.concat([data_corr,pd.DataFrame(y_corr,columns=[col])], axis=1)
            data_base = pd.concat([data_base,pd.DataFrame(y_base,columns=[col])], axis=1)

            x_fit = pd.DataFrame(x[np.where((x > roi[0,0])&(x < roi[1,1]))], columns= ['x'])
            y_fit = pd.DataFrame(y_corr[np.where((x > roi[0,0])&(x < roi[1,1]))],columns=[col])
            
            if 'x' in data_corr_trim.columns:
                data_corr_trim = pd.concat([data_corr_trim,y_fit],axis=1)
            else:
                data_corr_trim = pd.concat([data_corr_trim,x_fit], axis = 1)
    return data_corr_trim, data_corr, data_base


def norm_spectra(data_corr_trim,method_type = "intensity"):
    data_norm = pd.DataFrame()
    for col_name in data_corr_trim.columns:
        if col_name != 'x':
            y_fit_norm_intensity = pd.DataFrame(rp.normalise(data_corr_trim[col_name],x = data_corr_trim['x'],method = method_type))
            data_norm = pd.concat([data_norm,y_fit_norm_intensity],axis = 1)       
    return data_norm
