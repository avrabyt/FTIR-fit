import rampy as rp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapezoid
import os
import numpy as np
import lmfit
import pybroom as br

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


def create_residuals_df(res_stack,y_fit):
    '''
    Creates the dataframe ready to plot

    Parameters
    -----------
        res_stack : ND Array
            N dimensional array with all the peaks stacked together
        y_fit : 1D Pandas
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
    res_df = pd.concat([y_fit, res_df], axis=1)
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
        if '_XY' or 'XY' in filename:
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

    Parameters
    -----------
    data : Dataframe
        The raw data/spectra obtained from the loaded files
    roi : np array -> Usage : roi = np.array([(1347,1365),(1774,1800)])
        The region of interest, used for polynomial fit of the spectra
    pol_order : Integer
        Uses polynomial function for baseline correction of the spectra
        rampy.baseline() module used, refer to Rampy python package
    
    Returns
    -----------
    data_corr_trim : DataFrame
        Trimmed spectra
    data_corr : DataFrame
        Corrected spectra(but not trimmed) hence can be used for comparing with raw spectra 
    data_base : DataFrame
        Baseline obtained for each spectra

    This function is later used for plotting and comparasion with raw data.
    Refer to plot_utils.plot_spectral_corrections()

    Usage
    ------    
    trim_data,cor_spectra,bdata = correct_spectra(data,roi,pol_order = 3)
    '''
    # Get the x axis as nump
    x = data['x'].to_numpy()
    # Initialize the dataframe to concat later
    data_corr_trim = pd.DataFrame()
    data_base = pd.DataFrame()
    data_corr = pd.DataFrame()
    # Already merge the x axis except in case of data-trim since length it will alter
    data_corr = pd.concat([data_corr,data['x']], axis = 1)
    data_base = pd.concat([data_base,data['x']], axis = 1)
    for col in data.columns:
        if col != 'x':
            y = data[col].to_numpy()            
            y_corr, y_base = rp.baseline(x,y,roi,'poly',polynomial_order = pol_order)        
            # Merging to the dataframe
            data_corr = pd.concat([data_corr,pd.DataFrame(y_corr,columns=[col])], axis=1)
            data_base = pd.concat([data_base,pd.DataFrame(y_base,columns=[col])], axis=1)
            # Trimming / Correction 
            x_fit = pd.DataFrame(x[np.where((x > roi[0,0])&(x < roi[1,1]))], columns= ['x'])
            y_fit = pd.DataFrame(y_corr[np.where((x > roi[0,0])&(x < roi[1,1]))],columns=[col])
            # Add the y_fit to the dataframe
            data_corr_trim = pd.concat([data_corr_trim,y_fit],axis = 1)
    # Add the x-axis to the dataframe
    data_corr_trim = pd.concat([x_fit,data_corr_trim], axis = 1)
    return data_corr_trim, data_corr, data_base


def norm_spectra(data_corr_trim,method_type = "intensity"):
    '''
    Perform normalization on the trimmed data/spectra.

    Parameters
    ----------
    data_corr_trim : DataFrame
        The corretcted and trimmed dataframe. Refer to correct_spectra(data,roi,pol_order = 3) 
    method_type : Str
        Type of method to normalize spectra. Using Rampy package. rp.normalise()
        Refer to rampy package for more types
    
    '''
    data_norm = pd.DataFrame()
    for col_name in data_corr_trim.columns:
        if col_name != 'x':
            y_fit_norm_intensity = pd.DataFrame(rp.normalise(data_corr_trim[col_name],x = data_corr_trim['x'],method = method_type))
            data_norm = pd.concat([data_norm,y_fit_norm_intensity],axis = 1)   
    data_norm = pd.concat([data_corr_trim['x'],data_norm],axis = 1)            
    return data_norm

def run_multifit(normSpectra,params, algo = 'leastsq', message = True):
    '''
    Run fit along the stored spectra simultaneously.

    Parameters
    -----------
    normSpectra : DataFrame
        The normalized spectra which will be used for running the lmfit
    params : lmfit.Parameters
        Requires to be initialized as a script
    algo : Str
        Algorithim which lmfit will use to fit 
    message : Boolean
        If True, a message will be printed on the progress
    
    Return
    ----------
    df_stats : DataFrame
        Consisiting of the lmfit report statistics, such as Chi Square, nev etc
    df_variables : DataFrame
        All the fitting variables , row wise accesible for each spectra, using loc argument 
    df_residuals : DataFrames
        Residuals are stored as columns for each spectra
    '''
    # Initialize storing DataFrames
    df_stats = pd.DataFrame()
    df_variables = pd.DataFrame()
    df_residual = pd.DataFrame()
    df_ready_to_plot = pd.DataFrame()
    df_residual = pd.concat([df_residual,normSpectra['x']],axis = 1) 

    for cols in normSpectra:  
        if cols!='x':
            
            # Assigining x and y axis to work for fits
            x_fit = normSpectra['x']
            y_fit = normSpectra[cols]
            
            # Running individual fit
            result = lmfit.minimize(residual, params, method = algo, args=(x_fit,y_fit))
            
            ##----- IMPORTANT -------###
            # IT Changes based on the total no of peaks/components
            sum_peak, peak1,peak2,peak3,peak4,peak5,peak6, peak7 = residual(result.params,x_fit)
            res_stack = np.column_stack((sum_peak,peak1,peak2,peak3,peak4,peak5,peak6,peak7))
            # print(res_stack)
            # Calling custom function to create dataframe ready-to-plot
            res_df = create_residuals_df(res_stack,y_fit)
            
            # len_res_df = len(res_df)
            # res_df.index = [cols]*len_res_df # Adding index for unique ID
            # print(len(res_df.columns))
            
            stat_glance = br.glance(result)
            stat_glance.index = [cols]

            dt = br.tidy(result)
            nl = len(dt)
            dt.index = [cols]*nl # Adding index for unique ID

            df_ready_to_plot = pd.concat([df_ready_to_plot,res_df], axis = 0)
            
            df_variables = pd.concat([df_variables,dt])
            df_stats = pd.concat([df_stats,stat_glance])
            df_residual = pd.concat([df_residual,pd.DataFrame(result.residual,columns=[cols])],axis = 1) 
            # print(len(df_ready_to_plot.columns))
            if message is True:
                print(cols +"..completed ...")

    return df_stats, df_variables, df_residual, df_ready_to_plot 
