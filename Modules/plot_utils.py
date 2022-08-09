import matplotlib.pyplot as plt

def plot_spectral_corrections(data,cor_spectra,bdata,roi):
    '''A grid plot describing the baseline corrections, with polynomial fits in respect to the original spectra.

    Paramaeters
    -----------
    data : Pandas DataFrame 
        The original dataframe of the raw data loaded
    cor_spectra : Pandas DataFrame
        Corrected spectra after baseline correction
    bdata : Pandas DataFrame
        The baseline obtained with the desired polynomial degree obtained for each spectra        

    Returns
    -----------
    plt : matplotlib.pyplot.plot object 
    '''

    f = plt.figure('Spectral Corrections',figsize=(16,12))
    for i,cols in enumerate(data.columns):
        if cols != 'x' :   
            plt.subplot(6,5,i)
            plt.plot(data['x'],bdata[cols], label='Baseline')
            plt.plot(data['x'],cor_spectra[cols],'r-.' ,label='Corrected')
            plt.plot(data['x'],data[cols],'k:' ,label='Original')
            plt.xlim(roi[0,0],roi[1,1])
            y_max = data[cols].max()
            # print(y_max)
            plt.ylim(-0.1,y_max)
            plt.title(cols,loc = 'right', fontweight = 'bold')
            plt.hlines(y= 0,xmin= roi[0,0], xmax= roi[1,1], color='grey', linestyle ='dashed', linewidth = 1.5)
            plt.gca().invert_xaxis()
            plt.gca().sharex = True
            plt.gca().sharey = True
            plt.gca().set_xlabel = 'Frequency, cm$^{-1}$'
        plt.figlegend(['Baseline', 'Corrected', 'Original'])  
    plt.tight_layout()
    # plt.show()
    return plt

def plot_norm_signals(normSpectra):
    f = plt.figure('Normalized Signal',figsize=(16,12))
    for s,cols in enumerate(normSpectra.columns):
        if cols != 'x' :   
            plt.subplot(6,5,s)
            plt.plot(normSpectra['x'],normSpectra[cols], label=cols)
            plt.ylim(-0.1,1.01)
            plt.title('Norm Signal_'+cols)
            plt.hlines(y= 0, xmin = normSpectra['x'].iloc[0], xmax = normSpectra['x'].iloc[-1], color='grey', linestyle ='dashed', linewidth = 1.5)
            plt.gca().invert_xaxis()   
    
    plt.tight_layout()
    return plt


