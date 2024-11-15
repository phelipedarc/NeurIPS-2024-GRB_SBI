#IMPORTS AND FUNCTIONS:
import os, sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from torch import nn
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm
from scipy import stats as st
from sbi import utils as utils_sbi
from sbi import analysis as analysis
from sbi.inference import prepare_for_sbi, simulate_for_sbi, SNLE, ratio_estimator_based_potential, SNRE_A,SNRE,SNPE
from sbi.inference import likelihood_estimator_based_potential, ImportanceSamplingPosterior, MCMCPosterior
import torch
from sbi.utils.get_nn_models import posterior_nn
import pickle
import math


import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import afterglowpy as grb
import h5py
import numpy as np
#import bisect
import math
import scipy
from scipy import interpolate
#from pyphot import (unit, Filter)
from scipy.integrate import quad
import matplotlib as mpl
from astropy.time import Time
import copy

import nmma.em.io as io
from nmma.em.model import SimpleKilonovaLightCurveModel,GRBLightCurveModel, SVDLightCurveModel, KilonovaGRBLightCurveModel, GenericCombineLightCurveModel
from nmma.em import training, utils, model_parameters
from sncosmo.bandpasses import _BANDPASSES

from scipy.special import logsumexp

def corner_plot(posterior_samples, save=False):
    param_ranges = (
    (50, 60),       # logE0
    (-6, 2),        # logn0
    (-2, -0.5),     # logthetac
    (-6, -0.3),     # logepse
    (-6, -0.3),     # logepsb
    (2.01, 2.9),    # p
    (-5.0, 0.0),    # logxiN
    (-3, -1),       # logmej1
    (-1, -0.5),     # logvej1
    (1, 5),         # beta1
    (-2, 0.5),      # logkappa_r1
    (0.2, 0.4),     # Ye1
    (-3, -1),       # logmej2
    (-2, -1),       # logvej2
    (1, 5),         # beta2
    (-0.5, 2),      # logkappa_r2
    (0.1, 0.2)      # Ye2
    )

    import corner
    import matplotlib as mpl
    colormap_names = ['magma']
    title_format = "{0:.1f}"
    cmap = plt.cm.get_cmap('magma')
    colors = cmap(np.linspace(0.25,0.85,17))
    textsize=12
    #quantiles_f=[0.05,0.5, 0.95]
    quantiles_f=[0.16,0.5, 0.84]
    save=True
    #colors = ['lightcoral', 'olive', 'mediumpurple', 'teal','burlywood']
    fig = corner.corner(posterior_samples, 
                        label_kwargs={'size':20},show_titles=False, 
                                   quantiles=quantiles_f,
                                    title_quantiles= quantiles_f,
                                   levels=[0.1,0.32, 0.68, 0.95], 
                                   labels=parameter_names,color='black',bins=20,
                                   contour_kwargs={"linewidths": 1, "colors": "black"},truths = theta_true, truth_color='skyblue',
                                   hist_kwargs={"histtype": "stepfilled", "linewidth": 4, "alpha": 1.0}, smooth=1.0, range=param_ranges)
    corner_axes = np.array(fig.get_axes()).reshape(posterior_samples.shape[-1], posterior_samples.shape[-1])


    figure2 = corner.corner(posterior_samples, show_titles=True, fig=fig, 
                            quantiles=quantiles_f,
                            title_quantiles =quantiles_f,
                           levels=[0.1,0.32, 0.68, 0.95], 
                           title_kwargs={"fontsize": 24},
                           bins=20,
                            hist_kwargs={"histtype": "step", "linewidth": 4, "alpha": 0.01},smooth=1.0, alpha=0.01, range=param_ranges)




    #cmap = plt.cm.get_cmap('gist_rainbow', posterior_samples_final.shape[-1])
    #colors = [cmap(i) for i in range(sampler_flatchain.shape[-1])]
    #colors = ['lightcoral', 'olive', 'mediumpurple', 'teal','burlywood']


    for i in range(posterior_samples.shape[-1]):
        corner_axes[i, 0].tick_params(labelsize=textsize)
        corner_axes[-1, i].tick_params(labelsize=textsize)


    k=0
    for ax in np.diag(corner_axes):
        # for child in corner_axes[i, 0].get_children()[:-6]:
        #     child.set_color(colors[i])
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
        children = np.array(ax.get_children())
        children[0].set_color(colors[k])
        children[0].set_lw(2)
        k+=1

    for i in range(len(corner_axes)):
        for j in range(i):
            ax = corner_axes[i, j]

            children = np.array(ax.get_children())
            children[0].set_color(colors[j])



            children[1].set_edgecolor(colors[j])
            # children[2].set_edgecolor(colors[j]) #grid
            children[3].set_edgecolor(colors[j])
            children[4].set_edgecolor(colors[j])
            children[5].set_edgecolor(colors[j])
            children[6].set_edgecolor(colors[j])




            start_color = mpl.colors.to_rgba(colors[j])
            mesh = ax.collections[1]

             # Create a LinearSegmentedColormap from white to the specified color
            cmap_data = {
                'red':   ((0.0, start_color[0], start_color[0]), (1.0, 1.0, 1.0)),
                'green': ((0.0, start_color[1], start_color[1]), (1.0, 1.0, 1.0)),
                'blue':  ((0.0, start_color[2], start_color[2]), (1.0, 1.0, 1.0)),
            }
            cmap = mpl.colors.LinearSegmentedColormap('WhiteToColor', cmap_data)
            mesh.set_cmap(cmap)

        plt.subplots_adjust(top=0.9)  # Adjust title position


    axes = np.array(fig.axes).reshape((17, 17))
    for i in range(4):
        ax = axes[i,i]
        label0 = ['NPE Model','','','']
        #valor mais provavel: ---- no
        # ax.axvline(np.quantile(posterior_samples, 0.5, axis=0)[i],  linestyle='-', color='k', label=label0[i])
        # ax.axvline(np.quantile(posterior_samples, 0.05, axis=0)[i],  linestyle='--', color='k')
        # ax.axvline(np.quantile(posterior_samples, 0.95, axis=0)[i],  linestyle='--', color='k')
    if save==True:
        plt.savefig('/tf/astrodados/phelipedata/NEURIPS2025/GRB_mdn_600.pdf',dpi=300, bbox_inches='tight')   
    plt.show()   

def plot_fit_lc(posterior, n_sample):
    samples = posterior.sample((n_sample,), x=x_o)
    samples = np.array(samples)
    
    cmap = plt.cm.get_cmap('turbo')
    colores = cmap(np.linspace(0.25, 0.85, 11))
    plt.figure(dpi=120, figsize=(15, 6))

    shift_factors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Factors of 10 for each band
    fmts = ['o', 's', '^', 'D', 'X', 'v', 'P', 'H', '>', '<']
    bands_work = ['U', 'R', 'I', 'Z', 'H', 'K', 'J', 'F277W', 'F356W', 'F444W']

    for i in range(len(samples)):
        logE0, logn0, logthetac, logepse, logepsb, p, logxiN, logmej1, logvej1, beta1, logkappa_r1, Ye1, logmej2, logvej2, beta2, logkappa_r2, Ye2 = samples[i]
        
        # Define the luminosity distance of the source
        dlsource = 291  # in [Mpc]

        # The KN magnitude lightcurve is:
        _, mag_kn = Me2017x2_model_ejecta(10**logmej1, 10**logvej1, beta1, 10**logkappa_r1, Ye1, 10**logmej2, 10**logvej2, beta2, 10**logkappa_r2, Ye2, dlsource)

        # The GRB magnitude lightcurve is:
        mag_grb = GRB_model_lc(logE0, logn0, logthetac, logepse, logepsb, p, logxiN, dlsource)

        # Combine the two!
        total_mag = {}
        total_mag_app = {}
        for f in mag_kn:
            total_mag[f] = (
                -5.0
                / 2.0
                * logsumexp(
                    [-2.0 / 5.0 * np.log(10) * mag_grb[f], -2.0 / 5.0 * np.log(10) * mag_kn[f]],
                    axis=0,
                ) / np.log(10))
            dl = dlsource  # in [Mpc]
            total_mag_app[f] = total_mag[f] + 5. * np.log10(dl * (10**6)) - 5

        # Separating per bands
        for idx, bandname in enumerate(bands_work):
            shift_factor = 10**shift_factors[idx]

            if bandname == 'U':
                magABband = magAB_data[(band == bandname) | (band == 'white')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'white')]
                timeband = dtime[(band == bandname) | (band == 'white')]
                magABbandmodel = total_mag['desu']

            elif bandname == 'R':
                magABband = magAB_data[(band == bandname) | (band == 'r') | (band == 'F070W')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'r') | (band == 'F070W')]
                timeband = dtime[(band == bandname) | (band == 'r') | (band == 'F070W')]
                magABbandmodel = total_mag['desr']

            elif bandname == 'I':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['desi']

            elif bandname == 'Z':
                magABband = magAB_data[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]
                timeband = dtime[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]
                magABbandmodel = total_mag['desz']

            elif bandname == 'J':
                magABband = magAB_data[(band == bandname) | (band == 'F115W') | (band == 'F150W')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'F115W') | (band == 'F150W')]
                timeband = dtime[(band == bandname) | (band == 'F115W') | (band == 'F150W')]
                magABbandmodel = total_mag['2massj']

            elif bandname == 'H':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['2massh']

            elif bandname == 'K':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['2massks']

            elif bandname == 'F277W':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['f277w']

            elif bandname == 'F356W':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['f356w']

            elif bandname == 'F444W':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['f444w']

            # Plotting with shifting
            plt.plot(sample_times, magABbandmodel * shift_factor, color=colores[idx], alpha=0.3)
            if i == 0:
                plt.errorbar(timeband, magABband * shift_factor, yerr=magABerrband * shift_factor, 
                             fmt=fmts[idx], color=colores[idx], ecolor='black', markeredgecolor='black', label=f'{bandname}',ms=8 )
            else:
                plt.errorbar(timeband, magABband * shift_factor, yerr=magABerrband * shift_factor, 
                             fmt=fmts[idx], color=colores[idx], ecolor='black', markeredgecolor='black',ms=8 )

    plt.xscale('log')
    plt.yscale('symlog', linthresh=1e-1)  # Use symlog for y-axis
    plt.ylim(-1, -5e10)
    plt.ylabel('Absolute Magnitude')
    plt.xlabel('Time [days]')
    plt.legend()
    plt.show()
    
def ABmag_to_flux(magAB_data):
    # Calculate the fluxdensity from magAB_data
    exponent = -0.4 * (magAB_data + 48.6 + 5 * np.log10(dlsource * 1e6) - 5)
    fluxdensity = 1e23 * 10**exponent    
    return fluxdensity
    
def ABmagerr_to_fluxerr(magAB_data_err):
    # Calculate the fluxdensity from magAB_data
    exponent = -0.4 * (magAB_data_err + 48.6)
    fluxdensity_err = 1e23 * 10**exponent    
    return fluxdensity_err

# def ABmag_to_fluxjy(magAB_data):
#     # Calculate the fluxdensity from magAB_data
    
#     return fluxdensity




def plot_fit_flux(posterior, n_sample, posterior_samples, paper=True):
    import matplotlib.patheffects as pe
    samples = posterior.sample((n_sample,), x=x_o)
    samples = np.array(samples)
    
    cmap = plt.cm.get_cmap('nipy_spectral')
    colores = cmap(np.linspace(0.25, 0.85, 11))
    plt.figure(dpi=120, figsize=(20, 12))

    shift_factors = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10]  # Factors of 10 for each band
    fmts = ['o', 's', '^', 'D', 'X', 'v', 'P', 'H', '>', '<']
    bands_work = ['U', 'R', 'I', 'Z', 'J','H','K', 'F277W', 'F356W', 'F444W']

    percentile_50 = []
    percentile_up = []
    percentile_down = []
    for m in range(17):
        percentile_50.append(np.percentile(posterior_samples[:,int(m)],50))
        percentile_up.append(np.percentile(posterior_samples[:,int(m)],84))
        percentile_down.append(np.percentile(posterior_samples[:,int(m)],16))
    percentile_50 = np.array(percentile_50)
    

    for i in range(len(samples)):
        logE0, logn0, logthetac, logepse, logepsb, p, logxiN, logmej1, logvej1, beta1, logkappa_r1, Ye1, logmej2, logvej2, beta2, logkappa_r2, Ye2 = samples[i]
        
        # Define the luminosity distance of the source
        dlsource = 291  # in [Mpc]

        # The KN magnitude lightcurve is:
        _, mag_kn = Me2017x2_model_ejecta(10**logmej1, 10**logvej1, beta1, 10**logkappa_r1, Ye1, 10**logmej2, 10**logvej2, beta2, 10**logkappa_r2, Ye2, dlsource)

        # The GRB magnitude lightcurve is:
        mag_grb = GRB_model_lc(logE0, logn0, logthetac, logepse, logepsb, p, logxiN, dlsource)

        # Combine the two!
        total_mag = {}
        total_mag_app = {}
        for f in mag_kn:
            total_mag[f] = (
                -5.0
                / 2.0
                * logsumexp(
                    [-2.0 / 5.0 * np.log(10) * mag_grb[f], -2.0 / 5.0 * np.log(10) * mag_kn[f]],
                    axis=0,
                ) / np.log(10))
            dl = dlsource  # in [Mpc]
            total_mag_app[f] = total_mag[f] + 5. * np.log10(dl * (10**6)) - 5
        
        #True:
        logE0_true, logn0_true, logthetac_true, logepse_true, logepsb_true, p_true, logxiN_true, logmej1_true, logvej1_true, beta1_true, logkappa_r1_true, Ye1_true, logmej2_true, logvej2_true, beta2_true, logkappa_r2_true, Ye2_true =percentile_50
        # The KN magnitude lightcurve is:
        _, mag_kntrue = Me2017x2_model_ejecta(10**logmej1_true, 10**logvej1_true, beta1_true, 10**logkappa_r1_true, Ye1_true, 10**logmej2_true, 10**logvej2_true, beta2_true, 10**logkappa_r2_true, Ye2_true, dlsource)

        # The GRB magnitude lightcurve is:
        mag_grbtrue = GRB_model_lc(logE0_true, logn0_true, logthetac_true, logepse_true, logepsb_true, p_true, logxiN_true, dlsource)

        # Combine the two!
        total_magtrue = {}
        total_mag_apptrue = {}
        for f in mag_kntrue:
            total_magtrue[f] = (
                -5.0
                / 2.0
                * logsumexp(
                    [-2.0 / 5.0 * np.log(10) * mag_grbtrue[f], -2.0 / 5.0 * np.log(10) * mag_kntrue[f]],
                    axis=0,
                ) / np.log(10))
            dl = dlsource  # in [Mpc]
            total_mag_apptrue[f] = total_magtrue[f] + 5. * np.log10(dl * (10**6)) - 5
        
        #-----------------------------------------------------------------------------------------------------------------------
        #paper values:
        if paper==True:
            # Choose the "true" parameters.
            p_logE0_true = 53.87
            p_logn0_true = -4.11
            p_logthetac_true = -1.38
            p_logepse_true = -2.9
            p_logepsb_true = -3.96
            p_p_true = 2.64
            p_logxiN_true = -3.19 
            p_logmej1_true = -1.56
            p_logvej1_true = -0.72
            p_beta1_true = 3.09
            p_logkappa_r1_true = -0.24 
            p_Ye1_true = 0.3
            p_logmej2_true = -1.28 
            p_logvej2_true = -1.51
            p_beta2_true = 2.25
            p_logkappa_r2_true = 1.66 
            p_Ye2_true = 0.15

            # The KN magnitude lightcurve is:
            _, p_mag_kntrue = Me2017x2_model_ejecta(10**p_logmej1_true, 10**p_logvej1_true, p_beta1_true, 10**p_logkappa_r1_true, p_Ye1_true, 10**p_logmej2_true, 10**p_logvej2_true, p_beta2_true, 10**p_logkappa_r2_true, p_Ye2_true, dlsource)

            # The GRB magnitude lightcurve is:
            p_mag_grbtrue = GRB_model_lc(p_logE0_true, p_logn0_true, p_logthetac_true, p_logepse_true, p_logepsb_true, p_p_true, p_logxiN_true, dlsource)

            # Combine the two!
            p_total_magtrue = {}
            p_total_mag_apptrue = {}
            for f in p_mag_kntrue:
                p_total_magtrue[f] = (
                    -5.0
                    / 2.0
                    * logsumexp(
                        [-2.0 / 5.0 * np.log(10) * p_mag_grbtrue[f], -2.0 / 5.0 * np.log(10) * p_mag_kntrue[f]],
                        axis=0,
                    ) / np.log(10))
                dl = dlsource  # in [Mpc]
                p_total_mag_apptrue[f] = p_total_magtrue[f] + 5. * np.log10(dl * (10**6)) - 5

        
        # Separating per bands
        for idx, bandname in enumerate(bands_work):
            
            
            
            shift_factor0 = 10**(shift_factors[idx])

            
            
            if bandname == 'U':
                magABband = magAB_data[(band == bandname) | (band == 'white')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'white')]
                timeband = dtime[(band == bandname) | (band == 'white')]
                magABbandmodel = total_mag['desu']
                magABbandmodeltrue = total_magtrue['desu']
                magABbandmodeltrue0 = p_total_magtrue['desu']
                err_flux = fluxdensityerr[(band == bandname) | (band == 'white')]

            elif bandname == 'R':
                magABband = magAB_data[(band == bandname) | (band == 'r') | (band == 'F070W')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'r') | (band == 'F070W')]
                timeband = dtime[(band == bandname) | (band == 'r') | (band == 'F070W')]
                magABbandmodel = total_mag['desr']
                magABbandmodeltrue0 = p_total_magtrue['desr']
                magABbandmodeltrue = total_magtrue['desr']
                
                
                err_flux = fluxdensityerr[(band == bandname) | (band == 'r') | (band == 'F070W')]

            elif bandname == 'I':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['desi']
                magABbandmodeltrue = total_magtrue['desi']
                magABbandmodeltrue0 = p_total_magtrue['desi']
                err_flux = fluxdensityerr[(band==bandname)]

            elif bandname == 'Z':
                magABband = magAB_data[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]
                timeband = dtime[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]
                magABbandmodel = total_mag['desz']
                magABbandmodeltrue = total_magtrue['desz']
                magABbandmodeltrue0 = p_total_magtrue['desz']
                err_flux = fluxdensityerr[(band == bandname) | (band == 'z') | (band == 'Y') | (band == 'F105W')]

            elif bandname == 'J':
                magABband = magAB_data[(band == bandname) | (band == 'F115W') | (band == 'F150W')]
                magABerrband = magAB_data_err[(band == bandname) | (band == 'F115W') | (band == 'F150W')]
                timeband = dtime[(band == bandname) | (band == 'F115W') | (band == 'F150W')]
                magABbandmodel = total_mag['2massj']
                magABbandmodeltrue = total_magtrue['2massj']
                magABbandmodeltrue0 = p_total_magtrue['2massj']
                err_flux = fluxdensityerr[(band == bandname) | (band == 'F115W') | (band == 'F150W')]

            elif bandname == 'H':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['2massh']
                magABbandmodeltrue = total_magtrue['2massh']
                magABbandmodeltrue0 = p_total_magtrue['2massh']
                err_flux = fluxdensityerr[(band==bandname)]

            elif bandname == 'K':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['2massks']
                magABbandmodeltrue = total_magtrue['2massks']
                magABbandmodeltrue0 = p_total_magtrue['2massks']
                err_flux = fluxdensityerr[(band==bandname)]

            elif bandname == 'F277W':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['f277w']
                magABbandmodeltrue = total_magtrue['f277w']
                magABbandmodeltrue0 = p_total_magtrue['f277w']
                err_flux = fluxdensityerr[(band==bandname)]

            elif bandname == 'F356W':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                magABbandmodel = total_mag['f356w']
                magABbandmodeltrue = total_magtrue['f356w']
                magABbandmodeltrue0 = p_total_magtrue['f356w']
                err_flux = fluxdensityerr[(band==bandname)]


            elif bandname == 'F444W':
                magABband = magAB_data[(band == bandname)]
                magABerrband = magAB_data_err[(band == bandname)]
                timeband = dtime[(band == bandname)]
                err_flux = fluxdensityerr[(band==bandname)]
                magABbandmodel = total_mag['f444w']
                magABbandmodeltrue = total_magtrue['f444w']
                magABbandmodeltrue0 = p_total_magtrue['f444w']
                
                
                
            # Plotting with shifting
            plt.plot(sample_times, ABmag_to_flux(magABbandmodel) * shift_factor0, color=colores[idx], alpha=0.07)
            plt.plot(sample_times, ABmag_to_flux(magABbandmodeltrue) * shift_factor0, color=colores[idx], alpha=1.0, ls='--', path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])
            plt.plot(sample_times, ABmag_to_flux(magABbandmodeltrue0) * shift_factor0, color='red', alpha=1.0, ls='--')
            if i == 0:
                plt.errorbar(timeband, ABmag_to_flux(magABband) * shift_factor0, yerr=err_flux * shift_factor0, 
                             fmt=fmts[idx], color=colores[idx], ecolor='black', markeredgecolor='black', label=f'{bandname}',ms=10 )
            else:
                plt.errorbar(timeband, ABmag_to_flux(magABband) * shift_factor0, yerr=err_flux * shift_factor0, 
                             fmt=fmts[idx], color=colores[idx], ecolor='black', markeredgecolor='black',ms=10 )
                
            #  # Plotting with shifting
            # plt.plot(sample_times, magABbandmodel * shift_factor, color=colores[idx], alpha=0.5)
            # plt.plot(sample_times, magABbandmodeltrue * shift_factor, color='black', alpha=1.0, ls='--')
            # plt.plot(sample_times, magABbandmodeltrue0 * shift_factor, color='red', alpha=1.0, ls='--')
            # if i == 0:
            #     plt.errorbar(timeband, magABband * shift_factor, yerr=magABerrband * shift_factor, 
            #                  fmt=fmts[idx], color=colores[idx], ecolor='black', markeredgecolor='black', label=f'{bandname}',ms=8 )
            # else:
            #     plt.errorbar(timeband, magABband * shift_factor, yerr=magABerrband * shift_factor, 
            #                  fmt=fmts[idx], color=colores[idx], ecolor='black', markeredgecolor='black',ms=8 )   
                
                

    plt.xscale('log')
    plt.yscale('log')
    # plt.yscale('symlog', linthresh=1e-1)  # Use symlog for y-axis
    # plt.ylim(-1, -5e10)
    plt.ylim(10**-8, 4*10**7)
    #plt.ylim(10**-2, 10**45)
    plt.xlim(0.5,1.1e2)
    # plt.ylabel('Flux density (Jy)')
    plt.xlabel('Time [days]')
    plt.legend()
    plt.show()
    
    
    
    
    

# define the luminosity distance of the source
dlsource = 291 # in [Mpc]

# let's read the lightcurve data!
data = pd.read_csv('/tf/astrodados/phelipedata/NEURIPS2025/41586_2023_6979_MOESM4_ESM.csv')
dtime = np.array(data['delta_T']) # in [day]
band = np.array(data['Filter'])
fluxdensity = np.array(data['flux_density']) # in [Jy]
errplus=np.array(data['flux_density_err+']) # in [Jy]
errminus=np.array(data['flux_density_err-']) # in [Jy]
nu = np.array(data['nu']) # in [Hz]

#-----------------------------------------------------------
dtime=dtime[~np.isnan(errplus)]
band=band[~np.isnan(errplus)]
fluxdensity=fluxdensity[~np.isnan(errplus)]
nu=nu[~np.isnan(errplus)]
errminus=errminus[~np.isnan(errplus)]
errplus=errplus[~np.isnan(errplus)]
fluxdensityerr = .5*(errplus+errminus) #mean of the error
#-----------------------------------------------------------
# now let's converted density flux to AB magnitude (see https://en.wikipedia.org/wiki/AB_magnitude),  
# see that the second factor is just converting apparent magnitude into absolute magnitude
magAB_data = (-48.6 + -1 * np.log10(fluxdensity / 1e23) * 2.5) - 5.*np.log10(dlsource*(10**6))+5
magAB_data_err = (1.08574/fluxdensity)*fluxdensityerr # this factor is just to converted the fluxdensity error on the absolute/apparent magnitude
magApparent_data = (-48.6 + -1 * np.log10(fluxdensity / 1e23) * 2.5)
#-----------------------------------------------------------
#-----------------------------------------------------------
# create the time evolution
tmin, tmax, dt = 0.1, 70.0, 0.05

sample_times = np.arange(tmin, tmax + dt, dt)

colors = {'desu':'gold', '2massks':'blue', 'desr':'red', 'desi':'green', 'desz':'gray', '2massj':'purple', 
          '2massh':'brown', 'f277w':'orange', 'f356w':'pink', 'f444w':'violet'}
parameter_names = [
    "logE0", "logn0", "logthetac", "logepse", "logepsb", "p", "logxiN", 
    "logmej1", "logvej1", "beta1", "logkappa_r1", "Ye1", "logmej2", 
    "logvej2", "beta2", "logkappa_r2", "Ye2"
]

# create the KN lightcurve
kn_model = SimpleKilonovaLightCurveModel(sample_times=sample_times, filters=colors.keys())

def Me2017x2_model_ejecta(mej_1,vej_1,beta_1,kappa_r_1,Ye_1,mej_2,vej_2,beta_2,kappa_r_2,Ye_2, dl):

    bestfit_params1 = {
    "luminosity_distance": dl,
    "beta": beta_1,
    "log10_kappa_r": np.log10(kappa_r_1),
    #"timeshift": 0.183516607107672,
    "log10_vej": np.log10(vej_1),
    "log10_mej": np.log10(mej_1),
    "Ye": Ye_1,
    }
    bestfit_params2 = {
    "luminosity_distance": dl,
    "beta": beta_2,
    "log10_kappa_r": np.log10(kappa_r_2),
    #"timeshift": 0.183516607107672,
    "log10_vej": np.log10(vej_2),
    "log10_mej": np.log10(mej_2),
    "Ye": Ye_2,
    }
    lbol_1, mag_1 = kn_model.generate_lightcurve(sample_times, bestfit_params1)
    lbol_2, mag_2 = kn_model.generate_lightcurve(sample_times, bestfit_params2)
    
    lbol = lbol_1 + lbol_2
    mag = {}
    #merging the two magnitudes
    for k in mag_1.keys():
        mag[k] = -2.5*np.log10(10**(-mag_1[k]*0.4) + 10**(-mag_2[k]*0.4))
    #mag is the absolute magnitude (-3 - -30) and not the aparent(18-24) 
    return lbol, mag
def GRB_model_lc(logE0, logn0, logthetac, logepse, logepsb, p, logxiN, dlsource):
    
    colors = {'desu':'gold', '2massks':'blue', 'desr':'red', 'desi':'green', 'desz':'gray', '2massj':'purple', 
          '2massh':'brown', 'f277w':'orange', 'f356w':'pink', 'f444w':'violet'}
    
    
    # now lets create the GRB lightcurve
    param={"luminosity_distance": dlsource,
            "Ebv": 0.0,
            'jetType': 0,
            "inclination_EM": 0,
            "log10_E0": logE0,
            "thetaCore": .5*10**(logthetac),
            "thetaWing": 4*.5*10**(logthetac),
            "log10_n0": logn0,
            'p': p,    # electron energy distribution index
            "log10_epsilon_e": logepse,    # epsilon_e
            "log10_epsilon_B": logepsb,   # epsilon_B
            'xi_N': 10**(logxiN)}
    
    grb_model = GRBLightCurveModel(sample_times=sample_times, filters=colors.keys())
    _, mag_grb = grb_model.generate_lightcurve(sample_times, param)
    
    # absolute magnitude and not the apparent.
    return mag_grb

def simulator_lc_SBI_cuda(theta):
   
    theta = np.array(theta.cpu())
    logE0, logn0, logthetac, logepse, logepsb, p, logxiN, logmej1, logvej1, beta1, logkappa_r1, Ye1, logmej2, logvej2, beta2, logkappa_r2, Ye2 = theta
    
    dlsource = 291 # in [Mpc]
    
    # Calculate KN and GRB magnitude lightcurves
    _, mag_kn = Me2017x2_model_ejecta(10**logmej1, 10**logvej1, beta1, 10**logkappa_r1, Ye1, 10**logmej2, 10**logvej2, beta2, 10**logkappa_r2, Ye2, dlsource)
    mag_grb = GRB_model_lc(logE0, logn0, logthetac, logepse, logepsb, p, logxiN, dlsource)
    
    # Combine the two lightcurves
    total_mag = {
        f: -5.0 / 2.0 * logsumexp(
            [-2.0 / 5.0 * np.log(10) * mag_grb[f], -2.0 / 5.0 * np.log(10) * mag_kn[f]],
            axis=0
        ) / np.log(10)
        for f in mag_kn
    }

    dl_log10_factor = 5.0 * np.log10(dlsource * 1e6) - 5
    total_mag_app = {f: total_mag[f] + dl_log10_factor for f in total_mag}
    
    bands_work = ['U', 'R', 'I', 'Z', 'H', 'K', 'J', 'F277W', 'F356W', 'F444W']
    
    band_dict = {
        'U': ['desu', ['U', 'white']],
        'R': ['desr', ['R', 'r', 'F070W']],
        'I': ['desi', ['I']],
        'Z': ['desz', ['Z', 'z', 'Y', 'F105W']],
        'J': ['2massj', ['J', 'F115W', 'F150W']],
        'H': ['2massh', ['H']],
        'K': ['2massks', ['K']],
        'F277W': ['f277w', ['F277W']],
        'F356W': ['f356w', ['F356W']],
        'F444W': ['f444w', ['F444W']]
    }

    mag_modelo = []
    tempo_real = []
    
    for bandname, (total_mag_key, relevant_bands) in band_dict.items():
        mask = np.isin(band, relevant_bands)
        timeband = dtime[mask]
        interp_func = interp1d(sample_times, total_mag[total_mag_key], bounds_error=False, fill_value='extrapolate')
        magABbandmodel = interp_func(timeband)
        mag_modelo.extend(magABbandmodel)
        tempo_real.extend(timeband)
    
    mag_modelo = np.array(mag_modelo)
    tempo_real = np.array(tempo_real)
    sorted_indices = np.argsort(tempo_real)
    mag_modelo = ABmag_to_flux(mag_modelo[sorted_indices])
    tempo_real = tempo_real[sorted_indices]
    mag_modelo = torch.tensor(mag_modelo)
    mag_modelo = mag_modelo.unsqueeze(0)
    return mag_modelo

def simulator_lc(theta):
    
    logE0, logn0, logthetac, logepse, logepsb, p, logxiN, logmej1, logvej1, beta1, logkappa_r1, Ye1, logmej2, logvej2, beta2, logkappa_r2, Ye2 = theta
    
    # define the luminosity distance of the source
    dlsource = 291 # in [Mpc]
    
    # the KN magnitude lightcurve is:
    _, mag_kn = Me2017x2_model_ejecta(10**logmej1,10**logvej1,beta1,10**logkappa_r1,Ye1,10**logmej2,10**logvej2,beta2,10**logkappa_r2,Ye2, dlsource)
    
    #The GRB magnitude lightcurve is:
    mag_grb = GRB_model_lc(logE0, logn0, logthetac, logepse, logepsb, p, logxiN, dlsource)
    
    # let's combine the two!
    total_mag={}
    total_mag_app={}
    for f in mag_kn:
        total_mag[f] = (
            -5.0
            / 2.0
            * logsumexp(
                [-2.0 / 5.0 * np.log(10) * mag_grb[f], -2.0 / 5.0 * np.log(10) * mag_kn[f]],
                axis=0,
            )/ np.log(10))
        dl=dlsource #in [Mpc]
        total_mag_app[f] = total_mag[f]+5.*np.log10(dl*(10**6))-5
     
    bands_work = ['U', 'R', 'I', 'Z', 'H', 'K', 'J', 'F277W', 'F356W', 'F444W']
    
    
    
    mag_modelo = []
    mag_err_real=[]
    mag_real=[]
    tempo_real=[]
    
    printa = True
    #Separating per bands
    for bandname in bands_work:
        if bandname=='U':
            magABband = magAB_data[(band==bandname)|(band=='white')]
            magABerrband = magAB_data_err[(band==bandname)|(band=='white')]
            timeband = dtime[(band==bandname)|(band=='white')]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['desu'])(j) for j in timeband])

            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(mag_real)
                print(timeband)
                print(f'Mag real len: {len(mag_real)} Mag modelo {len(magABbandmodel)}')

            sigma2 = magABerrband**2
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)  

        if bandname=='R':
            magABband = magAB_data[(band==bandname)|(band=='r')|(band=='F070W')]
            magABerrband = magAB_data_err[(band==bandname)|(band=='r')|(band=='F070W')]
            timeband = dtime[(band==bandname)|(band=='r')|(band=='F070W')]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['desr'])(j) for j in timeband])

            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            sigma2 = magABerrband**2
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='I':
            magABband = magAB_data[(band==bandname)]
            magABerrband = magAB_data_err[(band==bandname)]
            timeband = dtime[(band==bandname)]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['desi'])(j) for j in timeband])
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            sigma2 = magABerrband**2
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')

        if bandname=='Z':
            magABband = magAB_data[(band==bandname)|(band=='z')|(band=='Y')|(band=='F105W')]
            magABerrband = magAB_data_err[(band==bandname)|(band=='z')|(band=='Y')|(band=='F105W')]
            timeband = dtime[(band==bandname)|(band=='z')|(band=='Y')|(band=='F105W')]

            # Calculate flux in a single X-ray band and time

            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['desz'])(j) for j in timeband])
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)        
            sigma2 = magABerrband**2
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='J':
            magABband = magAB_data[(band==bandname)|(band=='F115W')|(band=='F150W')]
            magABerrband = magAB_data_err[(band==bandname)|(band=='F115W')|(band=='F150W')]
            timeband = dtime[(band==bandname)|(band=='F115W')|(band=='F150W')] 

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['2massj'])(j+ np.random.random_sample()*1e-4) for j in timeband])
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            sigma2 = magABerrband**2
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='H':
            magABband = magAB_data[(band==bandname)]
            magABerrband = magAB_data_err[(band==bandname)]
            timeband = dtime[(band==bandname)]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['2massh'])(j) for j in timeband])
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)

            sigma2 = magABerrband**2
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='K':
            magABband = magAB_data[(band==bandname)]
            magABerrband = magAB_data_err[(band==bandname)]
            timeband = dtime[(band==bandname)]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['2massks'])(j) for j in timeband])
            #---------------------------------
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            #---------------------------------
            sigma2 = magABerrband**2
            
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='F277W':
            magABband = magAB_data[(band==bandname)]
            magABerrband = magAB_data_err[(band==bandname)]
            timeband = dtime[(band==bandname)]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['f277w'])(j) for j in timeband])

            #---------------------------------
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            #---------------------------------
            sigma2 = magABerrband**2
            
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='F356W':
            magABband = magAB_data[(band==bandname)]
            magABerrband = magAB_data_err[(band==bandname)]
            timeband = dtime[(band==bandname)]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['f356w'])(j) for j in timeband])

            #---------------------------------
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            #---------------------------------

            sigma2 = magABerrband**2
            
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            
            
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

        if bandname=='F444W':
            magABband = magAB_data[(band==bandname)]
            magABerrband = magAB_data_err[(band==bandname)]
            timeband = dtime[(band==bandname)]

            # Calculate flux in a single X-ray band and time
            magABbandmodel = np.array([interpolate.interp1d(sample_times, total_mag['f444w'])(j) for j in timeband])

            #---------------------------------
            mag_modelo.extend(magABbandmodel)
            mag_real.extend(magABband)
            mag_err_real.extend(magABerrband)
            tempo_real.extend(timeband)
            #---------------------------------

            sigma2 = magABerrband**2
            
            if printa==True:
                
                print(f'----------{bandname}----------')
                print(magABbandmodel)
                print(magABband)
                print(timeband)
                print(f'Mag real len: {len(magABband)} Mag modelo {len(magABbandmodel)}')
            
            #return -0.5 * np.sum((magABband - magABbandmodel) ** 2 / sigma2)

    return np.array(mag_modelo),np.array(mag_real), np.array(mag_err_real), np.array(tempo_real)


logE0_true = 53.87
logn0_true = -4.11
logthetac_true = -1.38
logepse_true = -2.9
logepsb_true = -3.96
p_true = 2.64
logxiN_true = -3.19 
logmej1_true = -1.56
logvej1_true = -0.72
beta1_true = 3.09
logkappa_r1_true = -0.24 
Ye1_true = 0.3
logmej2_true = -1.28 
logvej2_true = -1.51
beta2_true = 2.25
logkappa_r2_true = 1.66 
Ye2_true = 0.15


theta_true = np.array([logE0_true, logn0_true, logthetac_true, logepse_true, logepsb_true, p_true, logxiN_true, 
                    logmej1_true, logvej1_true, beta1_true, logkappa_r1_true, Ye1_true, 
                    logmej2_true, logvej2_true, beta2_true, logkappa_r2_true, Ye2_true]) 
mag_modelo0,mag_real0, mag_err_real0, tempo_real0 = simulator_lc(theta_true)

prior = utils_sbi.BoxUniform(low = torch.tensor([50, -6,-2 ,-6,-6,2.01,-5.0,     -3, -1, 1,-2,0.2,    -3, -2, 1,-0.5,0.1]), 
                         high = torch.tensor([60, 2,-0.5,-0.3,-0.3,2.9,0.0,  -1,-0.5,5, 0.5,0.4,  -1, -1, 5,  2, 0.2]))
from sbi.utils import RestrictedPrior, get_density_thresholder

class Model_NN(nn.Module):
    def __init__(self, outfeat):
        super().__init__()
        
        # Branch 1 (for input with shape (22,1))
        self.branch1_fc1 = nn.Linear(in_features=22, out_features=22*10)
        self.branch1_act1 = nn.ReLU()
        
        self.branch1_fc2 = nn.Linear(in_features=22*10, out_features=22*20)
        self.branch1_act2 = nn.ReLU()
        
        self.branch1_fc3 = nn.Linear(in_features=22*20, out_features=22*30)
        self.branch1_act3 = nn.ReLU()
        
        self.branch1_fc4 = nn.Linear(in_features=22*30, out_features=256)        
        self.branch1_act4 = nn.ReLU()
        
        
        
        # Branch 2 (for input with shape (11,1))
        self.branch2_fc1 = nn.Linear(in_features=11, out_features=11*10)
        self.branch2_act1 = nn.ReLU()
        
        self.branch2_fc2 = nn.Linear(in_features=11*10, out_features=11*20)
        self.branch2_act2 = nn.ReLU()
        
        self.branch2_fc3 = nn.Linear(in_features=11*20, out_features=11*30)
        self.branch2_act3 = nn.ReLU()
        
        self.branch2_fc4 = nn.Linear(in_features=11*30, out_features=256)        
        self.branch2_act4 = nn.ReLU()
        
        # Two outputs
        self.flat2 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256*2, out_features=256)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=outfeat)
        self.act5 = nn.ReLU()
        
    def forward(self, xi):
        # Splitting the input:
        input1 = xi[:, :, :22]  # early emission
        input2 = xi[:, :, 22:]  # late emission
        
        # Branch 1
        x1 = self.branch1_fc1(input1)
        x1 = self.branch1_act1(x1)
        x1 = self.branch1_fc2(x1)
        x1 = self.branch1_act2(x1)
        x1 = self.branch1_fc3(x1)
        x1 = self.branch1_act3(x1)
        x1 = self.branch1_fc4(x1)
        x1 = self.branch1_act4(x1)
        
        # Branch 2
        x2 = self.branch2_fc1(input2)
        x2 = self.branch2_act1(x2)
        x2 = self.branch2_fc2(x2)
        x2 = self.branch2_act2(x2)
        x2 = self.branch2_fc3(x2)
        x2 = self.branch2_act3(x2)
        x2 = self.branch2_fc4(x2)
        x2 = self.branch2_act4(x2)
        
        
        
        #gabriel addition
        # lstm_out, _ = self.lstm(x)  # LSTM output
        # # We take the output from the last time step
        # last_time_step_out = lstm_out[:, -1, :]
        
        # Concatenate the outputs from both branches along the sequence dimension
        x = torch.cat((x1, x2), dim=2)

        x = self.flat2(x)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act5(x)
        
        return x

















#TRAINING SECTION


from time import time
embed_net =  Model_NN(outfeat=128)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
density_estimator_build_fn = posterior_nn(model='mdn',embedding_net=embed_net,num_bins=15,num_transforms =5,hidden_features=30)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
inference = SNPE(prior=prior, density_estimator=density_estimator_build_fn,device='cpu')
inference_snle = SNLE(prior=prior, density_estimator=density_estimator_build_fn,device='cpu')
proposal = prior
t_i = time()
x_o = ABmag_to_flux(mag_real0)
rounds= 10
num_sim = 10000
for _ in range(rounds):
    print('Round:',_)
    simulator, a = prepare_for_sbi(simulator_lc_SBI_cuda, proposal)
    theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_sim, num_workers=80)
    #TRAINING THE SNPE:
    print('Training the SNPE')
    density_estimator = inference.append_simulations(torch.as_tensor(theta).to('cpu'), torch.as_tensor(x).to('cpu')).train(force_first_round_loss=True,
                                                                                                                            training_batch_size=512,
                                                                                                                            stop_after_epochs=600, 
                                                                                                                            max_num_epochs=10000)

    #TRAINING THE SNLE:
    print('Training the SNLE')
    _ = inference_snle.append_simulations(torch.as_tensor(theta).to('cpu'), torch.as_tensor(x).to('cpu')).train()
    ######################################################################################################
    print('Updating the Prion')
    posterior = inference.build_posterior().set_default_x((torch.as_tensor(x_o).to('cuda')))
    accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
    proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")

t_f = time() # stopping timer
delta = (t_f-t_i)
print(f'\nTotal Training time: {(delta)/60} minutes\n') 


posterior_samples = posterior.sample((2000,), x=x_o)
posterior_samples = np.array(posterior_samples)
corner_plot(posterior_samples, save=False) #600 epochs 10/08/24