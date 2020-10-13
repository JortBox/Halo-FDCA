#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
'''

import numpy as np
import astropy.units as u
import sys
import scipy.stats as stats
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import os
#import aplpy
from scipy.optimize import curve_fit
import matplotlib.colors as mplc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from scipy import ndimage
from scipy import signal

from . import fdca_utils as utils

Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)

titlesize = 20
labelsize = 13

plt.style.use('classic')
plt.rc('text',usetex=True)
plt.rc('font', family='serif')

def quick_imshow(obj, data, noise=True, name = 'plot'):
    fig, ax = plt.subplots()
    if noise:
        NORM = Normalize(vmin=-2*(obj.rmsnoise/obj.pix_area).to(uJyarcsec2).value,
                        vmax=20*(obj.rmsnoise/obj.pix_area).to(uJyarcsec2).value)
    else:
        NORM = None

    im = ax.imshow((data/obj.pix_area).to(uJyarcsec2).value, cmap='gist_heat',
                    origin='lower',norm = NORM, extent=(obj.ra.value.min(),
                    obj.ra.value.max(), obj.dec.value.min(), obj.dec.value.max()) )
    draw_sizebar(obj,ax)
    draw_ellipse(obj,ax)

    # Create colorbar
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$')

    plt.xlabel('RA [deg]')
    plt.ylabel('DEC [deg]')

    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(name+'.pdf')
    plt.show()

def fit_result(obj, model, data, noise, mask=False, regrid=False):
    halo      = obj.halo
    ra        = halo.ra.value
    dec       = halo.dec.value
    bmin      = halo.bmin
    bmaj      = halo.bmaj
    scale     = 1.
    xlabel    = 'RA [Deg]'
    ylabel    = 'DEC [Deg]'
    scale = 1.

    #if mask:
    image_mask = obj.image_mask

    if regrid:
        data  = utils.regridding(obj.halo,data, decrease_fov=True)
        model = utils.regridding(obj.halo,model)
        #if mask:
        image_mask = utils.regridding(obj.halo, obj.image_mask*u.Jy, mask= not obj.halo.cropped).value
        noise      = utils.findrms(data.value)*u.Jy
        scale = (np.array((bmin.value,bmaj.value))/halo.pix_size).value
        bmin  = bmin/(scale[0]*halo.pix_size)
        bmaj  = bmaj/(scale[1]*halo.pix_size)
        ra         = np.arange(0,data.shape[1])#halo.ra.value
        dec        = np.arange(0,data.shape[0])#halo.dec.value
        xlabel    = 'Pixels'
        ylabel    = 'Pixels'

        #plt.imshow(image_mask)
        #plt.show()

    fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)

    for axi in axes.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(5))
        axi.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axi.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    fig.set_size_inches(3.2*5,5.1)

    draw_sizebar(halo,axes[0], scale, regrid)
    draw_ellipse(halo,axes[0], bmin, bmaj, regrid)

    data  = (data/halo.pix_area).to(uJyarcsec2).value
    noise = (noise/halo.pix_area).to(uJyarcsec2).value
    model = (model/halo.pix_area).to(uJyarcsec2).value


    masked_data = np.copy(data)
    #if mask:
    if regrid:
        masked_data[image_mask > obj.mask_treshold*image_mask.max()] =-100.
    else:
        masked_data[image_mask==1]= -100.

    if regrid:
        NORMres = mplc.Normalize(vmin=-2.*noise, vmax=1.*masked_data.max())
    else: NORMres = mplc.Normalize(vmin=-2.*noise, vmax=2.*masked_data.max())
    Normdiv = mplc.TwoSlopeNorm(vcenter=0., vmin=0.8*masked_data.min(), vmax=0.8*masked_data.max())

    im1 = axes[0].imshow(masked_data,cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    #try:
    LEVEL = np.arange(1,7)*noise
    cont1 = axes[0].contour(model,colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
    cont2 = axes[0].contour(masked_data,colors='lightgreen', levels=np.array([-99.8]),
                        alpha=0.6, linestyles='-',extent=(ra.max(),ra.min(),dec.min(),dec.max()),
                        norm = NORMres,linewidths=1.5)
        #pass
    #except:
    #    print('PROCESSING: Failed making contours')
    #    pass

    axes[0].annotate('$V(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[0].set_title("Radio data", fontsize=titlesize)
    axes[0].set_xlabel(xlabel, fontsize=labelsize)
    axes[0].set_ylabel(ylabel, fontsize=labelsize)
    axes[0].grid(color='white', linestyle='-', alpha=0.25)

    plt.tight_layout()

    im2 = axes[1].imshow(model,cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    axes[1].annotate('$I(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[1].set_title(obj.modelName.replace('_',' ')+" model", fontsize=titlesize)
    axes[1].set_xlabel(xlabel, fontsize=labelsize)
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    cbar = fig.colorbar(im2,ax=axes[1])
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$',fontsize=labelsize)
    #cbar.formatter = ScalarFormatter(useMathText=False)
    #cbar.formatter = ticker.LogFormatter(base=10.,labelOnlyBase=True)
    #cbar.formatter = ticker.StrMethodFormatter('%.2f')
    plt.tight_layout()


    im3 = axes[2].imshow(data-model, cmap='PuOr_r', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = Normdiv)
    cont4 = axes[2].contour(masked_data,
                        colors='lightgreen', levels=np.array([-99.8]), alpha=0.6, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    try:
        cont3 = axes[2].contour(model, alpha=0.7, colors='black', levels=[2*noise],
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm=NORMres)
        axes[2].clabel(cont3, fontsize=12, inline=1, fmt='2$\\sigma_{\\mathrm{rms}}$',colors='black')
    except: pass
    axes[2].annotate('$V(x,y)-I(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='black')
    axes[2].set_title("Residual image", fontsize=titlesize)
    axes[2].set_xlabel(xlabel, fontsize=labelsize)
    axes[2].grid(color='black', linestyle='-', alpha=0.25)
    plt.tight_layout()
    import matplotlib.ticker as ticker

    cbar = fig.colorbar(im3,ax=axes[2])
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$',fontsize=labelsize)
    #cbar.formatter = ScalarFormatter(useMathText=False)
    #cbar.formatter = ticker.LogFormatter(base=10.,labelOnlyBase=True)
    #cbar.formatter = ticker.StrMethodFormatter('%.2f')

    if regrid:
        plt.savefig(halo.plotPath +halo.file.replace('.fits','')+'_mcmc_model'+obj.filename_append+'_REGRID.pdf')
    else:
        plt.savefig(halo.plotPath +halo.file.replace('.fits','')+'_mcmc_model'+obj.filename_append+'.pdf')
    #plt.show()
    plt.clf()
    plt.close(fig)

def fit_result_noise(obj, model, mask=False):
    fig, axes = plt.subplots(ncols=3, nrows=1)
    halo = obj.halo
    ra  = halo.ra.value
    dec = halo.dec.value

    noise = utils.advanced_noise_modeling(obj)
    noise = (noise/halo.pix_area).to(uJyarcsec2)
    fit = (model/halo.pix_area).to(uJyarcsec2)
    dat = (halo.data/halo.pix_area).to(uJyarcsec2)

    #dat[model.value>2*halo.rmsnoise.value]=np.nan
    noise_conv_mask = np.copy(noise)
    #noise_conv_mask[model.value>2*halo.rmsnoise.value]=np.nan


    for axi in axes.flat:
        axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    fig.set_size_inches(3.1*5,5)

    NORM    = Normalize(vmin=-2*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                        vmax=20*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value)
    NORMres = Normalize(vmin=-2.*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                        vmax=20*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value)
    LEVEL = np.arange(2,7)*3*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value

    im1 = axes[0].imshow(noise.value,
                        cmap='gist_heat', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    axes[0].annotate('$\\kappa(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[0].set_title("Modelled noise", fontsize=titlesize)
    axes[0].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[0].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[0].grid(color='white', linestyle='-', alpha=0.25)
    #draw_sizebar(halo,axes[0])
    #draw_ellipse(halo,axes[0])
    plt.tight_layout()

    im2 = axes[1].imshow(noise.value+fit.value,
                            cmap='gist_heat', origin='lower',
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    cont1 = axes[1].contour((halo.data/halo.pix_area).to(uJyarcsec2).value,
                            colors='white', levels=LEVEL, alpha=0.7,
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    axes[1].annotate('$I(x,y)+\\kappa(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[1].set_title("Exponential model", fontsize=titlesize)
    axes[1].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[1])
    draw_ellipse(halo,axes[1])
    cbar = fig.colorbar(im2)
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$', fontsize=labelsize)
    plt.tight_layout()

    noiserange = (halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
    bins = np.linspace(-5*noiserange,5*noiserange, 80)
    hist1 = axes[2].hist(dat.value.flat, bins=bins, color='black', histtype='step',
                            hatch="//", lw=2, label='$V(x,y)$')
    hist2 = axes[2].hist(noise_conv_mask.value.flat, bins=bins, color='red',
                            histtype='step', hatch="\\\\", lw=2, label='$\\kappa(x,y)$')

    axes[2].set_title("Noise distribution", fontsize=titlesize)
    axes[2].set_xlabel('$\\mu$Jy arcsec$^{-2}$', fontsize=labelsize)
    axes[2].set_ylabel('Relative count', fontsize=labelsize)
    axes[2].legend(frameon=False, loc='upper left',fontsize=labelsize)
    plt.tight_layout()

    if mask:
        plt.savefig(halo.plotPath +halo.target+"_ADVnoise_visualisation_mask.pdf")
    else:
        plt.savefig(halo.plotPath +halo.target+"_ADVnoise_visualisation.pdf")
    #plt.show()
    plt.clf()
    plt.close(fig)

def noise_distribution(obj):
    halo = obj.halo
    nbin = 100
    bins = np.linspace(-5*halo.rmsnoise.value,
                            8*halo.rmsnoise.value, nbin)
    x    = np.linspace(-5*halo.rmsnoise.value,
                            8*halo.rmsnoise.value, 1000)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    hist_data, data_bins = np.histogram(halo.data.value.ravel(), bins=bins)
    imagenoise = utils.gauss(x, *halo.noise_char)
    noise      = utils.advanced_noise_modeling(obj)

    plt.hist(halo.data.value.flat, bins=bins, color='black', histtype='step', hatch="//", lw=2, label='$V(x,y)$')
    plt.hist(noise.value.flat, bins=bins, color='gray', histtype='step', hatch="\\\\", lw=2, label='$\\kappa(x,y)$')
    plt.plot(x,imagenoise, color='red', label='Noise model', lw=3, alpha=0.6)

    plt.xlabel('Jy/Pixel')
    plt.ylabel('Relative count')
    plt.title("Noise distribution of halo in "+halo.name)
    plt.legend(frameon=False, loc='upper right')
    plt.ylim(0,1.3*np.max(imagenoise))
    plt.xlim(-5*halo.rmsnoise.value,8*halo.rmsnoise.value)
    plt.tight_layout()
    plt.minorticks_on()
    plt.savefig(obj.halo.modelPath+obj.halo.target+"_noise_distribution.pdf")
    plt.show()
    #plt.clf()

    #print(np.shape(halo.data.value))

def fit_result_mcmc(obj, func):
    fig, axes = plt.subplots(ncols=3, nrows=1)
    halo = obj
    ra  = halo.ra.value
    dec = halo.dec.value
    for axi in axes.flat:
        axi.yaxis.set_major_locator(plt.MaxNLocator(3))
    fig.set_size_inches(3.4*5,5)



    fit = (func(obj, *halo.params).reshape(len(halo.x_pix), len(halo.y_pix)))*u.Jy

    NORM = Normalize(vmin=-2*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                     vmax=20*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value)
    LEVEL = np.arange(1,7)*3*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value

    im1 = axes[0].imshow((halo.data/halo.pix_area).to(uJyarcsec2).value,
                        cmap='gist_heat', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    cont1 = axes[0].contour((fit/halo.pix_area).to(uJyarcsec2).value,
                            colors='white', levels=LEVEL, alpha=0.7,
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    axes[0].set_title("Subtracted halo $V(x,y)$")
    axes[0].set_xlabel('RA [deg]')
    axes[0].set_ylabel('DEC [deg]')
    axes[0].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[0])
    draw_ellipse(halo,axes[0])
    plt.tight_layout()

    im2 = axes[1].imshow((fit/halo.pix_area).to(uJyarcsec2).value,
                            cmap='gist_heat', origin='lower',
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    axes[1].set_title("Exponential model $I(x,y)$")
    axes[1].set_xlabel('RA [deg]')
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    #draw_sizebar(halo,axes[1])
    #draw_ellipse(halo,axes[1])
    plt.tight_layout()

    im3 = axes[2].imshow((halo.data/halo.pix_area).to(uJyarcsec2).value-\
                        (fit/halo.pix_area).to(uJyarcsec2).value,
                        cmap='gist_heat', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    cont3 = axes[2].contour((fit/halo.pix_area).to(uJyarcsec2).value,
                            colors='white', levels=[2*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value],
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORM)
    axes[2].clabel(cont3, fontsize=12, inline=1, fmt='2$\\sigma_{\\mathrm{rms}}$')
    axes[2].set_title("Residual image")
    axes[2].set_xlabel('RA [deg]')
    axes[2].grid(color='white', linestyle='-', alpha=0.25)
    #draw_sizebar(halo,axes[2])
    #draw_ellipse(halo,axes[2])
    plt.tight_layout()

    cbar = fig.colorbar(im3)
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$')

    plt.savefig(halo.plotPath +halo.target+"_mcmc_model_visual.pdf")
    plt.show()
    #plt.clf()

def draw_sizebar(obj,ax, scale, regrid=False):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    if regrid:
        length = 0.1/obj.factor.to(u.Mpc/u.deg)/(scale[1]*obj.pix_size)
    else:
        length = 0.1/obj.factor.to(u.Mpc/u.deg)

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    asb = AnchoredSizeBar(ax.transData,length.value,
                          r"100 kpc",
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False, color='white')#, fontsize=labelsize)
    ax.add_artist(asb)

def draw_ellipse(obj,ax, bmin, bmaj, regrid=False):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
    """
    Draw an ellipse of width=0.1, height=0.15 in data coordinates
    """

    bpa = obj.bpa.value
    if regrid:
        bpa = 0
    try:
        ae = AnchoredEllipse(ax.transData, width=bmaj.value, height=bmin.value,
                                angle=bpa, loc='lower left', pad=0.3, borderpad=0.3,
                                frameon=True,color='lightskyblue')
    except:
        ae = AnchoredEllipse(ax.transData, width=bmaj.value, height=bmin.value,
                                angle=bpa, loc='lower left', pad=0.3, borderpad=0.3,
                                frameon=True)

    ax.add_artist(ae)

def power_mass_relation(power, mass, targets):
    mcxc = np.zeros(len(targets))
    for i in range(len(targets)):
        print(targets[i][:4])
        if targets[i][:4]=='MCXC':
            mcxc[i]=1
        else:
            mcxc[i]=0

    plt.errorbar(mass[:,0][mcxc==0], power[:,0][mcxc==0], yerr=power[:,1][mcxc==0],
                xerr=mass[:,1][mcxc==0], lw=1, markersize=3., capsize=4, mew=1. ,
                fmt='o', alpha=0.8, color='black', marker='.')
    plt.scatter(mass[:,0][mcxc==0],power[:,0][mcxc==0],s=20,color='black',  marker='.')

    plt.errorbar(mass[:,0][mcxc==1], power[:,0][mcxc==1], yerr=power[:,1][mcxc==1],
                lw=1, markersize=3., capsize=4, mew=1. ,
                fmt='o', alpha=0.8, color='deepskyblue', marker='.')
    plt.scatter(mass[:,0][mcxc==1],power[:,0][mcxc==1],s=20,color='deepskyblue',marker='.')
    plt.minorticks_on()
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$P_{1.4 \;\mathrm{GHz}}$ [Watts/Hz]')
    plt.xlabel('M$_{500}$ [M$_{\\odot}$]')
    plt.savefig('result.pdf')
    plt.clf()

def model_comparisson(halo, mask=False):
    fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)
    model4 = halo.result4.model
    model6 = halo.result6.model
    model8 = halo.result8.model
    ra        = halo.ra.value
    dec       = halo.dec.value
    for axi in axes.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(5))
        axi.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axi.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    fig.set_size_inches(3.2*5,5.1)

    vmin=-2*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
    vmax=4*(halo.result4.params_units[0])

    LEVEL = np.arange(1,7)*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
    NORM    = LogNorm(vmin=0.4*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                        vmax=20*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value)
    #NORM = SymLogNorm(2.*halo.result4.params_units[0] , linscale=1.0, vmin=vmin, vmax=vmax)

    data = np.copy(halo.result4.data)

    NORMres = Normalize(vmin=-2.*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                        vmax=1.*(data/halo.pix_area).to(uJyarcsec2).value.max())
    if mask:
        data.value[halo.result4.image_mask==1]= -100.

    im1 = axes[0].imshow((data/halo.pix_area).to(uJyarcsec2).value,
                        cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        cont1 = axes[0].contour((model4/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont2 = axes[0].contour((data/halo.pix_area).to(uJyarcsec2).value,
                        colors='lightgreen', levels=np.array([-99.8]), alpha=0.9, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    except:
        print('PROCESSING: Failed making contours')
        pass

    axes[0].set_title('Circular\n $S_{\\mathrm{144 MHz}}=%.0f\\pm%.0f$ mJy' % (halo.result4.flux_val.value, halo.result4.flux_err.value), fontsize=15)
    axes[0].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[0].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[0].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[0])
    draw_ellipse(halo,axes[0])
    plt.tight_layout()

    im2 = axes[1].imshow((data/halo.pix_area).to(uJyarcsec2).value,
                        cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        cont3 = axes[1].contour((model6/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont4 = axes[1].contour((data/halo.pix_area).to(uJyarcsec2).value,
                        colors='lightgreen', levels=np.array([-99.8]), alpha=0.9, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    except:
        print('PROCESSING: Failed making contours')
        pass

    axes[1].set_title('Elliptical\n $S_{\\mathrm{144 MHz}}=%.0f\\pm%.0f$ mJy' % (halo.result6.flux_val.value, halo.result8.flux_err.value), fontsize=15)
    axes[1].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[1].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[1])
    draw_ellipse(halo,axes[1])
    plt.tight_layout()

    im3 = axes[2].imshow((data/halo.pix_area).to(uJyarcsec2).value,
                        cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        cont5 = axes[2].contour((model8/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont6 = axes[2].contour((data/halo.pix_area).to(uJyarcsec2).value,
                        colors='lightgreen', levels=np.array([-99.8]), alpha=0.9, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    except:
        print('PROCESSING: Failed making contours')
        pass


    axes[2].set_title('Skewed \n $S_{\\mathrm{144 MHz}}=%.0f\\pm%.0f$ mJy' % (halo.result8.flux_val.value, halo.result8.flux_err.value), fontsize=15)
    axes[2].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[2].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[2].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[2])
    draw_ellipse(halo,axes[2])
    plt.tight_layout()


    import matplotlib.ticker as ticker

    cbar = fig.colorbar(im3)
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$',fontsize=labelsize)
    #cbar.formatter = ScalarFormatter(useMathText=False)
    #cbar.formatter = ticker.LogFormatter(base=10.,labelOnlyBase=True)
    cbar.formatter = ticker.StrMethodFormatter('%.2f')


    plt.savefig(halo.plotPath +halo.file.replace('.fits','')+'_mcmc_model_ALL.pdf')
    #plt.show()
    plt.clf()
    plt.close(fig)
