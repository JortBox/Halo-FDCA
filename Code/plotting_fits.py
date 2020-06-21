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
from scipy.optimize import curve_fit
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from scipy import ndimage
from scipy import signal

import utils

Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)

titlesize = 20
labelsize = 13

#plt.style.use('classic')
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
                    origin='bottom',norm = NORM, extent=(obj.ra.value.min(),
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

def fit_result(obj, model, mask=False):
    fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)
    halo      = obj.halo
    ra        = halo.ra.value
    dec       = halo.dec.value
    for axi in axes.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(5))
        axi.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axi.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    fig.set_size_inches(3.2*5,5.1)

    vmin=-2*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
    vmax=4*(obj.params_units[0])

    NORM    = LogNorm(vmin=0.4*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                        vmax=20*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value)
    NORM = SymLogNorm(2.*obj.params_units[0] , linscale=1.0, vmin=vmin, vmax=vmax)
    NORMres = Normalize(vmin=-2.*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
                        vmax=1.*(u.Jy*obj.params[0]/halo.pix_area).to(uJyarcsec2).value)

    data = np.copy(obj.data)
    if mask:
        data.value[obj.image_mask==1]= -100.

    im1 = axes[0].imshow((data/halo.pix_area).to(uJyarcsec2).value,
                        cmap='inferno', origin='bottom',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        LEVEL = np.arange(1,7)*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
        cont1 = axes[0].contour((model/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont2 = axes[0].contour((data/halo.pix_area).to(uJyarcsec2).value,
                        colors='lightgreen', levels=np.array([-99.8]), alpha=0.6, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
        #pass
    except:
        print('PROCESSING: Failed making contours')
        pass

    axes[0].annotate('$V(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[0].set_title("Subtracted halo", fontsize=titlesize)
    axes[0].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[0].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[0].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[0])
    draw_ellipse(halo,axes[0])
    plt.tight_layout()

    im2 = axes[1].imshow((model/halo.pix_area).to(uJyarcsec2).value,
                        cmap='inferno', origin='bottom',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    axes[1].annotate('$I(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[1].set_title("Exponential model", fontsize=titlesize)
    axes[1].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    #draw_sizebar(halo,axes[1])
    #draw_ellipse(halo,axes[1])
    plt.tight_layout()

    im3 = axes[2].imshow(abs((halo.data/halo.pix_area).to(uJyarcsec2).value-\
                        (model/halo.pix_area).to(uJyarcsec2).value),
                        cmap='inferno', origin='bottom',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    cont4 = axes[2].contour((data/halo.pix_area).to(uJyarcsec2).value,
                        colors='lightgreen', levels=np.array([-99.8]), alpha=0.6, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    try:
        cont3 = axes[2].contour((model/halo.pix_area).to(uJyarcsec2).value, alpha=0.7,
                            colors='white', levels=[2*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value],
                            extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm=NORMres)
        axes[2].clabel(cont3, fontsize=12, inline=1, fmt='2$\\sigma_{\\mathrm{rms}}$')
    except: pass
    axes[2].annotate('$|V(x,y)-I(x,y)|$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')
    axes[2].set_title("Residual image", fontsize=titlesize)
    axes[2].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[2].grid(color='white', linestyle='-', alpha=0.25)
    #draw_sizebar(halo,axes[2])
    #draw_ellipse(halo,axes[2])
    plt.tight_layout()
    import matplotlib.ticker as ticker

    cbar = fig.colorbar(im3)
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$',fontsize=labelsize)
    #cbar.formatter = ScalarFormatter(useMathText=False)
    #cbar.formatter = ticker.LogFormatter(base=10.,labelOnlyBase=True)
    cbar.formatter = ticker.StrMethodFormatter('%.2f')

    plt.savefig(halo.plotPath +halo.file.replace('.fits','')+'_mcmc_model'+obj.filename_append+'.pdf')
    #plt.show()
    plt.clf()
    plt.close(fig)

def draw_sizebar(obj,ax):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    length = 1./obj.factor.to(u.Mpc/u.deg)
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    asb = AnchoredSizeBar(ax.transData,length.value,
                          r"1 Mpc",
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False, color='white')#, fontsize=labelsize)
    ax.add_artist(asb)

def draw_ellipse(obj,ax):
    """
    Draw an ellipse of width=0.1, height=0.15 in data coordinates
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
    try:
        ae = AnchoredEllipse(ax.transData, width=obj.bmaj.value, height=obj.bmin.value,
                                angle=obj.bpa.value,
                                loc='lower left', pad=0.3, borderpad=0.3,
                                frameon=True,color='lightskyblue')
    except:
        ae = AnchoredEllipse(ax.transData, width=obj.bmaj.value, height=obj.bmin.value,
                                angle=obj.bpa.value,
                                loc='lower left', pad=0.3, borderpad=0.3,frameon=True)
    ax.add_artist(ae)
