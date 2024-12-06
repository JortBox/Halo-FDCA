#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
'''

import numpy as np
import astropy.units as u
import corner
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib.ticker import ScalarFormatter

from . import fdca_utils as utils

Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)

titlesize = 20
labelsize = 13



def samplerplot(obj):
    try:
        labels = obj.labels[~obj.frozen[obj.params]]
    except:
        #labels = obj.labels[~obj.frzn[obj.prms]]
        labels = obj.dim * ['']
    
    fig, axes = plt.subplots(ncols=1, nrows=obj.dim, sharex=True)
    axes[0].set_title("Number of walkers: " + str(obj.walkers), fontsize=25)
    for axi in axes.flat:
        axi.yaxis.set_major_locator(plt.MaxNLocator(3))
        fig.set_size_inches(2 * 10, 15)

    for i in range(obj.dim):
        axes[i].plot(
            obj.sampler[:, :, i].transpose(), color="black", alpha=0.3, lw=0.5
        )
        
        axes[i].set_ylabel(labels[i], fontsize=20)
        axes[-1].set_xlabel("steps", fontsize=20)
        axes[i].axvline(obj.burntime, ls="dashed", color="red")
        axes[i].tick_params(labelsize=20)
        plt.xlim(0, obj.sampler.shape[1])

    if obj.save:
        try:
            plt.savefig(
                obj.halo.plotPath
                + obj.halo.file.replace(".fits", "")
                + "_walkers"
                + obj.filename_append
                + ".pdf"
            )
        except:
            obj.logger.error("Failed to save walkers plot")
        plt.clf()
        plt.close(fig)
    else:
        plt.show()
        
def cornerplot(obj):
    try:
        labels = obj.labels[~obj.frzn[obj.prms]]
        labels_uits = obj.labels_units[~obj.frzn[obj.prms]]    
    except:
        labels = obj.labels[~obj.frozen[obj.params]]
        labels_uits = obj.labels_units[~obj.frozen[obj.params]]
        
    #try:
    popt_units = utils.transform_units(obj, obj.popt)
    fig = corner.corner(
        obj.samples_units,
        labels=labels_uits,
        truths=popt_units[obj.params & ~obj.frozen],
        quantiles=[0.160, 0.5, 0.840],
        show_titles=True,
        max_n_ticks=3,
        title_fmt="1.2g",
    )
    '''   
    except AttributeError as e:  
        obj.logger.errer(f"Failed to create corner plot with units: {e}")         
        fig = corner.corner(
            obj.samples,
            labels=labels, 
            quantiles=[0.160, 0.5, 0.840],
            truths=np.asarray(obj.popt[obj.params.values.T & ~obj.frozen.values.T]).flatten(),
            show_titles=True,
            title_fmt=".5f",
        )'''
    if obj.save:
        try:
            plt.savefig(
                obj.halo.plotPath
                + obj.halo.file.replace(".fits", "")
                + "_cornerplot"
                + obj.filename_append
                + ".pdf"
            )
        except:
            obj.logger.error("Failed to save corner plot")
        plt.clf()
        plt.close(fig)
    else:
        plt.show()        
        
        


def fit_result(obj, model, data, noise, mask=False, regrid=False):
    halo   = obj.halo
    ra     = halo.ra.value
    dec    = halo.dec.value
    bmin   = halo.bmin
    bmaj   = halo.bmaj
    scale  = 1.
    xlabel = 'RA [Deg]'
    ylabel = 'DEC [Deg]'
    scale  = 1.

    #if mask:
    image_mask = obj.image_mask
    ncomp = len(model) if isinstance(model, list) else 1

    if regrid:
        data  = utils.regridding(obj.halo, data)
        if ncomp > 1:
            for i in range(ncomp):
                model[i] = utils.regridding(obj.halo, model[i]*u.Jy)
        else:
            model = utils.regridding(obj.halo, model)
            
        #if mask:
        image_mask = utils.regridding(obj.halo, obj.image_mask.astype(int)*u.Jy, mask= not obj.halo.cropped).value
        noise  = utils.findrms(data.value)*u.Jy
        scale  = (np.array((bmin.value,bmaj.value))/halo.pix_size).value
        bmin   = bmin/(scale[0]*halo.pix_size)
        bmaj   = bmaj/(scale[1]*halo.pix_size)
        ra     = np.arange(0,data.shape[1])#halo.ra.value
        dec    = np.arange(0,data.shape[0])#halo.dec.value
        xlabel = 'Pixels'
        ylabel = 'Pixels'


    fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(20, 7), layout='constrained')
    axes = axes.flatten()

    for axi in axes.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(5))
        axi.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axi.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    draw_sizebar(halo,axes[0], scale, regrid)
    draw_ellipse(halo,axes[0], bmin, bmaj, regrid)

    data  = (data/halo.pix_area).to(uJyarcsec2).value
    noise = (noise/halo.pix_area).to(uJyarcsec2).value
    if ncomp > 1:
        for i in range(ncomp):
            model[i] = (model[i]/halo.pix_area).to(uJyarcsec2).value
    else:
        model = (model/halo.pix_area).to(uJyarcsec2).value
        
    masked_data = np.copy(data)
    #if mask:
    if regrid:
        masked_data[image_mask < obj.mask_treshold*image_mask.max()] =-10000.
    else:
        masked_data[~image_mask]= -10000.


    if regrid:
        NORMres = mplc.Normalize(vmin=-2.*noise, vmax=1.*masked_data.max())
    else: NORMres = mplc.Normalize(vmin=-2.*noise, vmax=1.*masked_data.max())
    LEVEL = np.array([1,2,4,8,16,32,64,128,256,512,1024,2048,4096])*noise
    

    im1 = axes[0].imshow(masked_data,cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    if ncomp > 1:
        for i in range(ncomp):
            axes[0].contour(model[i],colors='white', levels=LEVEL, alpha=0.6,
                                extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
    else:
        axes[0].contour(model,colors='white', levels=LEVEL, alpha=0.6,
                                extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
    axes[0].contour(masked_data,colors='lightgreen', levels=np.array([-9999.8]),
                        alpha=0.6, linestyles='-',extent=(ra.max(),ra.min(),dec.min(),dec.max()),
                        norm = NORMres,linewidths=1.5)
    axes[0].annotate('$V(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')

    if ncomp > 1:
        model_sum = np.asarray(model).sum(axis=0)
    else:
        model_sum = model

    im2 = axes[1].imshow(model_sum,cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    axes[1].annotate('$I(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='white')


    #Trying two different functions since names were changed in recent matplotlib 3.3 update.
    try:
        Normdiv = mplc.TwoSlopeNorm(vcenter=0., vmin=0.8*(data-model_sum).min(), vmax=0.8*(data-model_sum).max())
    except:
        Normdiv = mplc.DivergingNorm(vcenter=0., vmin=0.8*(data-model_sum).min(), vmax=0.8*(data-model_sum).max())
    im3 = axes[2].imshow(data-model_sum, cmap='PuOr_r', origin='lower',extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = Normdiv)
    axes[2].contour(masked_data,
                        colors='red', levels=np.array([-9999.8]), alpha=0.6, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    try:
        if ncomp > 1:
            for i in range(ncomp):
                cont3 = axes[2].contour(model[i], alpha=0.7, colors='black', levels=[2*noise],extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm=NORMres)
                axes[2].clabel(cont3, fontsize=12, inline=1, fmt=f'comp {i}',colors='black')
        else:
            cont3 = axes[2].contour(model, alpha=0.7, colors='black', levels=[2*noise],extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm=NORMres)
            axes[2].clabel(cont3, fontsize=12, inline=1, fmt='2$\\sigma_{\\mathrm{rms}}$',colors='black')
    except: pass
    axes[2].annotate('$V(x,y)-I(x,y)$',xy=(0.5, 1), xycoords='axes fraction',
                        fontsize=titlesize, xytext=(0, -9), textcoords='offset points',
                        ha='center', va='top', color='black')
    
    axes[0].set_title("Radio data", fontsize=titlesize)
    axes[0].set_xlabel(xlabel, fontsize=labelsize)
    axes[0].set_ylabel(ylabel, fontsize=labelsize)
    axes[0].grid(color='white', linestyle='-', alpha=0.25)
    
    axes[1].set_title(obj.model_name.replace('_',' ')+" model", fontsize=titlesize)
    axes[1].set_xlabel(xlabel, fontsize=labelsize)
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    
    axes[2].set_title("Residual image", fontsize=titlesize)
    axes[2].set_xlabel(xlabel, fontsize=labelsize)
    axes[2].grid(color='black', linestyle='-', alpha=0.25)
    cbar = fig.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar = fig.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar.ax.set_ylabel('$\\mu$Jy arcsec$^{-2}$',fontsize=labelsize)

    if regrid:
        plt.savefig(halo.plotPath +halo.file.replace('.fits','')+'_mcmc_model'+obj.filename_append+'_REGRID.pdf')
    else:
        plt.savefig(halo.plotPath +halo.file.replace('.fits','')+'_mcmc_model'+obj.filename_append+'.pdf')
    #plt.show()
    plt.clf()
    plt.close(fig)


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
    asb = AnchoredSizeBar(ax.transData,length.value*2.5,
                          r"250 kpc",
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
                                angle=-bpa, loc='lower left', pad=0.3, borderpad=0.3,
                                frameon=True,color='lightskyblue')
    except:
        ae = AnchoredEllipse(ax.transData, width=bmaj.value, height=bmin.value,
                                angle=-bpa, loc='lower left', pad=0.3, borderpad=0.3,
                                frameon=True)

    ax.add_artist(ae)

def model_comparisson(halo, mask=False):
    fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True)
    bmin      = halo.bmin
    bmaj      = halo.bmaj
    scale     = 1.
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

    data = (np.copy(halo.result4.data)/halo.pix_area).to(uJyarcsec2).value
    noise = (halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
    masked_data = data.copy()
    #if mask:
    masked_data[~halo.result4.image_mask]= -10000.

    LEVEL = np.arange(1,7)*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value
    #NORM    = mplc.LogNorm(vmin=0.4*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
    #                    vmax=20*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value)
    #NORM = SymLogNorm(2.*halo.result4.params_units[0] , linscale=1.0, vmin=vmin, vmax=vmax)



    #NORMres = mplc.Normalize(vmin=-2.*(halo.rmsnoise/halo.pix_area).to(uJyarcsec2).value,
    #                    vmax=1.*(data/halo.pix_area).to(uJyarcsec2).value.max())
    NORMres = mplc.Normalize(vmin=-2.*noise, vmax=2.*masked_data.max())

    im1 = axes[0].imshow(masked_data,
                        cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        cont1 = axes[0].contour((model4/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont2 = axes[0].contour(masked_data,
                        colors='lightgreen', levels=np.array([-999.8]), alpha=0.9, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    except:
        print('PROCESSING: Failed making contours')
        pass

    axes[0].set_title('Circular\n $S_{\\mathrm{1.5 GHz}}=%.1f\\pm%.1f$ mJy' % (halo.result4.flux_val.value, halo.result4.flux_err.value), fontsize=15)
    axes[0].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[0].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[0].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[0], scale)
    draw_ellipse(halo,axes[0], bmin, bmaj)
    #plt.tight_layout()

    im2 = axes[1].imshow(masked_data,
                        cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        cont3 = axes[1].contour((model6/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont4 = axes[1].contour(masked_data,
                        colors='lightgreen', levels=np.array([-999.8]), alpha=0.9, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    except:
        print('PROCESSING: Failed making contours')
        pass

    axes[1].set_title('Elliptical\n $S_{\\mathrm{1.5 GHz}}=%.1f\\pm%.1f$ mJy' % (halo.result6.flux_val.value, halo.result8.flux_err.value), fontsize=15)
    axes[1].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[1].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[1].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[0], scale)
    draw_ellipse(halo,axes[0], bmin, bmaj)
    #plt.tight_layout()

    im3 = axes[2].imshow(masked_data,
                        cmap='inferno', origin='lower',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres)
    try:
        cont5 = axes[2].contour((model8/halo.pix_area).to(uJyarcsec2).value,
                        colors='white', levels=LEVEL, alpha=0.6,
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.)
        cont6 = axes[2].contour(masked_data,
                        colors='lightgreen', levels=np.array([-999.8]), alpha=0.9, linestyles='-',
                        extent=(ra.max(),ra.min(),dec.min(),dec.max()), norm = NORMres,linewidths=1.5)
    except:
        print('PROCESSING: Failed making contours')
        pass


    axes[2].set_title('Skewed \n $S_{\\mathrm{1.5 GHz}}=%.1f\\pm%.1f$ mJy' % (halo.result8.flux_val.value, halo.result8.flux_err.value), fontsize=15)
    axes[2].set_xlabel('RA [deg]', fontsize=labelsize)
    axes[2].set_ylabel('DEC [deg]', fontsize=labelsize)
    axes[2].grid(color='white', linestyle='-', alpha=0.25)
    draw_sizebar(halo,axes[0], scale)
    draw_ellipse(halo,axes[0], bmin, bmaj)
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
