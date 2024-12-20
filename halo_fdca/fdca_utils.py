#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
'''

from __future__ import division
import os, sys
import pyregion

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import ndimage
from skimage.measure import block_reduce
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

np.seterr(divide='ignore', invalid='ignore')

rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
Jydeg2 = u.Jy / (u.deg * u.deg)
mJyarcsec2 = u.mJy / (u.arcsec * u.arcsec)
uJyarcsec2 = 1.0e-3 * u.mJy / (u.arcsec * u.arcsec)


def set_linked_loc(obj, array, loop=True):
    for i, fit in enumerate(obj.fits):
        if obj.link[i]:
            if i == 0:
                link_x = array[f'comp_{i}']['x0']
                link_y = array[f'comp_{i}']['y0']
            if obj.link[i] and i > 0:
                array[f'comp_{i}']['x0'] = link_x
                array[f'comp_{i}']['y0'] = link_y
                fit.frozen_vals[fit.frozen[fit.frozen].index.get_loc("x0")] = link_x
                fit.frozen_vals[fit.frozen[fit.frozen].index.get_loc("y0")] = link_y
                
    return array

def get_initial_guess(halo):
    r_guess = halo.radius/(3.5*halo.pix_size)
    r_bound = halo.data.shape[0]/2.
    if r_guess >= halo.data.shape[1]/2.: r_guess = halo.data.shape[1]/4.

    p0     = [
        halo.I0, 
        halo.centre_pix[0]+halo.margin[2], 
        halo.centre_pix[1]+halo.margin[0], 
        r_guess,r_guess,r_guess,r_guess,0.,0.,0.
    ]
    bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf, 0., -np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
               r_bound,r_bound,r_bound,r_bound,np.inf, np.inf, np.inf])
    return p0,bounds

def add_parameter_labels(obj, array=None):
    full_array = np.zeros(obj.params.shape)
    array = np.zeros(obj.params[obj.params].shape) if array is None else array
    full_array[obj.params] = np.array(array)
    parameterised_array = pd.DataFrame.from_dict({'params': full_array},
                            orient='index',columns=obj.paramNames).loc['params']
    return parameterised_array

def add_labels(obj, array=None, expand=True):
    array = np.zeros(obj.dim) if array is None else array
    full_array = np.zeros(obj.params.shape)
    full_array[obj.params & ~obj.frozen] = np.array(array)
    parameterised_array = pd.DataFrame.from_dict({'params': full_array},
                            orient='index',columns=obj.paramNames).loc['params']
    if expand:
        parameterised_array[obj.frozen] = obj.frozen_vals
    return parameterised_array

def gauss(x, mu, sigma, A):
    return A*np.exp(-1./2*((x-mu)/sigma)**2.)

def noise_modelling(obj):
    noise = 15.*(np.random.randn(len(obj.halo.x_pix),len(obj.halo.y_pix))-0.030)*u.Jy
    return (convolve_with_gaussian(obj.halo, noise)*obj.halo.rmsnoise).value

def noise_characterisation(obj, data):
    mask = np.copy(data)
    #mask[obj.data.value>2*obj.rmsnoise.value]=np.nan

    nbin = 100
    
    bins = np.linspace(-5*obj.rmsnoise.value,
                            8*obj.rmsnoise.value, nbin)
    x    = np.linspace(-5*obj.rmsnoise.value,
                            8*obj.rmsnoise.value, 1000)
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])

    hist_data, data_bins = np.histogram(mask.ravel(), bins=bins)
    popt, pcov = curve_fit(gauss, xdata=binscenters, ydata=hist_data,
                            p0=(0,0.000003, 5000))
    return popt

def advanced_noise_modeling(obj,seed=False):
    if seed:
        np.random.seed(12345)

    noise       = np.random.randn(len(obj.halo.x_pix),len(obj.halo.y_pix))*u.Jy
    noise_conv  = convolve_with_gaussian(obj.halo, noise)
    var         = np.mean((noise_conv.ravel())**2.)- np.mean(noise_conv.ravel())**2.
    noise_conv  = noise_conv*(obj.halo.noise_char[1]/np.sqrt(var))
    noise_conv -= np.mean(noise_conv.ravel()) - obj.halo.noise_char[0]
    #plot.quick_imshow(obj.halo, noise_conv*u.Jy, noise=False)
    return noise_conv*u.Jy

def create_artificial_halo(obj, model, seed):
    theory_noise = advanced_noise_modeling(obj, seed).value
    #plot.quick_imshow(obj.halo, (model+theory_noise)*u.Jy, noise=False)
    return model+theory_noise

def export_fits(data, path, header=None):
    try:
        hdu = fits.PrimaryHDU(data.value, header=header)
    except:
        hdu = fits.PrimaryHDU(data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)

def masking(obj, full_size: bool=False, verbose: bool=False):
    try: halo = obj.halo
    except: halo = obj
    
    mask = halo.mask

    if mask:
        '''SET MASK:'''
        regionpath = halo.maskPath
        outfile    = halo.basedir+'/'+halo.file.replace('.fits','')+'_MASK.fits'
        mask_region(halo.path, regionpath, outfile)

        '''In 'Radio_Halo', there is a function to decrease the fov of an image. The mask
            is made wrt the entire image. fov_info makes the mask the same shape as
            the image and overlays it'''
        if full_size:
            image_mask = fits.open(outfile)[0].data[0,0,:,:]
        else:
            image_mask = fits.open(outfile)[0].data[
                0,0,halo.fov_info[0]:halo.fov_info[1],halo.fov_info[2]:halo.fov_info[3]
            ]
            
        if verbose:
            obj.logger.info('Mask set')
        else:
            obj.logger.debug('Mask set')
    else:
        if full_size:

            image_mask_shape = fits.open(halo.path)[0].data[0,0,:,:].shape
        else:
            image_mask_shape = fits.open(halo.path)[0].data[
                0,0,halo.fov_info[0]:halo.fov_info[1],halo.fov_info[2]:halo.fov_info[3]
            ].shape

        if verbose:
            obj.logger.warning('No mask set')
        else:
            obj.logger.debug('No mask set')
        image_mask = np.ones(image_mask_shape)
    return image_mask.astype(bool)


def mask_region(infilename: str, ds9region: str, outfilename: str):
    hdul=fits.open(infilename)
    hduflat = flatten(hdul)
    assert hdul[0].header["NAXIS"] == 4, "Only 4D data is supported"
    data = hdul[0].data[0,0]
    
    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    
    data[manualmask == False] = 1.0
    data[manualmask == True] = 0.0
    hdul[0].data[0,0] = data
    hdul.writeto(outfilename,overwrite=True)

    return outfilename

def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this') # type: ignore
    if naxis == 2:
        return fits.PrimaryHDU(header=f[0].header,data=f[0].data)

    w  = wcs.WCS(f[0].header)
    wn = wcs.WCS(naxis=2)

    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"]=2
    copy=('EQUINOX','EPOCH','BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'TELESCOP', 'OBSERVER')
    for k in copy:
        r=f[0].header.get(k)
        if r is not None:
            header[k]=r

    slice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            slice.append(np.s_[:],)
        else:
            slice.append(0)

    hdu = fits.PrimaryHDU(header=header,data=f[0].data[tuple(slice)])
    return hdu

def get_rms(hdu,boxsize=1000,niter=200,eps=1e-6,verbose=False):
    hdu = fits.open(hdu)
    data=hdu[0].data
    hdu.close()
    if len(data.shape)==4:
        _,_,ys,xs=data.shape
        subim=data[0,0,0:ys,0:xs].flatten()
    else:
        ys,xs=data.shape
        subim=data[0:ys,0:xs].flatten()
    oldrms=1
    subim = np.delete(subim,np.where(np.isnan(subim)))
    for i in range(niter):
        rms=np.std(subim)
        if np.abs(oldrms-rms)/rms < eps:
            return rms
        subim=subim[np.abs(subim)<5*rms]
        oldrms=rms
    raise Exception('Failed to converge')

def findrms(data, niter=100, maskSup=1e-7):
    m      = data[np.abs(data)>maskSup]
    rmsold = np.std(m)
    diff   = 1e-1
    cut    = 3.
    bins   = np.arange(np.min(m),np.max(m),(np.max(m)-np.min(m))/30.)
    med    = np.median(m)

    for i in range(niter):
        ind = np.where(np.abs(m-med)<rmsold*cut)[0]
        rms = np.std(m[ind])
        if np.abs((rms-rmsold)/rmsold)<diff: break
        rmsold = rms
    return rms

def regridding(obj, data, decrease_fov=False, mask=False):
    data_rot = rotate_image(obj, data.value, decrease_fov, mask)
    regrid   = regrid_to_beamsize(obj, data_rot)*data.unit
    return regrid

def pad_image(img):
    pivot = ((np.sqrt(2.0) / 2.0 - 0.5) * np.array(img.shape)).astype(
        np.int64
    )
    padX = [pivot[0], pivot[0]]
    padY = [pivot[1], pivot[1]]
    padded_img = np.pad(img, [padY, padX], "constant")
    fov_info = [
        -pivot[0],img.shape[0] + pivot[0],-pivot[1],img.shape[1] + pivot[1]
    ]
    return padded_img, fov_info

def rotate_image(obj,img, decrease_fov=False, mask=False):
    if mask: cval=1
    else: cval=0

    if not decrease_fov:
        if np.array(img.shape)[0]%2 == 0:
            img = np.delete(img, 0, 0)
        if np.array(img.shape)[1]%2 == 0:
            img = np.delete(img, 0, 1)

        pivot = (np.array(img.shape)/2).astype(np.int64)
        padX  = [int(img.shape[1]) - pivot[0], pivot[0]]
        padY  = [int(img.shape[0]) - pivot[1], pivot[1]]
        img_pad  = np.pad(img, [padY, padX], 'constant', constant_values=(cval))
        img_rot  = ndimage.rotate(img_pad, -obj.bpa.value, reshape=False,mode='constant',cval=cval)
        return img_rot[padY[0]:-padY[1], padX[0]:-padX[1]]
    else:
        img_rot = ndimage.rotate(img, -obj.bpa.value, reshape=False,mode='constant',cval=cval)
        f= img_rot[obj.margin[2]:obj.margin[3], obj.margin[0]:obj.margin[1]]
        return f

def regrid_to_beamsize(obj, img, accuracy=100.):
    y_scale = np.sqrt(obj.beam_area*obj.bmin/obj.bmaj).value
    x_scale = (obj.beam_area/y_scale).value
    
    new_pix_size = np.array((y_scale,x_scale))
    accuracy = int(1./accuracy*100)

    scale = np.round(accuracy*new_pix_size/obj.pix_size).astype(np.int64).value
    pseudo_size = (accuracy*np.array(img.shape) ).astype(np.int64)
    pseudo_array = np.zeros((pseudo_size))

    orig_scale = (np.array(pseudo_array.shape)/np.array(img.shape)).astype(np.int64)
    elements   = np.prod(np.array(orig_scale,dtype='float64'))

    if accuracy == 1:
        pseudo_array = np.copy(img)
    else:
        for j in range(img.shape[0]):
            for i in range(img.shape[1]):
                pseudo_array[orig_scale[1]*i:orig_scale[1]*(i+1),
                             orig_scale[0]*j:orig_scale[0]*(j+1)] = img[i,j]/elements

    f= block_reduce(pseudo_array, block_size=tuple(scale), func=np.sum, cval=0)
    f=np.delete(f, -1, axis=0)
    f=np.delete(f, -1, axis=1)
    #plt.imshow(f)
    #plt.show()
    #print(pseudo_array.shape, scale, f.shape)
    return f

def gamma_dist(x, shape, scale):
    from scipy.special import gamma
    return (x**(shape-1.)*np.exp(-x/scale))/(gamma(shape)*(scale**shape))

def at(obj, parameter):
    par = np.array(obj.paramNames)[obj.params]
    return np.where(par == parameter)[0][0]

def transform_units(obj, params_orig, err=False, unlabeled=False, keys=None):
    params = params_orig.copy()
    if unlabeled:
        assert keys is not None, "Keys must be provided"
        for i,key in enumerate(keys):
            if key in ["x0", "y0"]:
                if err:
                    params[i] *= obj.halo.pix_size.to(u.deg).value
                else:
                    if key == "x0":
                        params[i] += obj.halo.fov_info[2]
                    else:
                        params[i] += obj.halo.fov_info[0]
        
            elif key == "I0":
                params[i] = ((u.Jy * params[i] / obj.halo.pix_area).to(uJyarcsec2)).value
            elif key in ["r1", "r2", "r3", "r4"]:
                params[i] = ((params[i] * obj.halo.pix2kpc).to(u.kpc)).value

        if "x0" in keys and "y0" in keys:
            param_sky = wcs.utils.pixel_to_skycoord(
                params[keys=='x0'], params[keys=='y0'], wcs.WCS(obj.halo.header), origin=1
            )
            params[keys=='x0'] = param_sky.ra.deg
            params[keys=='y0'] = param_sky.dec.deg
        return params
    else:
        for key in params.keys():
            if key in ["x0", "y0"]:
                if err:
                    params[key] *= obj.halo.pix_size.to(u.deg).value
                else:
                    if key == "x0":
                        params[key] += obj.halo.fov_info[2]
                    else:
                        params[key] += obj.halo.fov_info[0]
        
            elif key == "I0":
                params[key] = ((u.Jy * params[key] / obj.halo.pix_area).to(uJyarcsec2)).value
            elif key in ["r1", "r2", "r3", "r4"]:
                params[key] = ((params[key] * obj.halo.pix2kpc).to(u.kpc)).value

        param_sky = wcs.utils.pixel_to_skycoord(
            params['x0'], params['y0'], wcs.WCS(obj.halo.header), origin=1
        )
        params['x0'] = param_sky.ra.deg
        params['y0'] = param_sky.dec.deg
        return params






def convolve_with_gaussian(obj, data):
    sigma1 = (obj.bmaj/obj.pix_size)/np.sqrt(8*np.log(2.))
    sigma2 = (obj.bmin/obj.pix_size)/np.sqrt(8*np.log(2.))
    kernel = Gaussian2DKernel(sigma1, sigma2, obj.bpa.to(u.rad))
    try:
        astropy_conv = convolve(data.value,kernel,boundary='extend',normalize_kernel=True)
    except:
        astropy_conv = convolve(data,kernel,boundary='extend',normalize_kernel=True)
    return astropy_conv


def convolve_model(halo, Ir, rotate):
    if rotate:
        Ir = rotate_image(halo,Ir,decrease_fov=True)
    return convolve_with_gaussian(halo, Ir).ravel()

def circle_model(obj, theta, rotate=False):
    G   = ((obj.x_pix-theta['x0'])**2+(obj.y_pix-theta['y0'])**2)/theta['r1']**2
    Ir  = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
    return convolve_model(obj.halo, Ir, rotate).ravel()

def ellipse_model(obj, theta, rotate=False):
    G  = ((obj.x_pix-theta['x0'])/theta['r1'])**2+((obj.y_pix-theta['y0'])/theta['r2'])**2
    Ir = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
    return convolve_model(obj.halo, Ir, rotate).ravel()

def rotated_ellipse_model(obj, theta, rotate=False):
    x  = (obj.x_pix-theta['x0'])*np.cos(theta['ang']) + (obj.y_pix-theta['y0'])*np.sin(theta['ang'])
    y  = -(obj.x_pix-theta['x0'])*np.sin(theta['ang']) + (obj.y_pix-theta['y0'])*np.cos(theta['ang'])
    G  = (x/theta['r1'])**2.+(y/theta['r2'])**2.
    Ir = theta['I0']*np.exp(-G**(0.5+theta['k_exp']))+theta['off']
    return convolve_model(obj.halo, Ir, rotate).ravel()

def skewed_model(obj, theta, rotate=False):
    G_pp = G(obj.x_pix, obj.y_pix, theta['I0'],theta['x0'],theta['y0'],theta['r1'],theta['r3'],theta['ang'],  1.,  1.)
    G_mm = G(obj.x_pix, obj.y_pix, theta['I0'],theta['x0'],theta['y0'],theta['r2'],theta['r4'],theta['ang'], -1., -1.)
    G_pm = G(obj.x_pix, obj.y_pix, theta['I0'],theta['x0'],theta['y0'],theta['r1'],theta['r4'],theta['ang'],  1., -1.)
    G_mp = G(obj.x_pix, obj.y_pix, theta['I0'],theta['x0'],theta['y0'],theta['r2'],theta['r3'],theta['ang'], -1.,  1.)

    Ir = theta['I0']*(G_pp+G_pm+G_mm+G_mp)
    return convolve_model(obj.halo, Ir, rotate).ravel()

def G(x,y, I0, x0, y0, re_x,re_y, ang, sign_x, sign_y):
    x_rot =  (x-x0)*np.cos(ang)+(y-y0)*np.sin(ang)
    y_rot = -(x-x0)*np.sin(ang)+(y-y0)*np.cos(ang)
    func  = (np.sqrt(sign_x * x_rot)**4.)/(re_x**2.) +\
            (np.sqrt(sign_y * y_rot)**4.)/(re_y**2.)

    exponent = np.exp(-np.sqrt(func))
    exponent[np.where(np.isnan(exponent))]=0.
    return exponent

def set_data_to_use(obj, data):
    # obj: has to be Fitting or Processing object
    if obj.rebin:
        binned_data = regridding(obj.halo, data, decrease_fov=obj.halo.cropped)
        #if not obj.mask:
        #image_mask = np.ones(obj.data.shape)
            
        binned_image_mask = regridding(
            obj.halo, 
            obj.image_mask.astype(int) * u.Jy, 
            decrease_fov=obj.halo.cropped, 
            mask=obj.mask
        ).value
        use = binned_data.value
        
        return use.ravel()[
            binned_image_mask.ravel()
            >= obj.mask_treshold * binned_image_mask.max()
        ]
    else:
        if obj.mask:
            return obj.data.value.ravel()[obj.image_mask.astype(int).ravel() <= 0.5]
        else:
            return obj.data.value.ravel()