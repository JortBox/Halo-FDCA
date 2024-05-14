#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
'''

from __future__ import division
import os
import logging
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

def get_initial_guess(halo):
    r_guess = halo.radius/(3.5*halo.pix_size)
    r_bound = halo.data.shape[0]/2.
    if r_guess >= halo.data.shape[1]/2.: r_guess = halo.data.shape[1]/4.

    diff   = np.abs(halo.margin)
    p0     = [halo.I0, halo.centre_pix[0]+diff[0],
              halo.centre_pix[1]+diff[2], r_guess,r_guess,r_guess,r_guess,0.,0.,0.]
    bounds = ([0.,0.,0.,0.,0.,0.,0.,-np.inf, 0., -np.inf],
              [np.inf,halo.data.shape[0],halo.data.shape[1],
               r_bound,r_bound,r_bound,r_bound,np.inf, np.inf, np.inf])
    return p0,bounds

def add_parameter_labels(obj, array) -> pd.Series|pd.DataFrame:
    full_array = np.zeros(obj.params.shape)
    full_array[obj.params] = np.array(array)
    parameterised_array = pd.DataFrame.from_dict({'params': full_array},
                            orient='index',columns=obj.paramNames).loc['params']
    return parameterised_array

def convolve_model(halo, Ir, rotate):
    if rotate:
        Ir = rotate_image(halo,Ir,decrease_fov=True)
    return convolve_with_gaussian(halo, Ir).ravel()

def gauss(x,mu,sigma, A):
    return A*np.exp(-1./2*((x-mu)/sigma)**2.)

def convolve_with_gaussian(obj, data):
    sigma1 = (obj.bmaj/obj.pix_size)/np.sqrt(8*np.log(2.))
    sigma2 = (obj.bmin/obj.pix_size)/np.sqrt(8*np.log(2.))
    kernel = Gaussian2DKernel(sigma1, sigma2, obj.bpa.to(u.rad))
    try:
        astropy_conv = convolve(data.value,kernel,boundary='extend',normalize_kernel=True)
    except:
        astropy_conv = convolve(data,kernel,boundary='extend',normalize_kernel=True)
    return astropy_conv

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

def masking(obj, mask, full_size=False):
    try: halo = obj.halo
    except: halo = obj

    if mask:
        '''FIND MASK:'''
        if os.path.isfile(halo.maskPath):
            mask = True
        else:
            mask=False
            obj.log.log(logging.ERROR,'No regionfile found,continueing without mask')


        '''SET MASK:'''
        if mask:
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
            obj.log.log(logging.INFO,'MCMC Mask set')
            os.remove(outfile)
    else:
        obj.log.log(logging.INFO,'MCMC No mask set')
        mask=False

    if mask==False:
        image_mask = np.zeros_like(obj.data)
    return image_mask, mask


def mask_region(infilename,ds9region,outfilename):
    hdu=fits.open(infilename)
    hduflat = flatten(hdu)
    map=hdu[0].data

    r = pyregion.open(ds9region)
    manualmask = r.get_mask(hdu=hduflat)
    hdu[0].data[0][0][np.where(manualmask == False)] = 0.0
    hdu[0].data[0][0][np.where(manualmask == True)] = 1.0
    hdu.writeto(outfilename,overwrite=True)

    return outfilename

def flatten(f):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this')
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

def setMask(self, data):
    regionpath = self.halo.maskPath
    outfile    = self.halo.basedir+'Data/Masks/'+self.halo.target+'_mask.fits'
    mask_region(self.halo.path, regionpath, outfile)

    '''In 'Radio_Halo', there is a function to decrease the fov of an image. The mask
       is made wrt the entire image. fov_info makes the mask the same shape as
       the image and overlays it'''
    self.image_mask = fits.open(outfile)[0].data[0,0,
                            self.halo.fov_info[0]:self.halo.fov_info[1],
                            self.halo.fov_info[2]:self.halo.fov_info[3]]


def regridding(obj, data, decrease_fov=False, mask=False):
    data_rot = rotate_image(obj, data.value, decrease_fov, mask)
    regrid   = regrid_to_beamsize(obj, data_rot)*data.unit
    return regrid

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
        #plt.imshow(img_rot[padY[0]:-padY[1], padX[0]:-padX[1]])
        #plt.show()
        return img_rot[padY[0]:-padY[1], padX[0]:-padX[1]]
    else:
        img_rot = ndimage.rotate(img, -obj.bpa.value, reshape=False,mode='constant',cval=cval)
        f= img_rot[obj.margin[2]:obj.margin[3], obj.margin[0]:obj.margin[1]]
        #plt.imshow(f)
        #plt.show()
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


def transform_units(obj, params):
    params[1] = params[1] + obj.halo.fov_info_mcmc[2]
    params[2] = params[2] + obj.halo.fov_info_mcmc[0]
    param_sky = wcs.utils.pixel_to_skycoord(
        params[1], params[2], wcs.WCS(obj.halo.header), origin=1
    )
    params[1] = param_sky.ra.deg
    params[2] = param_sky.dec.deg
    
    params[0] = ((u.Jy * params[0] / obj.halo.pix_area).to(uJyarcsec2)).value
    #params[1] = (
    #    params[1] - self.centre_pix[0]
    #) * self.halo.pix_size.value + self.centre_wcs.ra.deg
    #params[2] = (
    #    params[2] - self.centre_pix[1]
    #) * self.halo.pix_size.value + self.centre_wcs.dec.deg
    params[3] = ((params[3] * obj.halo.pix2kpc).to(u.kpc)).value
    if obj.modelName in ["ellipse", "rotated_ellipse", "skewed"]:
        params[4] = ((params[4] * obj.halo.pix2kpc).to(u.kpc)).value

    if obj.modelName == "skewed":
        params[5] = ((params[5] * obj.halo.pix2kpc).to(u.kpc)).value
        params[6] = ((params[6] * obj.halo.pix2kpc).to(u.kpc)).value
    if obj.modelName in ["rotated_ellipse", "skewed"]:
        params[obj.at("ang")] = params[obj.at("ang")]
    return params