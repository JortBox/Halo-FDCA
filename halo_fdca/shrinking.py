#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: May 2024

File to acces when an image has to be shrunk to be able to fit it properly.
'''

import sys
import argparse
import numpy as np

from astropy import wcs
from astropy.io import fits 
from astropy.coordinates import SkyCoord

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_in', help='(str) Cluster object name', type=str, default='none')
    parser.add_argument('-path_out', help='(str) Path to code output. Default: directory code is in.', default='none', type=str)
    parser.add_argument('-size', default=-1, type=int)
    parser.add_argument('-pad', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.path_out == "none":
        args.path_out = args.path_in
    return args


def check_input(
    image_centre: tuple[int,int] | SkyCoord | None,
    image_shape: tuple[int,int], 
    pad: bool, 
    size: int | tuple[int,int], 
    fitwcs: wcs.WCS
):     
    if isinstance(size, int):
        size = np.asarray((size//2, size//2), dtype=np.int64)
    elif isinstance(size, tuple[int,int]):
        size = np.asarray(size, dtype=np.int64)//2
    else:
        raise TypeError
    
    if isinstance(image_centre, tuple[int,int]):
        nax = np.asarray(image_centre)
    elif isinstance(image_centre, SkyCoord):
        nax = np.asarray(
            wcs.utils.skycoord_to_pixel(image_centre, fitwcs, origin=1), 
            dtype=np.int64
        )
    elif image_centre is None:
        nax = np.asarray(image_shape)//2
    else:
        raise TypeError
    
    assert np.all(size > 0)
    if not pad:
        assert image_shape[-1] >= size, "Size is larger than the image itself, allow padding"
        assert nax[0] - size[0] > 0
        assert nax[0] + size[0] <= image_shape[0]
        assert nax[1] - size[1] > 0 
        assert nax[1] + size[1] <= image_shape[1]  
        
    return nax, size


def shrink_fits(path, size: int | tuple[int,int] = -1, image_centre: tuple[int,int] | SkyCoord | None = None, pad: bool = False):
    oldhdul = fits.open(path)
    oldhdu: fits.PrimaryHDU = oldhdul[0] # type: ignore
    fitwcs = wcs.WCS(oldhdu.header)
    del oldhdul
    
    assert oldhdu.header["NAXIS"] == 4, "Only 4D data is supported"
    image_shape: tuple[int,int] = oldhdu.data[0,0].shape
         
    if size == -1:
        cutout = oldhdu.data
        hdu = fits.PrimaryHDU(cutout, header=oldhdu.header) # type: ignore
        hdu.writeto(path.replace(".fits", "-cutout.fits"), overwrite=True)
    
    nax, size = check_input(image_centre, image_shape, pad, size, fitwcs)
    
    if pad:
        limits = np.asarray([
            nax[0] - size[0],
            nax[0] + size[0] - image_shape[0],
            nax[1] - size[1],
            nax[1] + size[1] - image_shape[1]
        ]) 
        if limits[0] > 0: limits[0] = 0
        else: limits[0] = abs(limits[0])
            
        if limits[2] > 0: limits[2] = 0
        else: limits[2] = abs(limits[2])
            
        if limits[1] < 0: limits[1] = 0 
        if limits[3] < 0: limits[3] = 0
        
        newimage = np.pad(oldhdu.data[0,0], ((limits[0], limits[1]),(limits[2], limits[3])), "constant", constant_values=0)   
        newdata = np.ones((1,1) + newimage.shape) 
        newdata[:,:,0,0] = oldhdu.data[:,:,0,0]
        newdata[0,0] = newimage
        newnax = nax + limits[::2]
    else:
        newdata = oldhdu.data
        newnax = nax
    
    
    oldhdu.header["CRPIX1"] = int(oldhdu.header["CRPIX1"]) - (newnax[0] - size[0]) 
    oldhdu.header["CRPIX2"] = int(oldhdu.header["CRPIX2"]) - (newnax[1] - size[1]) 
    cutout = newdata[
        :,
        :,
        newnax[0]-size[0] : newnax[0]+size[0],
        newnax[1]-size[1] : newnax[1]+size[1]
    ]
        
    hdu = fits.PrimaryHDU(cutout, header=oldhdu.header) # type: ignore
    hdu.writeto(path.replace(".fits", "-cutout.fits"), overwrite=True)
    
if __name__ == "__main__":
    args = arguments()
    shrink_fits(args.path_in, size=args.size, pad=args.pad)