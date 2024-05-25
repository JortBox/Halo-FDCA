#File to acces whether an image has to be shrunk to be able to fit it properly.
from astropy.io import fits 
import numpy as np

def shrink_fits(path, size: int = -1, pad: bool = False):
    oldhdul = fits.open(path)
    oldhdu: fits.PrimaryHDU = oldhdul[0] # type: ignore
    del oldhdul

    assert oldhdu.header["NAXIS"] == 4, "Only 4D data is supported"
    if not pad:
        assert oldhdu.data.shape[-1] >= size, "Size is larger than the image itself"
    
    nax = np.asarray(oldhdu.data[0,0].shape)//2
    
    if size == -1:
        cutout = oldhdu.data
    else:
        size = int(size//2)
        oldhdu.header["CRPIX1"] = int(oldhdu.header["CRPIX1"]) - (nax[0] - size) 
        oldhdu.header["CRPIX2"] = int(oldhdu.header["CRPIX2"]) - (nax[1] - size) 
        cutout = oldhdu.data[:,:,nax[0]-size:nax[0]+size,nax[1]-size:nax[1]+size]
        
    hdu = fits.PrimaryHDU(cutout, header=oldhdu.header) # type: ignore
    hdu.writeto(path.replace(".fits", "-cutout.fits"), overwrite=True)