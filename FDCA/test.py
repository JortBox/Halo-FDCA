import sys
import os
import logging
import time
from multiprocessing import Pool

# Scipy, astropy, emcee imports
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from scipy import ndimage
from skimage.measure import block_reduce
from skimage.transform import rescale


import utils
x = np.linspace(0,1000,500)
y = utils.gamma_dist(x, 2.6,100.)
plt.plot(x,y)
plt.xlim(0,1000)
plt.show()
sys.exit()

x=np.arange(0,100)
y=np.arange(0,100)
#x,y = np.meshgrid(x,y)

img = np.zeros((x.shape[0],y.shape[0]))
img[25:75,25:75]=1

pivot = (np.array(img.shape)/2).astype(np.int64)
padX  = [int(img.shape[1]) - pivot[0], pivot[0]]
padY  = [int(img.shape[0]) - pivot[1], pivot[1]]
img_pad  = np.pad(img, [padY, padX], 'constant')
img_rot  = ndimage.rotate(img_pad, -45, reshape=False, order=4)
#plt.imshow(img_rot[padY[0]:-padY[1], padX[0]:-padX[1]])
#plt.show()
img_rot = img_rot[padY[0]:-padY[1], padX[0]:-padX[1]]



y_scale = 7
x_scale = 10
new_pix_size = np.array((y_scale,x_scale))
accuracy = int(1./1.*100)

scale = np.round(accuracy*new_pix_size).astype(np.int64)
pseudo_size = (accuracy*np.array(img_rot.shape) ).astype(np.int64)
pseudo_array = np.zeros((pseudo_size))

orig_scale = (np.array(pseudo_array.shape)/np.array(img.shape)).astype(np.int64)
elements   = np.prod(np.array(orig_scale,dtype='float64'))

if accuracy is 1:
    pseudo_array = np.copy(img_rot)
else:
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            pseudo_array[orig_scale[1]*i:orig_scale[1]*(i+1),
                     orig_scale[0]*j:orig_scale[0]*(j+1)] = img_rot[i,j]/elements

f= block_reduce(pseudo_array, block_size=tuple(scale), func=np.sum, cval=0)
#f=np.delete(f, -1, axis=0)
#f=np.delete(f, -1, axis=1)

plt.imshow(img, origin='bottom', norm = Normalize(vmin=0,vmax=1.2), cmap='cubehelix',
extent=(0-x.max()/2,0+x.max()/2, 0-y.max()/2, 0+y.max()/2))
plt.xlabel('Offset [Pixels]')
plt.ylabel('Offset [Pixels]')
plt.savefig('paper_plot_1.pdf')
plt.show()
plt.imshow(img_rot, origin='bottom', norm = Normalize(vmin=0,vmax=1.2), cmap='cubehelix',
extent=(0-x.max()/2,0+x.max()/2, 0-y.max()/2, 0+y.max()/2))
plt.colorbar()
plt.xlabel('Offset [Pixels]')
plt.ylabel('Offset [Pixels]')
plt.savefig('paper_plot_2.pdf')
plt.show()

x = np.arange(0,f.shape[1])
y = np.arange(0,f.shape[0])
plt.imshow(f, origin='bottom', norm = Normalize(vmin=0,vmax=80), cmap='cubehelix',
extent=(0-x.max()/2,0+x.max()/2, 0-y.max()/2, 0+y.max()/2))
plt.colorbar()
plt.xlabel('Offset [Pixels]')
plt.ylabel('Offset [Pixels]')
plt.savefig('paper_plot_3.pdf')
plt.show()


'''
target='Abell 1156'
cat = 'VII/110A/table3'
table  = Vizier.query_object(target,catalog=cat)
print(table[cat])
sys.exit()


cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
factor    = cosmology.kpc_proper_per_arcmin(0.13)
print((0.8644*u.Mpc/factor).to(u.deg))


hdul = fits.open('/Users/jortboxelaar/Desktop/cutout_PSZ2G084.69+42.28_masksubROBUST-0.5TAPER50kpc-MFS-image.fits')
hdul[0].data = hdul[0].data[:,:,60:160,60:160]
hdul[0].header['CRPIX1'] -= 60
hdul[0].header['CRPIX2'] -= 60
hdul.writeto('/Users/jortboxelaar/Desktop/cutout-extra_PSZ2G084.69+42.28_masksubROBUST-0.5TAPER50kpc-MFS-image.fits', overwrite=True)
'''
