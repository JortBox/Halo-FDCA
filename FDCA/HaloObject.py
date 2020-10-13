#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author: J.M. Boxelaar
Version: 08 June 2020
'''
# Built in module imports
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

# Subfiles imports
#import plotting_fits as plot
#import markov_chain_monte_carlo
from . import fdca_utils as utils

#plt.rc('text',usetex=True)
#plt.rc('font', family='serif')
np.seterr(divide='ignore', invalid='ignore')

rad2deg    = 180./np.pi
deg2rad    = np.pi/180.
Jydeg2     = u.Jy/(u.deg*u.deg)
mJyarcsec2 = u.mJy/(u.arcsec*u.arcsec)
uJyarcsec2 = 1.e-3*u.mJy/(u.arcsec*u.arcsec)

class Radio_Halo(object):
    '''
    -CLASS DESCRIPTION-
    This class initiates a Radio_Halo object containing all image and physical
    information. A Halo obect has to be passed to the MCMC module.
    The Halo class aslo performs preliminary processes to make MCMC possible

    -INPUT-
    object (str): Name of galaxy cluster. Currently only supports its PSZ2 or MCXC name.
            If another object needs to be passed, fill in the physical
            characteristics manually
    path (str): Path to data read from 'database.dat'. Compatible with
          Leiden Observatory data structure.
    decrease_fov (bool): Declare if image size has to be decreased before MCMCing. Amount
                  of decreasement has ben automatically set to 3.5*r_e in
                  self.exponentialFit().
    logger: Configured logging object to log info to a .log file. If not given,
            a new file will be created.
    loc (SkyCoord object): Manually inserted cluster location as an astropy.SkyCoord
                        object. If None: location is gathered from a Vizier query.
                        Otherwise: provide Astropy SkyCoord object with approximate
                        centre of radio halo.
    M500 (float): Manually inserted mass. If None: mass is gathered from a Vizier query
                  If not None: must be value given in 1e14 SolMass
    R500 (float): Manually inserted R500 radius. If None: radius is gathered from
                  a Vizier query (MCXC only). If not None, must be value given
                  in Mega Parsec.
    z (float): Manually inserted redshift. If None: redshift is gathered from
               a Vizier query
    spectr_index (float): Manually inserted halo spectral index (S_v = v^(spectr_index)).
                          Value is used when extrapolating flux density and calculating
                          power values. Default is -1.2 (No conclusions can be drawn
                          from using this default value in calculations).
    '''
    def __init__(self, object, path, decreased_fov=False, maskpath=None, mask=False,
                logger=logging, loc=None, M500=None, R500=None, z=None, spectr_index=-1.2):

        self.user_radius = R500
        self.user_loc    = loc
        self.log = logger
        if object[:4] == 'MCXC':
            self.cat = 'J/A+A/534/A109/mcxc'
        elif object[:4] == 'PSZ2':
            self.cat = 'J/A+A/594/A27/psz2'
        elif object[:3] == 'WHL':
            self.cat = 'J/MNRAS/436/275/table2'
        elif object[:5] == 'Abell':
            self.cat = 'VII/110A/table3'
        else:
            self.cat=None
            self.log.log(logging.ERROR,'Unknown what catalogue to use. If no costum values are given, filling values will be used')

        self.target = str(object)
        self.path   = path
        self.alpha  = spectr_index
        self.name   = self.target.replace('MCXC','MCXC ')
        self.name   = self.target.replace('PSZ2','PSZ2 ')
        self.name   = self.target.replace('Abell','Abell ')
        self.name   = self.target.replace('WHL','')
        self.cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        self.table  = Vizier.query_object(self.name,catalog=self.cat)

        self.initiatePaths(maskpath)
        data = self.unpack_File()
        self.get_beam_area()
        self.original_image = np.copy(data)

        x = np.arange(0, data.shape[1], step=1, dtype='float')
        y = np.arange(0, data.shape[0], step=1, dtype='float')
        self.x_pix, self.y_pix = np.meshgrid(x,y)

        self.get_object_location(loc)
        self.extract_object_info(M500, R500, z)

        self.fov_info              = [0,data.shape[0],0,data.shape[1]]
        self.image_mask, self.mask = utils.masking(self, mask)
        self.exponentialFit(data, first=True) # Find centre of the image centre_pix

        if self.header['BUNIT']=='JY/BEAM' or self.header['BUNIT']=='Jy/beam':
            self.data = data*(u.Jy/self.beam2pix)
        else:
            self.log.log(logging.CRITICAL,'Possibly other units than jy/beam, CHECK HEADER UNITS!')
            sys.exit()
            
        self.pix_to_world()
        self.set_image_characteristics(decreased_fov)

    def initiatePaths(self, maskpath):
        self.basedir = os.getcwd()+'/'

        txt           = self.path.split('/')
        self.file     = txt[-1]
        self.dataPath = '/'+'/'.join(txt[:-1])+'/'

        if not os.path.isdir(self.basedir+'Output/'):
            self.log.log(logging.INFO,'Creating modelling directory')
            os.makedirs(self.basedir+'Output/')

        self.plotPath  = self.basedir+'Output/Plots/'
        self.modelPath = self.basedir+'Output/Samples/'

        if not os.path.isdir(self.modelPath):
            self.log.log(logging.INFO,'Creating modelling directory')
            os.makedirs(self.modelPath)
        if not os.path.isdir(self.plotPath):
            self.log.log(logging.INFO,'Creating plotting directory')
            os.makedirs(self.plotPath)

        if maskpath == None:
            self.maskPath = self.basedir+'Output/'+self.target+'.reg'
        else:
            self.maskPath = maskpath

    def get_object_location(self, loc):
        if loc is not None:
            self.loc = loc
        elif self.target[:4] == 'MCXC':
            coord    = str(self.table[self.cat]['RAJ2000'][0])+' '\
                        + str(self.table[self.cat]['DEJ2000'][0])
            self.loc = SkyCoord(coord, unit=(u.hourangle,u.deg))
        elif self.target[:5] == 'Abell':
            coord    = str(self.table[self.cat]['_RA.icrs'][0])+' '\
                        + str(self.table[self.cat]['_DE.icrs'][0])
            self.loc = SkyCoord(coord, unit=(u.hourangle,u.deg))
        elif self.target[:4] == 'PSZ2':
            coord    = [self.table[self.cat]['RAJ2000'][0],self.table[self.cat]['DEJ2000'][0]]
            self.loc = SkyCoord(coord[0], coord[1], unit=u.deg)
        elif self.target[:3] == 'WHL':
            coord    = [self.table[self.cat]['RAJ2000'][0],self.table[self.cat]['DEJ2000'][0]]
            self.loc = SkyCoord(coord[0], coord[1], unit=u.deg)
        else:
            self.log.log(logging.WARNING,'No halo world location given. Assuming image centre.')
            self.log.log(logging.INFO,'- Not giving an approximate location can affect MCMC performance -')
            cent_pix = (np.array([self.original_image.shape])/2).astype(np.int64)
            w        = wcs.WCS(self.header)
            coord    = w.celestial.wcs_pix2world(cent_pix,0)
            self.loc = SkyCoord(coord[0,0], coord[0,1], unit=u.deg)
            self.user_loc = False




    def extract_object_info(self, M500, R500, z):
        '''Written for MCXC catalogue. Information is gathered from there. If custom
        parameters are given, these will be used. if nothing is found, filling
        values are set. This is only a problem if you try to calculate radio power.'''
        try:
            if self.target[:4] == 'MCXC':
                self.M500 = float(self.table[self.cat]['M500'][0])*1.e14*u.Msun
                self.L500 = float(self.table[self.cat]['L500'][0])*1.e37*u.Watt
                self.R500 = float(self.table[self.cat]['R500'][0])*u.Mpc
                self.z    = float(self.table[self.cat]['z'][0])
                self.M500_std = 0.*u.Msun

            elif self.target[:3] == 'WHL':
                self.z    = float(self.table[self.cat]['z'][0])
                self.R500 = 1.*u.Mpc
                self.M500 = 3.e14*u.Msun
                self.user_radius = False
                self.log.log(logging.WARNING,'No R500 key found. setting R500='\
                                        +str(self.R500.value)+'Mpc to continue')

            elif self.target[:5] == 'Abell':
                try:
                    self.z = float(self.table[self.cat]['z'][0])
                except:
                    self.z = 0.1
                    self.log.log(logging.WARNING,'No valid z key found. setting z='\
                                        +str(self.z)+' as filling to continue. Ignore this message if z != None')
                self.R500 = 1.*u.Mpc
                self.user_radius = False
                self.log.log(logging.WARNING,'No R500 key found. setting R500='\
                                        +str(self.R500.value)+'Mpc to continue')


            elif self.target[:4] == 'PSZ2':
                self.M500 = float(self.table[self.cat]['MSZ'][0])*1.e14*u.Msun
                self.M500_std = np.max([float(self.table[self.cat]['E_MSZ'][0]),
                                   float(self.table[self.cat]['e_MSZ'][0])])*1.e14*u.Msun
                self.z    = float(self.table[self.cat]['z'][0])
                try:
                    self.R500 = float(self.table[self.cat]['R500'][0])*u.Mpc
                except:
                    self.R500 = 1.*u.Mpc
                    self.user_radius = False
                    self.log.log(logging.WARNING,'No R500 key found. setting R500='\
                                            +str(self.R500.value)+'Mpc to continue')
            else:
                self.R500 = 1.*u.Mpc
                self.z    = 0.1
                self.user_radius = False
                self.log.log(logging.WARNING,'No R500 key found. setting R500='\
                                    +str(self.R500.value)+'Mpc as filling to continue. Ignore this message if R500 != None')
                self.log.log(logging.WARNING,'No z key found. setting z='\
                                    +str(self.z)+'Mpc as filling to continue. Ignore this message if z != None')
        except:
            print('catalogue search FAILED')
            self.R500 = 1.*u.Mpc
            self.z    = 0.1
            self.user_radius = False
            self.log.log(logging.WARNING,'No R500 key found. setting R500='\
                                +str(self.R500.value)+'Mpc as filling to continue. Ignore this message if R500 != None')
            self.log.log(logging.WARNING,'No z key found. setting z='\
                                +str(self.z)+'Mpc as filling to continue. Ignore this message if z500 != None')

        if M500 is not None:
            self.M500 = float(M500)*1.e14*u.Msun
            self.M500_std = 0.*u.Msun
            self.log.log(logging.INFO,'Custom M500 mass set')
        if R500 is not None:
            self.R500 = float(R500)*u.Mpc
            self.log.log(logging.INFO,'Custom R500 radius set')
            self.user_radius=self.R500
        if z is not None:
            self.z = float(z)
            self.log.log(logging.INFO,'Custom redshift set')

        self.factor      = self.cosmology.kpc_proper_per_arcmin(self.z).to(u.Mpc/u.deg)
        self.radius_real = self.R500/self.factor
        self.freq        = (self.header['CRVAL3']*u.Hz).to(u.MHz)

    def set_image_characteristics(self, decrease_img_size):
        self.rmsnoise,self.imagenoise = u.Jy*self.get_noise(self.data*self.beam2pix)/self.beam2pix
        if decrease_img_size:
            self.decrease_fov(self.data)
            x = np.arange(0, np.shape(self.data.value)[1], step=1, dtype='float')
            y = np.arange(0, np.shape(self.data.value)[0], step=1, dtype='float')
            self.x_pix, self.y_pix     = np.meshgrid(x,y)

            self.image_mask, self.mask = utils.masking(self, self.mask)
            self.exponentialFit(self.data.value)
        else:
            pivot = ((np.sqrt(2.)/2.-0.5)*np.array(self.data.shape)).astype(np.int64)
            padX  = [pivot[0], pivot[0]]
            padY  = [pivot[1], pivot[1]]
            self.data_mcmc     = np.pad(self.data, [padY, padX], 'constant')
            self.fov_info_mcmc = [-pivot[0],self.data.shape[0]+pivot[0],
                                  -pivot[1],self.data.shape[1]+pivot[1]]
            self.fov_info      = [0,self.data.shape[0],0,self.data.shape[1]]
            self.margin = np.array(self.fov_info)-np.array(self.fov_info_mcmc)
            self.data = self.data[self.fov_info[0]:self.fov_info[1],
                             self.fov_info[2]:self.fov_info[3]]
            self.ra   =  self.ra[self.fov_info[2]:self.fov_info[3]]
            self.dec  = self.dec[self.fov_info[0]:self.fov_info[1]]

        self.noise_char = utils.noise_characterisation(self,self.data.value)
        self.pix2kpc    = self.pix_size*self.factor.to(u.kpc/u.deg)

    def get_beam_area(self):
        try:
            self.bmaj      = self.header['BMIN']*u.deg
            self.bmin      = self.header['BMAJ']*u.deg
            self.bpa       = self.header['BPA']*u.deg
        except KeyError:
            string    = str(self.header['HISTORY'])
            self.bmaj = self.findstring(string, 'BMAJ')*u.deg
            self.bmin = self.findstring(string, 'BMIN')*u.deg
            self.bpa  = self.findstring(string, 'BPA')*u.deg

        self.pix_size  = abs(self.header['CDELT2'])*u.deg
        beammaj        = self.bmaj/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
        beammin        = self.bmin/(2.*(2.*np.log(2.))**0.5) # Convert to sigma
        self.pix_area  = abs(self.header['CDELT1']*self.header['CDELT2'])*u.deg*u.deg
        self.beam_area = 2.*np.pi*1.0*beammaj*beammin
        self.beam2pix  = self.beam_area/self.pix_area

    def unpack_File(self):
        self.hdul = fits.open(self.path)
        try:
            data = self.hdul[0].data[0,0,:,:]
        except:
            data = self.hdul[0].data
        self.header = self.hdul[0].header
        data[np.isnan(data)]=0
        return data

    def findstring(self, string, key):
        string = string.split('\n')
        for i in range(len(string)):
            if string[i].find(key) != -1 and string[i].find('CLEAN') != -1:
                line = string[i]
        the_key = line.find(key)
        start = line[the_key:].find('=')+the_key+1
        while line[start]==' ':
            start+=1

        if line[start:].find(' ') == -1:
            return float(line[start:])
        end = line[start:].find(' ')+start
        return float(line[start:end])


    def get_noise(self, data, ampnoise=0.2):
        rmsnoise   = utils.findrms(data.value)
        #rmsnoise   = utils.get_rms(self.path)
        imagenoise = 0.#np.sqrt((ampnoise*data)**2+(rmsnoise*np.sqrt(1./self.beam2pix))**2)
        return rmsnoise, imagenoise

    def decrease_fov(self, data, width=2):
        ''' Function decreases image size based on first fit in exponentialFit.
        Slightly bigger image is used in MCMC. data is stored in self.data_mcmc'''
        self.cropped = False
        error        = False
        image_width = width*self.radius/self.pix_size
        test_fov = [int(self.centre_pix[1] - np.sqrt(2.01)*image_width),
                    int(self.centre_pix[1] + np.sqrt(2.01)*image_width),
                    int(self.centre_pix[0] - np.sqrt(2.01)*image_width),
                    int(self.centre_pix[0] + np.sqrt(2.01)*image_width)]
        for margin in test_fov:
            if margin < 0 or margin > np.array(self.data.shape).min():
                error = True
        if error:
            self.log.log(logging.ERROR,'{}: Decreasing FoV not possible. Halo is too big'.format(self.target))

            pivot = ((np.sqrt(2.)/2.-0.5)*np.array(data.shape)).astype(np.int64)
            padX  = [pivot[0], pivot[0]]
            padY  = [pivot[1], pivot[1]]
            self.data_mcmc     = np.pad(data, [padY, padX], 'constant')
            self.fov_info_mcmc = [-pivot[0],self.data.shape[0]+pivot[0],
                                  -pivot[1],self.data.shape[1]+pivot[1]]
            self.fov_info      = [0,self.data.shape[0],0,self.data.shape[1]]
        else:
            self.fov_info = [int(self.centre_pix[1] - image_width),
                             int(self.centre_pix[1] + image_width),
                             int(self.centre_pix[0] - image_width),
                             int(self.centre_pix[0] + image_width)]
            self.fov_info_mcmc = [int(self.centre_pix[1] - np.sqrt(2.01)*image_width),
                                  int(self.centre_pix[1] + np.sqrt(2.01)*image_width),
                                  int(self.centre_pix[0] - np.sqrt(2.01)*image_width),
                                  int(self.centre_pix[0] + np.sqrt(2.01)*image_width)]
            self.data_mcmc = data[self.fov_info_mcmc[0]:self.fov_info_mcmc[1],
                                  self.fov_info_mcmc[2]:self.fov_info_mcmc[3]]
            self.cropped = True

        self.margin = np.array(self.fov_info)-np.array(self.fov_info_mcmc)
        self.data = data[self.fov_info[0]:self.fov_info[1],
                         self.fov_info[2]:self.fov_info[3]]
        self.ra   =  self.ra[self.fov_info[2]:self.fov_info[3]]
        self.dec  = self.dec[self.fov_info[0]:self.fov_info[1]]
        #plt.imshow(self.data.value)
        #plt.show()

    def pix_to_world(self):
        w = wcs.WCS(self.header)
        centre_pix  = np.array([[self.centre_pix[0],self.centre_pix[1]]])
        world_coord = w.celestial.wcs_pix2world(centre_pix,0)
        if world_coord[0,0]<0.: world_coord[0,0] += 360
        if world_coord[0,1]<0.: world_coord[0,1] += 360

        self.centre_wcs = (np.array([world_coord[0,0],world_coord[0,1]])*u.deg)

        self.ra   = np.arange(0,len(self.x_pix))*self.pix_size
        self.dec  = np.arange(0,len(self.y_pix))*self.pix_size
        self.ra  -= self.ra[self.centre_pix[0]]-self.centre_wcs[0]
        self.dec -= self.dec[self.centre_pix[1]]-self.centre_wcs[1]

    def find_halo_centre(self, data, first):
        if first or self.original_image.shape == self.data.shape:
            w           = wcs.WCS(self.header)
            centre_wcs  = np.array([[self.loc.ra.deg,self.loc.dec.deg]])
            world_coord = w.celestial.wcs_world2pix(centre_wcs,0,ra_dec_order=True)
            return np.array([world_coord[0,0],world_coord[0,1]])
        else:
            return np.array((data.shape[1]/2.,data.shape[0]/2.),dtype=np.int64)


    def pre_mcmc_func(self, obj, *theta):
        I0, x0, y0, re = theta
        model = obj.circle_model((obj.x_pix,obj.y_pix), I0, x0, y0, re )
        if obj.mask:
            return model[obj.image_mask.ravel() == 0]
        else: return model

    def exponentialFit(self, data, first=False):
        plotdata = np.copy(data)
        plotdata[self.image_mask==1]=0
        max_flux   = np.max(plotdata)
        centre_pix = self.find_halo_centre(data, first)
        if not first: size = self.radius/(3.5*self.pix_size)
        else: size = data.shape[1]/4.
        bounds  = ([0.,0.,0.,0.,],
                  [np.inf,data.shape[0],
                          data.shape[1],
                          data.shape[1]/2.])
        if self.user_radius != False:
            size = (self.radius_real/2.)/self.pix_size

        image = data.ravel()
        if self.mask:
            image = data.ravel()[self.image_mask.ravel() == 0]

        popt, pcov = curve_fit(self.pre_mcmc_func,self,
                                image, p0=(max_flux,centre_pix[0],
                                centre_pix[1],size), bounds=bounds)

        if (self.user_radius != False and self.radius_real<(3.5*popt[3]*self.pix_size)):# or popt[3]>0.5*image.shape[0]:
            popt[3]=size
            print('size overwrite')

        #if first:
        self.radius = 3.5*popt[3]*self.pix_size
        self.centre_pix = np.array([popt[1],popt[2]], dtype=np.int64)
        self.I0 = popt[0]

    def circle_model(self, coords, I0, x0, y0, re):
        x,y = coords
        r   = np.sqrt((x-x0)**2+(y-y0)**2)
        Ir  =  I0 * np.exp(-(r/re))
        return Ir.ravel()

    def Close(self):
        self.hdul.close()
        self.log.log(logging.INFO,'closed Halo object {}'.format(self.target))
