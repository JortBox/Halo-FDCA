#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: J.M. Boxelaar
Version: 08 June 2020
"""
# Built in module imports
import sys, os, logging
from datetime import datetime

# Scipy, astropy, emcee imports
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy import wcs
import astropy.units as u
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

from . import fdca_utils as utils
from . import fdca_logger

np.seterr(divide="ignore", invalid="ignore")

rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
Jydeg2 = u.Jy / (u.deg * u.deg)
mJyarcsec2 = u.mJy / (u.arcsec * u.arcsec)
uJyarcsec2 = 1.0e-3 * u.mJy / (u.arcsec * u.arcsec)


class RadioHalo(object):
    """
    -CLASS DESCRIPTION-
    This class initiates a RadioHalo object containing all image and physical
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
    """

    def __init__(
        self,
        object: str,
        path: str,
        decreased_fov: bool = False,
        mask_path = None,
        #mask: bool = False,
        logger=None,
        loc: SkyCoord=None,
        M500=None,
        R500=None,
        z=None,
        output_path: str = "",
        spectr_index: float = -1.2,
        rms: float = 0,
    ):
        if logger is None:
            fdca_logger.Logger('fdca', path = output_path)
            logger = fdca_logger.logger
            logger.info('Starting the FDCA pipeline...\n')
            self.logger = logger
        else:
            self.logger = logger
           
        self.cropped = False
        self.rmsnoise = rms  # manual noise level Jy/beam
        self.user_radius = R500

        if object[:4] == "MCXC":
            self.cat = "J/A+A/534/A109/mcxc"
        elif object[:4] == "PSZ2":
            self.cat = "J/A+A/594/A27/psz2"
        elif object[:3] == "WHL":
            self.cat = "J/MNRAS/436/275/table2"
        elif object[:5] == "Abell":
            self.cat = "VII/110A/table3"
        else:
            self.cat = None
            self.logger.warning(
                "Unknown what catalogue to use. If no costum values are given, filling values will be used",
            )

        self.target = str(object)
        self.path = path
        self.alpha = spectr_index
        self.name = self.target.replace("MCXC", "MCXC ")
        self.name = self.target.replace("PSZ2", "PSZ2 ")
        self.name = self.target.replace("Abell", "Abell ")
        self.name = self.target.replace("WHL", "")
        

        self.initiatePaths(mask_path, output_path)
        data, hader = self.load_data()
        self.original_image = np.copy(data)
        self.header = hader
        self.wcs = wcs.WCS(self.header)

        self.get_beam_area()
        self.get_object_location(loc)
        self.extract_object_info(M500, R500, z)
        
        if self.header["BUNIT"] == "JY/BEAM" or self.header["BUNIT"] == "Jy/beam":
            self.data = data * (u.Jy / self.beam2pix)
        else:
            self.logger.error("Possibly other units than jy/beam, CHECK HEADER UNITS!")
            sys.exit()
        
        x = np.arange(0, data.shape[1], step=1, dtype="float")
        y = np.arange(0, data.shape[0], step=1, dtype="float")
        self.x_pix, self.y_pix = np.meshgrid(x, y)

        self.fov_info = [0, data.shape[0], 0, data.shape[1]]
        self.image_mask = utils.masking(self, verbose=True)
        self.exponentialFit(data, first=True)  # Find centre of the image centre_pix

        self.pix_to_world()
        self.set_image_characteristics(decreased_fov)


    def initiatePaths(self, maskpath:str, outputpath: str):
        if outputpath.endswith("/"):
            self.basedir = outputpath[:-1]
        elif outputpath == "":
            self.basedir = os.getcwd()
        else:
            self.basedir = outputpath

        txt = self.path.split("/")
        self.file = txt[-1]
        self.dataPath = "/" + "/".join(txt[:-1]) + "/"

        self.plotPath = self.basedir + "/Plots/"
        self.modelPath = self.basedir + "/"

        if not os.path.isdir(self.modelPath):
            self.logger.info("Creating modelling directory")
            os.makedirs(self.modelPath)
        if not os.path.isdir(self.plotPath):
            self.logger.info("Creating plotting directory")
            os.makedirs(self.plotPath)

        if maskpath is None:
            self.maskPath = f"{self.basedir}/{self.target}.reg"
        else:
            self.maskPath = maskpath
            
        if os.path.isfile(self.maskPath):
            self.mask = True
        else:
            self.mask = False
            self.logger.warning(f"No mask found at {self.maskPath}. Masking will not be applied.")

    def get_object_location(self, loc):
        if loc is not None:
            if type(loc) == SkyCoord:
                self.loc = loc
            else:
                self.logger.error("Coordinate given is not a SkyCoord object. Please provide a valid SkyCoord object")
                sys.exit()
        else:
            from astroquery.ipac.ned import Ned
            try:
                self.logger.warning(f"No manual coordinate given, searching for {self.name} in NED.")
                table = Ned.query_object(self.name)
                self.loc = SkyCoord(table["RA"][0], table["DEC"][0], unit=u.deg)
            except:
                self.logger.error(f"{self.name} not found by NED. Assuming image centre.")
                cent_pix = np.asarray(self.original_image.shape, dtype=np.float64)//2.
                self.loc = wcs.utils.pixel_to_skycoord(cent_pix[0], cent_pix[1], self.wcs, origin=1)


    def extract_object_info(self, M500, R500, z):
        """Written for MCXC catalogue. Information is gathered from there. If custom
        parameters are given, these will be used. if nothing is found, filling
        values are set. This is only a problem if you try to calculate radio power."""
        try:
            self.table = Vizier.query_object(self.name, catalog=self.cat)

            if self.target[:4] == "MCXC":
                self.M500 = float(self.table[self.cat]["M500"][0]) * 1.0e14 * u.Msun
                self.L500 = float(self.table[self.cat]["L500"][0]) * 1.0e37 * u.Watt
                self.R500 = float(self.table[self.cat]["R500"][0]) * u.Mpc
                self.z = float(self.table[self.cat]["z"][0])
                self.M500_std = 0.0 * u.Msun

            elif self.target[:3] == "WHL":
                self.z = float(self.table[self.cat]["z"][0])
                self.R500 = 1.0 * u.Mpc
                self.M500 = 3.0e14 * u.Msun
                self.user_radius = False
                # self.logger.log(logging.WARNING,'No R500 key found. setting R500='\
                #                        +str(self.R500.value)+'Mpc to continue')

            elif self.target[:5] == "Abell":
                try:
                    self.z = float(self.table[self.cat]["z"][0])
                except:
                    self.z = 0.1
                    # self.logger.log(logging.WARNING,'No valid z key found. setting z='\
                    #                    +str(self.z)+' as filling to continue. Ignore this message if -z != None')
                self.R500 = 1.0 * u.Mpc
                self.user_radius = False
                # self.logger.log(logging.WARNING,'No R500 key found. setting R500='\
                #                        +str(self.R500.value)+'Mpc to continue')

            elif self.target[:4] == "PSZ2":
                self.M500 = float(self.table[self.cat]["MSZ"][0]) * 1.0e14 * u.Msun
                self.M500_std = (
                    np.max(
                        [
                            float(self.table[self.cat]["E_MSZ"][0]),
                            float(self.table[self.cat]["e_MSZ"][0]),
                        ]
                    )
                    * 1.0e14
                    * u.Msun
                )
                self.z = float(self.table[self.cat]["z"][0])
                try:
                    self.R500 = float(self.table[self.cat]["R500"][0]) * u.Mpc
                except:
                    self.R500 = 1.0 * u.Mpc
                    self.user_radius = False
            else:
                self.R500 = 1.0 * u.Mpc
                self.z = 0.1
                self.user_radius = False

        except:
            self.logger.warning(f"Catalogue {self.cat} found for {self.name}. Filling values will be used.")
            self.R500 = 1.0 * u.Mpc
            self.z = 0.1
            self.user_radius = False

        if M500 is not None:
            self.M500 = float(M500) * 1.0e14 * u.Msun
            self.M500_std = 0.0 * u.Msun
            self.logger.debug("Custom M500 mass set: " + str(self.M500))
        if R500 is not None:
            self.R500 = float(R500) * u.Mpc
            self.logger.debug("Custom R500 radius set: "+str(self.R500))
            self.user_radius = self.R500
        if z is not None:
            self.z = float(z)
            self.logger.debug("Custom redshift set: " + str(self.z))

        cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        self.factor = cosmology.kpc_proper_per_arcmin(self.z).to(u.Mpc / u.deg)
        self.radius_real = self.R500 / self.factor
        self.freq = (self.header["CRVAL3"] * u.Hz).to(u.MHz)

    def set_image_characteristics(self, decrease_img_size: bool):
        if self.rmsnoise == 0.0:
            self.rmsnoise, self.imagenoise = (
                self.get_noise(self.data * self.beam2pix) / self.beam2pix
            )
        else:
            self.rmsnoise = (self.rmsnoise / self.beam2pix)
            self.imagenoise = 0.0

        self.rmsnoise = (u.Jy * self.rmsnoise / self.pix_area).to(u.Jy / self.beam)
        
        self.logger.debug(f"rms noise {self.rmsnoise:.3e} ")
        self.logger.debug(f"rms noise {self.rmsnoise.to(u.Jy/u.arcsec**2):.3e}")

        if decrease_img_size:
            old_shape = self.data.shape
            self.decrease_fov(self.data)
            x = np.arange(0, np.shape(self.data.value)[1], step=1, dtype="float")
            y = np.arange(0, np.shape(self.data.value)[0], step=1, dtype="float")
            self.x_pix, self.y_pix = np.meshgrid(x, y)
            
            self.logger.info("Decreased image shape from " +str(old_shape)+ " to "+str(self.data.shape))

            self.image_mask = utils.masking(self)
            self.exponentialFit(self.data.value)
        else:
            self.data_mcmc, self.fov_info_mcmc = utils.pad_image(self.data)
            
            self.fov_info = [0, self.data.shape[0], 0, self.data.shape[1]]
            self.margin = np.array(self.fov_info) - np.array(self.fov_info_mcmc)
            self.data = self.data[
                self.fov_info[0] : self.fov_info[1], self.fov_info[2] : self.fov_info[3]
            ]
            self.ra = self.ra[self.fov_info[2] : self.fov_info[3]]
            self.dec = self.dec[self.fov_info[0] : self.fov_info[1]]

        self.logger.info("MCMC image shape: "+str(self.data_mcmc.shape))
        #self.noise_char = utils.noise_characterisation(self, self.data.value)
        self.pix2kpc = self.pix_size * self.factor.to(u.kpc / u.deg)
        
        
    def get_beam_area(self):
        try:
            self.bmaj = self.header["BMAJ"] * u.deg
            self.bmin = self.header["BMIN"] * u.deg
            self.bpa = self.header["BPA"] * u.deg
        except KeyError:
            string = str(self.header["HISTORY"])
            self.bmaj = self.findstring(string, "BMAJ") * u.deg
            self.bmin = self.findstring(string, "BMIN") * u.deg
            self.bpa = self.findstring(string, "BPA") * u.deg

        self.pix_size = abs(self.header["CDELT2"]) * u.deg
        beammaj = self.bmaj / (2.0 * (2.0 * np.log(2.0)) ** 0.5)  # Convert to sigma
        beammin = self.bmin / (2.0 * (2.0 * np.log(2.0)) ** 0.5)  # Convert to sigma
        
        self.pix_area = (abs(self.header["CDELT1"] * self.header["CDELT2"]) * u.deg * u.deg)
        self.beam_area = 2.0 * np.pi * 1.0 * beammaj * beammin
        self.beam2pix = self.beam_area / self.pix_area
        
        
        self.beam = u.def_unit("beam", self.beam_area)
        self.pixScale = u.def_unit("pixel", self.pix_size) # type: ignore
        self.pix = u.def_unit("pixel2", self.pix_area) # type: ignore

    def load_data(self):
        hdul = fits.open(self.path)
        try:
            data = hdul[0].data[0, 0, :, :]
        except:
            data = hdul[0].data
        header = hdul[0].header
        hdul.close()
        data[np.isnan(data)] = 0
        return data, header

    def findstring(self, string, key):
        string = string.split("\n")
        for i in range(len(string)):
            if string[i].find(key) != -1 and string[i].find("CLEAN") != -1:
                line = string[i]
        the_key = line.find(key)
        start = line[the_key:].find("=") + the_key + 1
        while line[start] == " ":
            start += 1

        if line[start:].find(" ") == -1:
            return float(line[start:])
        end = line[start:].find(" ") + start
        return float(line[start:end])

    def get_noise(self, data, ampnoise=0.2):
        rmsnoise = utils.findrms(data.value)
        # rmsnoise   = utils.get_rms(self.path)
        imagenoise = (
            0.0  # np.sqrt((ampnoise*data)**2+(rmsnoise*np.sqrt(1./self.beam2pix))**2)
        )
        return rmsnoise, imagenoise

    def decrease_fov(self, data, width=2):
        """Function decreases image size based on first fit in exponentialFit.
        Slightly bigger image is used in MCMC. data is stored in self.data_mcmc"""
        
        error = False
        image_width = width * self.radius / self.pix_size
        test_fov = [
            int(self.centre_pix[1] - np.sqrt(2.01) * image_width),
            int(self.centre_pix[1] + np.sqrt(2.01) * image_width),
            int(self.centre_pix[0] - np.sqrt(2.01) * image_width),
            int(self.centre_pix[0] + np.sqrt(2.01) * image_width),
        ]
        for margin in test_fov:
            if margin < 0 or margin > np.array(self.data.shape).min():
                error = True
        if error:
            self.logger.warning("{}: Decreasing FoV not possible. Halo is too big. Pad image".format(self.target))

            pivot = ((np.sqrt(2.0) / 2.0 - 0.5) * np.array(data.shape)).astype(np.int64)
            padX = [pivot[0], pivot[0]]
            padY = [pivot[1], pivot[1]]
            self.data_mcmc = np.pad(data, [padY, padX], "constant")
            self.fov_info_mcmc = [
                -pivot[0],
                self.data.shape[0] + pivot[0],
                -pivot[1],
                self.data.shape[1] + pivot[1],
            ]
            self.fov_info = [0, self.data.shape[0], 0, self.data.shape[1]]
        else:
            self.fov_info = [
                int(self.centre_pix[1] - image_width),
                int(self.centre_pix[1] + image_width),
                int(self.centre_pix[0] - image_width),
                int(self.centre_pix[0] + image_width),
            ]
            self.fov_info_mcmc = [
                int(self.centre_pix[1] - np.sqrt(2.01) * image_width),
                int(self.centre_pix[1] + np.sqrt(2.01) * image_width),
                int(self.centre_pix[0] - np.sqrt(2.01) * image_width),
                int(self.centre_pix[0] + np.sqrt(2.01) * image_width),
            ]
            self.data_mcmc = data[
                self.fov_info_mcmc[0] : self.fov_info_mcmc[1],
                self.fov_info_mcmc[2] : self.fov_info_mcmc[3],
            ]
            self.cropped = True

        self.margin = np.array(self.fov_info) - np.array(self.fov_info_mcmc)
        self.data = data[
            self.fov_info[0] : self.fov_info[1], self.fov_info[2] : self.fov_info[3]
        ]
        self.ra = self.ra[
            self.fov_info[0] : self.fov_info[1], self.fov_info[2] : self.fov_info[3]
        ]
        self.dec = self.dec[
            self.fov_info[0] : self.fov_info[1], self.fov_info[2] : self.fov_info[3]
        ]
        # plt.imshow(self.data.value)
        # plt.show()

    def pix_to_world(self):
        self.centre_wcs = wcs.utils.pixel_to_skycoord(self.centre_pix[0], self.centre_pix[1], self.wcs, origin=1)

        flat_x = self.x_pix.flatten()
        flat_y = self.y_pix.flatten()

        coords = wcs.utils.pixel_to_skycoord(flat_x, flat_y, self.wcs, origin=1)
        self.ra = coords.ra.deg.reshape(self.x_pix.shape)*u.deg
        self.dec = coords.dec.deg.reshape(self.y_pix.shape)*u.deg

    def get_initial_halo_params(self):
        return utils.get_initial_guess(self)[0]

    def pre_mcmc_func(self, obj, *theta):
        I0, x0, y0, re = theta
        model = obj.circle_model((obj.x_pix, obj.y_pix), I0, x0, y0, re)
        if obj.mask:
            return model[obj.image_mask.ravel()]
        else:
            return model

    def exponentialFit(self, data, first=False):
        plotdata = np.copy(data)
        plotdata[~self.image_mask] = 0
        if first:
            centre_pix = np.asarray(wcs.utils.skycoord_to_pixel(self.loc, self.wcs, origin=1))
            size = data.shape[1] / 4.0
            max_flux = np.max(plotdata)
        else:
            if self.cropped:
                centre_pix = [self.centre_pix[0] - self.fov_info[2], self.centre_pix[1] - self.fov_info[0]]
            else:
                centre_pix = self.centre_pix
            size = self.radius / (3.5 * self.pix_size)
            max_flux = self.I0
            
        bounds = (
            [0.0,0.0,0.0,0.0],
            [np.inf, data.shape[0], data.shape[1], data.shape[1] / 2.0],
        )
        if self.user_radius != False:
            size = (self.radius_real / 2.0) / self.pix_size
            
        image = data.ravel()#/max_flux # Normalise image for better fitting results
        if self.mask:
            image = data.ravel()[self.image_mask.ravel()]

        popt, pcov = curve_fit(
            self.pre_mcmc_func,
            self,
            image,
            p0=(max_flux, centre_pix[0], centre_pix[1], size),
            bounds=bounds,
        )

        if self.user_radius != False and self.radius_real < (
            3.5 * popt[3] * self.pix_size
        ):
            popt[3] = size
            print("size overwrite")

        # if first:
        self.radius = 3.5 * popt[3] * self.pix_size
        self.centre_pix = np.round(np.array([popt[1], popt[2]])).astype(np.int64)
        self.I0 = popt[0] #* max_flux # scale max flux back to original

    def circle_model(self, coords, I0, x0, y0, re):
        x, y = coords
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        Ir = I0 * np.exp(-(r / re))
        return Ir.ravel()

    def Close(self):
        #self.hdul.close()
        self.logger.info("closed Halo object {}".format(self.target))
