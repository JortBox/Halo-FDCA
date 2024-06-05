import halo_fdca as fdca
import sys, logging, os

from datetime import datetime

def init_logging(filename, path):
    path = path_out
    if path[-1]=='/': path = path[:-1]
    now = str(datetime.now())[:19]
    if not os.path.exists(path+'/log/'):
        os.makedirs(path+'/log/')

    d = {
            'version': 1,
            'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(name)-12s %(processName)-2s %(levelname)-8s %(message)s'
            }
            },
            'handlers': {
            'file': {
                'class': 'logging.FileHandler',
                'filename': path+'/log/'+filename+'_'+now.replace(' ','_')+'.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            },
            'root': {
            'level': 'INFO',
            'handlers': ['file'] #,'console'
            },
        }

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.config.dictConfig(d)
    return logging







target = 'A1033'
path_out = 'Example/Output_dir'

logging = init_logging(target, path_out)
logger  = logging.getLogger(target)


Halo_low = fdca.RadioHalo(
    target, 
    'Example/data_dir_test/A1033-LBA-sub-MFS-image.fits', 
    output_path=path_out, 
    z=0.122, 
    mask_path='Example/data_dir_test/A1033halo_lba.reg',
    logger=logger,
)

Halo_high = fdca.RadioHalo(
    target, 
    'Example/data_dir_test/A1033-HBA-sub-MFS-T50kpc.fits', 
    output_path=path_out, 
    z=0.122, 
    mask_path='Example/data_dir_test/A1033halo_hba.reg',
    decreased_fov=True,
    logger=logger,
)

multiFit = fdca.MultiWavelengthFitting([Halo_low, Halo_high])
#multiFit.run(save=True)

results = fdca.MultiWavaelenghtProcessing(multiFit)

for processing in results.results:
    processing.plot()
    processing.get_chi2_value()
    processing.get_flux()# error is one sigma (68%).
    processing.get_power()

    print(processing)
