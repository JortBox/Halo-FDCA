import sys, os
os.chdir(__file__.replace('main.py', ''))

import halo_fdca as fdca

def regular_run():
    Halo = fdca.RadioHalo(
        'A2744', 
        'Example/Data_dir/A2744_JVLA.image.fits', 
        output_path='Example/Output_dir', 
        z=0.308, 
        mask_path='Example/Data_dir/A2744halo.reg',
        decreased_fov=True
    )
    
    

    # freeze parameters in image units
    freeze = {
        "x0": 330.,
        "y0": 315.,
    }

    fit = fdca.Fit(Halo, walkers=10, steps=100, model='rotated_ellipse')#, freeze_params=freeze)
    fit.run()
    fit.save()
    fit.load()
    
    chi2 = fit.results.get_chi2()
    flux = fit.results.get_flux()
    fit.results.plot()
    
    

def load_from_file():
    fit = fdca.load("Example/Output_dir/A2744_JVLA.image_mcmc_samples_circle_mask.json")

    chi2 = fit.results.get_chi2()
    flux = fit.results.get_flux()
    fit.results.plot()


if __name__ == "__main__":
    regular_run()
    #load_from_file()