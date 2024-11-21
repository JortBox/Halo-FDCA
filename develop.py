import halo_fdca as fdca
import sys, os


Halo = fdca.RadioHalo(
    'A2744', 
    'Example/Data_dir/A2744_JVLA.image.fits', 
    output_path='Example/Output_dir', 
    z=0.308, 
    #mask_path='Example/Data_dir/A2744halo.reg',
    decreased_fov=False
)



#fit = fdca.Fit(Halo, walkers=10, steps=100, model = ["circle", "rotated_ellipse"])
#fit.run()
#sys.exit()

# freeze parameters in image units
freeze = {
    "x0": 330.,
    "y0": 315.,
}

fit = fdca.Fit(Halo, walkers=10, steps=100)#, freeze_params=freeze)
fit.run()
fit.save()
fit.load()

chi2 = fit.results.get_chi2()
flux = fit.results.get_flux()
fit.results.plot()
