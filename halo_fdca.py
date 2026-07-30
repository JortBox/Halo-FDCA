import halo_fdca as fdca

CONFIG = {
        "cluster" : "A2069",
        "z" : 0.115,
        "M500" : 5.31, #in 10^14 Msol
        "model" : "circle"
    }

cluster = CONFIG["cluster"]
redshift = CONFIG["z"]
M500 = CONFIG["M500"]
model = CONFIG["model"]

path = f'/home/alessandro_coppola/strw_leiden/images/{cluster}/sub/{cluster}_deep_diff_100kpc-MFS-image.fits'
mask_path=f'/home/alessandro_coppola/strw_leiden/images/{cluster}/mask/mask_T100.reg'

Halo = fdca.RadioHalo(f'{cluster}', path=path, mask_path=mask_path ,decreased_fov=True, M500=M500, z=redshift)

fit = fdca.Fit(Halo, model="circle")
fit.run()
fit.save(f'/home/alessandro_coppola/strw_leiden/images/{cluster}/halo-fdca/{cluster}_100kpc.json')
fit = fdca.load(f'/home/alessandro_coppola/strw_leiden/images/{cluster}/halo-fdca/{cluster}_100kpc.json')
fit.results.plot()

chi2 = fit.results.get_chi2()
flux, flux_uncertainty = fit.results.get_flux() 
power, power_uncertainty = fit.results.get_power()

samples = fit.get_samples()
parameter_names = fit.get_param_names()