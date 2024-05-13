import halo_fdca

Halo = halo_fdca.RadioHalo(
    'Abell2744', 
    'Example/Data_dir/A2744_JVLA.image.fits', 
    mask_path='Example/Data_dir/A2744halo.reg',
    loc='00 14 20.03 -30 23 17.8', # Has to be fixed
    z=0.308, 
    output_path='Example/Output_dir'
)  

halo_fdca.Fitting(Halo).run()
result = halo_fdca.Processing(Halo)

result.plot()
result.get_chi2_value()
result.get_flux()# error is one sigma (68%).
result.get_power()

Halo.Close()
