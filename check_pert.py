
"""
    A simple python script to analyze CrIS and viirs collocation results
    Author: Likun.Wang@noaa.gov (ESSIC/UMD)
    Date  : Oct 23 2018   
"""


import numpy as np
import glob

from scipy.interpolate import interp2d
from jpss import get_viirs_rsr, rad2temp_wl

import matplotlib.pyplot as plt

sat = 'N20'
band = 'I5'

def analysis_geo_lw (filelist, band='I5', sat='N20'):

    cris_v_rad = np.concatenate([np.load(f)['cris_v_rad'] for f in filelist], axis=2)
    cris_c_bt = np.concatenate([np.load(f)['cris_c_bt'] for f in filelist])

    if np.isnan(cris_v_rad).any(): return None, None

    alpha = np.load(filelist[0])['alpha']
    beta = np.load(filelist[0])['beta']


    srf_wl = get_viirs_rsr(band, grid = 'wavelength', sat='N20')
    cris_v_bt = rad2temp_wl(srf_wl['cwl'], cris_v_rad)

    bsize, asize, nScan, nFor, nFov = cris_v_bt.shape

    bt_dif_lw = np.array([np.std(cris_v_bt[m, n, :, iFOR, :].ravel() - cris_c_bt[:, iFOR, :].ravel())for  m, n, iFOR in np.ndindex(bsize, asize, nFor)]).reshape(bsize, asize, nFor)

    betaMin  = np.zeros(nFor)
    alphaMin = np.zeros(nFor)

    alphaZ = np.linspace(alpha.min(), alpha.max(), alpha.size*40)
    betaZ  = np.linspace(beta.min(), beta.max(), beta.size*40)

    for i in range(0, nFor): 

        z = bt_dif_lw[:, :, i]	
        f = interp2d(alpha, beta, z, kind='quintic')
        zi2 = f(alphaZ, betaZ)
        
        index = np.argmin(zi2)
        idb, ida = np.unravel_index(index, zi2.shape)
        betaMin[i], alphaMin[i] = (betaZ[idb]*1e6, alphaZ[ida]*1e6)

    return 	alphaMin, betaMin

################################################################
### Analyze the data 
Dir = './output/'
filelist = np.sort(glob.glob(Dir+'*.h5.npz'))
alphaM_i5, betaM_i5 = analysis_geo_lw(filelist, band='I5', sat='N20')
np.savez('cris_geo_alpha_beta', alphaM_i5=alphaM_i5, betaM_i5=betaM_i5)

################################################################
### Make plot 
plt.style.use('seaborn-talk')
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, 31), alphaM_i5, 'b-*', label='inTrack')
ax.plot(range(1, 31), betaM_i5, 'r-*', label=' xTrack')
yrange = [-3000,3000]		
ax.set_ylim(yrange)
ax.grid(True)
ax.set_ylabel('Angle   ['+r'$\mu$'+ 'rad]')
ax.set_xlabel('FOR Scan Position')
ax.set_title('Geolocation Accuracy relative to VIIRS')
ax.legend(loc='best')
 
ax2 = ax.twinx()   # mirror them
ax2.set_ylim(np.array(yrange)/16808. * 100.)
ax2.set_ylabel('Percentage of FOV Size [%]')
## add specification
ax2.axhline(y=10.7, c ='m', ls='--') 
ax2.axhline(y=-10.7, c ='m', ls='--') 
plt.show()

