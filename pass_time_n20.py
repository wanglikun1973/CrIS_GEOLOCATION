
"""
    A simple python script to check CrIS geolocation accuracy 
       using VIIRS I5 data
    Author: Likun.Wang@noaa.gov
    Date  : Oct 23 2018   
"""

import numpy as np
from time import ctime
import multiprocessing
import os
import pdb

from jpss import read_cris_sdr, read_cris_geo, apodize, \
	cris_sensor_info, get_viirs_rsr, rsr_on_grid, compute_band_rad, rad2temp
	
from collocation import cris_viirs_perturb

satname = 'jpss1'
sat = 'N20'
band = 'I5'

res = {'I5': 375./824000., 'M15': 375./824000.}  

## step angle 
step_angle = res[band]
anum = 17
bnum = 15
astart = -1.0*int(anum/2.0)*step_angle
bstart = -1.0*int(bnum/2.0)*step_angle
alpha = np.linspace(astart, -astart, num = anum)
beta  = np.linspace(bstart, -bstart, num = bnum)

## output Dir
dataDir = './output/'

## VIIRS RSR 
para = cris_sensor_info()
wvLW, wvMw, wvSW = para['wvNorm']
srf = get_viirs_rsr(band, sat=sat)
rsr = rsr_on_grid(srf['srf_w'], srf['srf_v'], wvLW)

### prepare for parallel processing
def work_one(args): 

    cris_sdr_files, cris_geo_files, viirs_sdr_files, viirs_geo_files = args

    if type(cris_sdr_files) is not list: 
        cris_sdr_files = [cris_sdr_files]
        cris_geo_files = [cris_geo_files] 

    print("================>", ctime())

    cris_lon, cris_lat, cris_satAzimuth, cris_satRange, cris_satZenith = read_cris_geo(cris_geo_files)
    cris_realLW, cris_realMW, cris_realSW, cris_sdrQa, cris_geoQa, cris_dayFlag = read_cris_sdr(cris_sdr_files) 

    print('cris_sdr', cris_sdr_files)
    print('cris_geo', cris_geo_files)
    print('viirs_sdr', viirs_sdr_files)
    print('viirs_geo', viirs_geo_files)
        
    outfile = dataDir + os.path.basename(cris_sdr_files[0])
    if os.path.isfile(outfile+'.npz'):
        print('file already there ... continue ...')
        return None
    
    data= cris_viirs_perturb (cris_geo_files, cris_sdr_files, viirs_geo_files, viirs_sdr_files, 
            alpha=alpha, beta=beta)
    if data is None: return None 
	
    cris_spcLW = apodize(cris_realLW)
    cris_c_rad = compute_band_rad(wvLW, cris_spcLW, rsr)
    cris_c_bt = rad2temp(srf['cwv'], cris_c_rad)
    cris_v_bt = data['cris_v_bt']
    cris_v_rad = data['cris_v_rad']
	
    np.savez(outfile, cris_lon=cris_lon, cris_lat = cris_lat, cris_satAzimuth=cris_satAzimuth, \
        cris_satRange = cris_satRange, cris_satZenith = cris_satZenith, cris_sdrQa = cris_sdrQa, \
        cris_c_bt = cris_c_bt, cris_v_bt=cris_v_bt, cris_v_rad=cris_v_rad, alpha=alpha, beta=beta, \
        cris_sdr_files = cris_sdr_files, cris_geo_files = cris_geo_files, \
        viirs_sdr_files = viirs_sdr_files, viirs_geo_files=viirs_geo_files)
    print('processing suceed ... done ...')
    return True    
    

def main():

    ### CrIS and VIIRS match file lists
    filelist = 'match_cris_viirs_files.npz'
    cris_sdr_lst  = np.load(filelist)['cris_sdr_lst']
    cris_geo_lst  = np.load(filelist)['cris_geo_lst']
    viirs_sdr_lst = np.load(filelist)['viirs_sdr_lst']
    viirs_geo_lst = np.load(filelist)['viirs_geo_lst']
    args = zip(cris_sdr_lst, cris_geo_lst, viirs_sdr_lst, viirs_geo_lst)
    
    parallelFlag = False 
    
    pdb.set_trace()
    
    if parallelFlag: 
        #######################################################################
        ### for parallel processing
        num = 4  # set to the number of workers you want (it defaults to the cpu count of your machine)
        pool = multiprocessing.Pool(processes=num)

        for y in pool.imap_unordered(work_one, args):
            print(y)
        pool.close()
        pool.join()
    else: 
        #######################################################################
        ### for  processing
        for arg in args[0:1]: work_one(arg)

    
if __name__== "__main__":
  main()    

	
	

