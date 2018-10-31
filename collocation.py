# -*- coding: utf-8 -*-
# collocation.py

# Copyright (c) 2015-2018, Likun Wang (Likun.Wang@noaa.gov)
# Copyright (c) 2015-2018, Earth System Science Interdisciplinary 
#           Center/Univ. of Maryland 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

""" Accurate and fast collocation CrIS and VIIRS two instrument.

Check this paper if you need more information: 
Wang, L.; Tremblay, D.; Zhang, B.; Han, Y.	Fast and Accurate 
Collocation of the Visible Infrared Imaging Radiometer Suite Measurements 
with Cross-Track Infrared Sounder. Remote Sens. 2016, 8, 76.

:Author:
  `Likun Wang <Likun.Wang@noaa.gov>`_

:Organization:
  Earth System Science Interdisciplinary Center/Univ. of Maryland 

:Version: 2018.9.5

"""


import numpy as np
import multiprocessing as mp
import time

from jpss import read_cris_sdr, read_cris_geo, read_viirs_sdr, read_viirs_geo
import geolocation as geo
import pdb


def collocate_cris_viirs_tuple((cris_geo_files, cris_sdr_files, \
						 viirs_geo_files, viirs_sdr_files)):
							
	data=collocate_cris_viirs(cris_geo_files, cris_sdr_files, viirs_geo_files, viirs_sdr_files)
	return data

def collocate_cris_viirs(cris_geo_files, cris_sdr_files, \
						 viirs_geo_files, viirs_sdr_files): 
	

	# read VIIRS data 
	viirs_lon, viirs_lat, viirs_satAzimuth, viirs_satRange, viirs_satZenith = read_viirs_geo(viirs_geo_files)
	viirs_bt, viirs_rad, viirs_sdrQa =read_viirs_sdr(viirs_sdr_files)
	
	# read CrIS data 
	cris_lon, cris_lat, cris_satAzimuth, cris_satRange, cris_satZenith = read_cris_geo(cris_geo_files)
	
	#cris_realLW, cris_realMW, cris_realSW, \
    #   cris_sdrQa, cris_geoQa, cris_dayFlag = read_cris_sdr(cris_sdr_files, sdrFlag=True)

	# compute CrIS POS, LOS, SAT vector 
	cris_pos, cris_los, cris_sat = geo.GEO2ECEF(cris_lon, cris_lat, cris_satAzimuth, cris_satRange, cris_satZenith)	
	
		
	# compute viirs POS  vector 
	viirs_pos, viirs_los, viirs_sat = geo.GEO2ECEF(viirs_lon, viirs_lat, viirs_satAzimuth, viirs_satRange, viirs_satZenith)	
	
	# cris_los is pointing from pixel to satellite, we need to
	#   change from satellite to pixel
	cris_los = -1.0*cris_los

	# using Kd-tree to find the closest pixel of VIIRS for each CrIS FOV
	dy, dx = geo.match_cris_viirs(cris_los, cris_pos, viirs_pos, viirs_sdrQa)
	
	if dy is None: return None 
	
	# viirs averaged radiances
 
	cris_v_rad = np.asarray([np.mean(viirs_rad[dy[k, j, i], dx[k, j, i]]) if len(dx[k, j, i]) > 0 else 0.0  for k, j, i in np.ndindex(cris_lat.shape) ])
	cris_v_std = np.asarray([np.std(viirs_rad[dy[k, j, i], dx[k, j, i]]) if len(dx[k, j, i]) > 0 else 0.0  for k, j, i in np.ndindex(cris_lat.shape) ])
	
	cris_v_bt  = np.asarray([np.mean(viirs_bt[dy[k, j, i],  dx[k, j, i]]) if len(dx[k, j, i]) > 0 else 0.0 for k, j, i in np.ndindex(cris_lat.shape) ])
	
	cris_v_rad = cris_v_rad.reshape(cris_lat.shape)
	cris_v_bt  = cris_v_bt.reshape(cris_lat.shape)
	cris_v_std = cris_v_std.reshape(cris_lat.shape)
		
	# collocation data field 
	field = dict()
	field['cris_v_rad']  = cris_v_rad
	field['cris_v_std']  = cris_v_std
	field['cris_v_bt']  = cris_v_bt
	field['cris_lat']    = cris_lat
	field['cris_lon']    = cris_lon
	#field['dx']  = dx
	#field['dy']  = dy
	
	field['cris_sdr_files']    =   cris_sdr_files
	field['cris_geo_files']    =   cris_geo_files
	field['viirs_sdr_files']   =   viirs_sdr_files
	field['viirs_geo_files']   =   viirs_geo_files
	
	return field
	
def cris_viirs_perturb (cris_geo_files, cris_sdr_files, viirs_geo_files, viirs_sdr_files, alpha, beta, pmw = None, psw = None): 
	

	# read VIIRS data 
	viirs_lon, viirs_lat, viirs_satAzimuth, viirs_satRange, viirs_satZenith = read_viirs_geo(viirs_geo_files)
	viirs_bt, viirs_rad, viirs_sdrQa = read_viirs_sdr(viirs_sdr_files)
	viirs_MidTime, viirs_PosECR, viirs_VelECR, viirs_Att = read_viirs_geo(viirs_geo_files, ephemeris=True)
	

	# read CrIS data 
	cris_lon, cris_lat, cris_satAzimuth, cris_satRange, cris_satZenith = read_cris_geo(cris_geo_files)
	cris_FORTime, cris_MidTime, cris_SCPosition, cris_SCVelocity, cris_SCAttitude = read_cris_geo(cris_geo_files, ephemeris=True)
	nScan, nFor, nFov = cris_lon.shape
	
	
	#cris_realLW, cris_realMW, cris_realSW, \
    #   cris_sdrQa, cris_geoQa, cris_dayFlag = read_cris_sdr(cris_sdr_files, sdrFlag=True)

	# compute CrIS POS, LOS, SAT vector 
	cris_pos, cris_los, cris_sat = geo.GEO2ECEF(cris_lon, cris_lat, cris_satAzimuth, cris_satRange, cris_satZenith)	
	
	# compute viirs POS  vector 
	viirs_pos, viirs_los, viirs_sat = geo.GEO2ECEF(viirs_lon, viirs_lat, viirs_satAzimuth, viirs_satRange, viirs_satZenith)	
	
	# cris_los is pointing from pixel to satellite, we need to
	#   change from satellite to pixel
	cris_los = -1.0*cris_los
	
	# compute perturbation LOS
	
	start_time = time.time()
	
	p_out, v_out, att_out = geo.interpolate_sat_vector(viirs_PosECR, viirs_VelECR, viirs_Att, viirs_MidTime, cris_FORTime.ravel())
	att_out = np.deg2rad(att_out/3600.00)
	
	Mat_ECItoOrb = geo.buildECIOrbFrame(cris_FORTime.ravel(), p_out, v_out)
	Mat_OrbtoECI = np.transpose(Mat_ECItoOrb, [0, 2, 1])
	Mat_OrbtoECI = np.broadcast_to(Mat_OrbtoECI, (nFov, nScan*nFor, 3, 3))
	Mat_OrbtoECI = np.transpose(Mat_OrbtoECI, (1, 0, 2, 3))
	Mat_OrbtoECI = Mat_OrbtoECI.reshape(nFov*nScan*nFor, 3, 3)
	
	
	Mat_SPCtoOrb = geo.orb2sc(att_out[:, 0], att_out[:, 1], att_out[:, 2])
	Mat_SPCtoOrb = np.broadcast_to(Mat_SPCtoOrb, (nFov, nScan*nFor, 3, 3))
	Mat_SPCtoOrb = np.transpose(Mat_SPCtoOrb, (1, 0, 2, 3))
	Mat_SPCtoOrb = Mat_SPCtoOrb.reshape(nFov*nScan*nFor, 3, 3)
		
	cris_losECR = geo.normalize_vector(cris_los)
	cris_losECR = cris_losECR.reshape(-1, 3)
	
	cris_FOVTime = np.broadcast_to(cris_FORTime, (nFov, nScan, nFor))
	cris_FOVTime = np.transpose(cris_FOVTime, (1, 2, 0))
	
	cris_losECI, temp = geo.ECR2ECI_NOVAS(cris_FOVTime.ravel(), cris_losECR, 0.0)
	cris_losOrb = geo.matrix_vector_product(Mat_OrbtoECI, cris_losECI)
	cris_losSPC = geo.matrix_vector_product(Mat_SPCtoOrb, cris_losOrb)
	
	asize = alpha.size
	bsize = beta.size
	
	alpha0, beta0 = geo.compute_alpha_beta(cris_losSPC)
	alphav, betav = np.meshgrid(alpha, beta)
	
	alpha0 = np.broadcast_to(alpha0, (bsize, asize, nScan*nFor*nFov))
	beta0  = np.broadcast_to(beta0, (bsize, asize, nScan*nFor*nFov))
	
	z = cris_losSPC[:, 2]
	z  = np.broadcast_to(z, (bsize, asize, nScan*nFor*nFov))
	
	
	alphav = np.broadcast_to(alphav, (nScan*nFor*nFov, bsize, asize))
	alphav = np.transpose(alphav, (1, 2, 0))
	betav = np.broadcast_to(betav, (nScan*nFor*nFov, bsize, asize))
	betav = np.transpose(betav, (1, 2, 0))
	
	x = np.tan(alphav + alpha0)*z
	y = np.tan(betav + beta0)*z
		
	cris_losSPC_pert = np.zeros((bsize, asize, nScan*nFor*nFov, 3))
	cris_losSPC_pert[:, :, :, 0] = x 
	cris_losSPC_pert[:, :, :, 1] = y 
	cris_losSPC_pert[:, :, :, 2] = z
	cris_losSPC_pert = geo.normalize_vector(cris_losSPC_pert)
	
	Mat_SPCtoECI = geo.matrix_matrix_product(Mat_SPCtoOrb, Mat_OrbtoECI)
	Mat_ECItoSPC = np.transpose(Mat_SPCtoECI, (0, 2, 1))
	Mat_ECItoSPC = np.broadcast_to(Mat_ECItoSPC, (bsize, asize, nScan*nFor*nFov, 3, 3))
	cris_losECI_pert = geo.matrix_vector_product(Mat_ECItoSPC, cris_losSPC_pert)
	
	cris_FOVTime_pert = np.broadcast_to(cris_FOVTime, (bsize, asize, nScan, nFor, nFov))
	
	cris_losECR_pert, temp = geo.ECI2ECR_NOVAS(cris_FOVTime_pert.ravel(), cris_losECI_pert.reshape(-1, 3), 0.0)
	cris_losECR_pert = cris_losECR_pert.reshape(bsize, asize, nScan, nFor, nFov, 3)
	
	print("Perturb --- %s seconds ---" % (time.time() - start_time))
	
		
	# using Kd-tree to find the closest pixel of VIIRS for each CrIS FOV
	dy, dx = geo.match_cris_viirs_pert(cris_los, cris_pos, viirs_pos, viirs_sdrQa, cris_losECR_pert)
	if dy is None: 
		return None 
	#ddy, ddx = geo.match_cris_viirs(cris_los, cris_pos, viirs_pos, viirs_sdrQa)
	
	
	# viirs averaged radiances
	cris_v_rad = np.asarray([np.mean(viirs_rad[dy[m, n, k, j, i], dx[m, n, k, j, i]]) for  m, n, k, j, i in np.ndindex(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])])
	cris_v_bt  = np.asarray([np.mean(viirs_bt[dy[m, n, k, j, i], dx[m, n, k, j, i]]) for m, n, k, j, i in np.ndindex(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])])
	
	cris_v_rad = cris_v_rad.reshape(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])
	cris_v_bt  = cris_v_bt.reshape(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])
	
	
	if pmw is not None and psw is not None:
		viirs_rad_mw = np.zeros_like(viirs_rad)
		viirs_rad_sw = np.zeros_like(viirs_rad)
		
		idx = np.where(viirs_sdrQa==0)
		viirs_rad_mw[idx] = pmw(viirs_rad[idx])
		viirs_rad_sw[idx] = psw(viirs_rad[idx])
		
		cris_v_rad_mw = np.asarray([np.mean(viirs_rad_mw[dy[m, n, k, j, i], dx[m, n, k, j, i]]) for  m, n, k, j, i in np.ndindex(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])])
		cris_v_rad_sw = np.asarray([np.mean(viirs_rad_sw[dy[m, n, k, j, i], dx[m, n, k, j, i]]) for  m, n, k, j, i in np.ndindex(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])])
	
		cris_v_rad_mw = cris_v_rad_mw.reshape(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])
		cris_v_rad_sw = cris_v_rad_sw.reshape(cris_losECR_pert.shape[0:cris_losECR_pert.ndim-1])
	
	# collocation data feild 
	field = dict()
	field['cris_v_rad']  = cris_v_rad
	field['cris_v_bt']  = cris_v_bt
	field['cris_lat']    = cris_lat
	field['cris_lon']    = cris_lon
	
	field['cris_sdr_files']    =   cris_sdr_files
	field['cris_geo_files']    =   cris_geo_files
	field['viirs_sdr_files']   =   viirs_sdr_files
	field['viirs_sdr_files']   =   viirs_sdr_files
	
	field['alpha']   =   alpha
	field['beta']    =    beta
		
	#field['dx']  = dx
	#field['dy']  = dy
	
	if pmw is not None and psw is not None:
		field['cris_v_rad_mw']  = cris_v_rad_mw
		field['cris_v_rad_sw']  = cris_v_rad_sw
	
	return field	
	
	
	
	
	
	
	
		
