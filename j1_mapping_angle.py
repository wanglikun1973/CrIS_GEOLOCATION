"""
    A simple python script to derive CrIS mapping angle 
        based on CrIS and viirs collocation analysis results
    Author: Likun.Wang@noaa.gov (ESSIC/UMD)
    Date  : Oct 23 2018   
"""


from __future__ import print_function

import numpy as np
import jpss
import geolocation as geo
from datetime import datetime

import pdb

print(geo.para['InstrumentId'])
print(geo.para['PktVersion'])

geo.para = jpss.cris_sensor_info(EngPktFile='./EngPkt/JPSS1_side1_V115_EngPkt.xml')
print(geo.para['InstrumentId'])
print(geo.para['PktVersion'])


t_time = geo.TIME2IET(datetime(2018, 05, 25, 00, 00, 00)) # time, arbitrary 
t_IET = np.zeros((t_time.size, 30), dtype=np.int64)
for i in np.arange(0, 30): t_IET[:, i] = t_time+ 2000000*i

losInSBF = geo.buildLosInSBF(t_IET)
losInSSMF, Mat_SBFtoSSMF = geo.buildFovLosInSSMF()
Mat_SCtoSBF = geo.buildMatSCToSBF()

commandedCrTrk =  np.asarray(geo.para['actualCrosstrackAngle'], dtype=np.float64)
commandedInTrk =  np.asarray(geo.para['actualIntrackAngle'], dtype=np.float64)
commandedCrTrk = commandedCrTrk*1e-6
commandedInTrk = commandedInTrk*1e-6

commandedCrTrk_C = np.zeros_like(commandedCrTrk)
commandedInTrk_C = np.zeros_like(commandedInTrk)


normalRMF = np.array([-1.0, 0, 1.0], dtype=np.float64)/np.sqrt(2.0)
normalSSMF = geo.matrix_vector_product(geo.rotationMatrixX(commandedCrTrk), normalRMF)
normalSSMF = geo.matrix_vector_product(geo.rotationMatrixY(commandedInTrk), normalSSMF)

#################################################
### load geolocation assessment results here 

cor_flag = False
alpha_mean_i5 = np.load('cris_geo_alpha_beta.npz')['alphaM_i5']
beta_mean_i5  = np.load('cris_geo_alpha_beta.npz')['betaM_i5']
    
    
for i in np.arange(0, 30):
    losInSSMF = np.array([1.0, 0.0, 0.0])
    dotProdAns = geo.dot_product(losInSSMF, normalSSMF[i]) 
	
    losScanSSMF = losInSSMF - 2*dotProdAns*normalSSMF[i]
    losScanSBF  = geo.matrix_vector_product(Mat_SBFtoSSMF, losScanSSMF)
    losScanSC   = geo.matrix_vector_product(Mat_SCtoSBF, losScanSBF)
    
    if cor_flag: 
        alpha, beta = geo.compute_alpha_beta(losScanSC)
        print(alpha, beta)
        alpha = alpha + alpha_mean_i5[i]*1e-6
        beta = beta  + beta_mean_i5[i]*1e-6
        print(alpha, beta)
        vv = np.array([np.tan(alpha), np.tan(beta), 1.0])*losScanSC[2]
        vv = geo.normalize_vector(vv)
    else: 
        vv = losScanSC
    
    geo.compute_alpha_beta(vv)
    losScanSC_C = vv
    losScanSBF_C = geo.matrix_vector_product(np.transpose(Mat_SCtoSBF), losScanSC_C)
    losScanSSMF_C = geo.matrix_vector_product(np.transpose(Mat_SBFtoSSMF), losScanSBF_C)
    normalSSMF_C = geo.normalize_vector(losScanSSMF_C -losInSSMF) * np.sqrt(2.0)
    
    print(normalSSMF_C)
    print(normalSSMF[i])

    r_roll =  -np.arcsin(normalSSMF_C[1])
    c_r= np.cos(r_roll)
    r_pitch = np.arcsin((normalSSMF_C[2] + normalSSMF_C[0]*c_r )/(1.0+c_r*c_r))

    commandedCrTrk_C[i] = r_roll*1e6
    commandedInTrk_C[i] = r_pitch*1e6	

    print('new', i, int(r_roll*1e6), int(r_pitch*1e6))
    print('old', i, int(commandedCrTrk[i]*1e6), int(commandedInTrk[i]*1e6))
    print('=========================================')
pdb.set_trace()	
	 
print('%10s %10s %10s %10s %10s %10s %10s' %
          ('FOR', 'CrTrk_V115', 'InTrk_V115', 'CrTrk_V114', 'InTrk_V114', 'V115-V114(CrTrk)', 'V115-V114(InTrk)'))     
for i in np.arange(0, 30): 
    print('%10d %10d %10d %10d %10d %10d %10d' %
    (i+1, np.rint(commandedCrTrk_C[i]), np.rint(commandedInTrk_C[i]), commandedCrTrk[i]*1e6, commandedInTrk[i]*1e6, 
    np.rint(commandedCrTrk_C[i])-commandedCrTrk[i]*1e6, np.rint(commandedInTrk_C[i])-commandedInTrk[i]*1e6)) 

with open('angle_V116_test.txt', 'w') as f:
    print('%10s %10s %10s %10s %10s %10s %10s' %
             ('FOR', 'CrTrk_V115', 'InTrk_V115', 'CrTrk_V114', 'InTrk_V114', 'V115-V114(CrTrk)', 'V115-V114(InTrk)'), file=f)
    
    for i in np.arange(0, 30):     
        print('%10d %10d %10d %10d %10d %10d %10d' %
            (i+1, np.rint(commandedCrTrk_C[i]), np.rint(commandedInTrk_C[i]), commandedCrTrk[i]*1e6, commandedInTrk[i]*1e6, 
            np.rint(commandedCrTrk_C[i])-commandedCrTrk[i]*1e6, np.rint(commandedInTrk_C[i])-commandedInTrk[i]*1e6), file=f)
 
	





