# -*- coding: utf-8 -*-
# geolocation.py

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

""" Accurate and fast geolocaiton coordinate transformation.

A collection of functions to handle geolocaiton coordinate transformation. 
Basically, it follows the same functions from ADL common geolocation library 
by calling the same library and using the same parameters for accuracy 
consideration. In addition, to achieve the fast calculations, some functions 
are vectorized.       

:Author:
  `Likun Wang <Likun.Wang@noaa.gov>`_

:Organization:
  Earth System Science Interdisciplinary Center/Univ. of Maryland 

:Version: 2018.9.5

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from numpy import sqrt, sin, cos, tan, deg2rad, \
    arctan2, arctan, arcsin, rad2deg, arccos    
from numpy import arcsin as asin
from numpy import arctan2 as atan2

from datetime import datetime, timedelta
from functools import partial
from itertools import izip

import pyproj
from novas.compat import cel2ter, ter2cel

from astropy.time import Time, TimeDelta
from astropy.utils import iers
from pykdtree.kdtree import KDTree

from jpss import cris_sensor_info

import time

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A*(1.0 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2

eq_rad_km=6.37813700000000e+3; #/* equatorial radius, KM */
eq_radm=6.37813700000000e+6; #/* equatorial radius, meters */
pole_radm=6.35675231424518e+6; #/* polar_radius, meters */
eccen_sqr=6.69437999014132e-3; #/* e^2 = f(2 - f) */
detic2centric=9.93305620009859e-1; #/* 1 - e^2 */
centric2detic=1.00673949674228e+0; #/* 1 / (1 - e^2) */
delta=6.73949674227643e-3; #/* ( 1/(1-f)^2 ) - 1 */

#Rotational angular velocity of Earth in radians/sec from IERS
#   Conventions (2003).
ANGVEL = 7.2921150e-5;

# Polar Wonder data
#iers.conf.auto_download = False
dat = iers.IERS_Auto.open()

# NPP/CrIS parameters 
para = cris_sensor_info()


#####################################################################################
def rotationMatrixY(thePitch): 

    """
    Compute rotation matrix around Y axis
    """
    thePitch = np.asarray(thePitch, dtype=np.float64)
    
    pitchMatrix = np.zeros((thePitch.size, 3, 3), dtype=np.float64)
    pitchMatrix[:, 0,0] = cos(thePitch)
    pitchMatrix[:, 0,1] = 0
    pitchMatrix[:, 0,2] = sin(thePitch)

    pitchMatrix[:, 1,0] = 0
    pitchMatrix[:, 1,1] = 1
    pitchMatrix[:, 1,2] = 0

    pitchMatrix[:, 2,0] = -sin(thePitch)
    pitchMatrix[:, 2,1] = 0
    pitchMatrix[:, 2,2] = cos(thePitch)
    
    return np.squeeze(pitchMatrix)
 
def rotationMatrixX(theRoll):

    """
    Compute rotation matrix around X axis
    """ 
    theRoll = np.asarray(theRoll, dtype=np.float64)
    
    rollMatrix = np.zeros((theRoll.size, 3, 3), dtype=np.float64)

    rollMatrix[:, 0,0] = 1;
    rollMatrix[:, 0,1] = 0;
    rollMatrix[:, 0,2] = 0;

    rollMatrix[:, 1,0] = 0;
    rollMatrix[:, 1,1] = cos(theRoll);
    rollMatrix[:, 1,2] = -sin(theRoll);

    rollMatrix[:, 2,0] = 0;
    rollMatrix[:, 2,1] = sin(theRoll);
    rollMatrix[:, 2,2] = cos(theRoll);

    return np.squeeze(rollMatrix)

def rotationMatrixZ(theYaw): 

    """
    Compute rotation matrix around Z axis
    """ 
    theYaw = np.asarray(theYaw, dtype=np.float64)

    yawMatrix = np.zeros((theYaw.size, 3, 3), dtype=np.float64)
    
    yawMatrix[:, 0,0] = cos(theYaw)
    yawMatrix[:, 0,1] = -sin(theYaw)
    yawMatrix[:, 0,2] = 0

    yawMatrix[:, 1,0] = sin(theYaw)
    yawMatrix[:, 1,1] = cos(theYaw)
    yawMatrix[:, 1,2] = 0

    yawMatrix[:, 2,0] = 0
    yawMatrix[:, 2,1] = 0
    yawMatrix[:, 2,2] = 1

    return np.squeeze(yawMatrix);


def dot_product(v, w):
    """
    Dot product of two vectors
    """
    return np.einsum('...j,...j->...', v, w)

def matrix_matrix_product(A, B):
    """
    Product of two matrix
    """
    return np.einsum('...jk,...kl->...jl', A, B)
    
def matrix_vector_product(M, v):
    """
    Product of matrix with vector
    """
    return np.einsum('...jk,...k->...j', M, v)  

def normalize_vector(v): 
    """
    Unit vector of the vectors
    """
    v = np.asarray(v, dtype=np.float64) 
    mag = np.sqrt(np.einsum('...i,...i', v, v))
    if mag.size ==1: return v/mag
    else: return v/np.expand_dims(mag, axis=-1) 

def mag_vector(v): 
    """
    the magnitude of the vectors
    """
    v = np.asarray(v, dtype=np.float64) 
    mag = np.sqrt(np.einsum('...i,...i', v, v))
    return mag
    
def findAnglesBetweenTwoVectors(v1s, v2s):
    """
    the angle of the two vectors
    """
    v1s = np.asarray(v1s, dtype=np.float64)
    v2s = np.asarray(v2s, dtype=np.float64)

    dot_v1_v2 = dot_product(v1s, v2s)
    dot_v1_v1 = dot_product(v1s, v1s)
    dot_v2_v2 = dot_product(v2s, v2s)
    return np.rad2deg(np.arccos(dot_v1_v2/(np.sqrt(dot_v1_v1)*np.sqrt(dot_v2_v2))))
    
def Triad (v1, v2, r1, r2):  
    """
    Derive transformation matrix from two vectors. The ideas are from
        https://en.wikipedia.org/wiki/Triad_method
    
    INPUTS: R = A V
            A is Transformation matrix [3,3]
            V1, V2:  the two vector in reference coordinates
            R1, R2:  the two vector in transfomed coordinates
    """ 
    vv1 = normalize_vector(v1)
    vv2 = normalize_vector(v2)
    vv3 = normalize_vector(np.cross(v1, v2))
    vv4 = np.cross(vv1, vv3)
    
    rr1 = normalize_vector(r1)
    rr2 = normalize_vector(r2)
    rr3 = normalize_vector(np.cross(r1, r2))
    rr4 = np.cross(rr1, rr3)
    
    a1 = np.column_stack((vv1, vv3, vv4))
    a2 = np.column_stack((rr1, rr3, rr4))
    a = matrix_matrix_product(a2, a1.T)
    
    return a
    
        
def compute_alpha_beta(v, degree=False): 

    """
    Convert three-variable vector into two-variable degree for 
        perturbation purpose. 
    """

    v = np.asarray(v, dtype=np.float64)
        
    if v.ndim == 1: v = np.expand_dims(v, axis=0)
    if v.ndim >= 2: 
        sz = v.shape
        v = v.reshape(-1, 3)
    
    alpha = atan2(v[:, 0], v[:, 2])
    beta  = atan2(v[:, 1], v[:, 2])
    
    if degree: alpha, beta = rad2deg(alpha), rad2deg(beta)
    
    return alpha.reshape(sz[:-1]), beta.reshape(sz[:-1])
    
        
def haversine(lon1, lat1, lon2, lat2, R=None):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    
    if R is None: R=6367.0
    
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(deg2rad, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arcsin(sqrt(a))
    km = R * c
    return km
    
def rotate_vec (rotationAxis, angle, oldVector, degree=True): 

    rotationAxis = np.asarray(rotationAxis, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    oldVector = np.asarray(oldVector, dtype=np.float64)
        
    ## check vector size 
    if rotationAxis.size != oldVector.size: 
        print ('The vector size does not match')
        return
    
    if rotationAxis.ndim == 1: 
        rotationAxis=np.expand_dims(rotationAxis, axis=0)
    
    if oldVector.ndim == 1: 
        oldVector=np.expand_dims(oldVector, axis=0) 
    
    if angle.size != 1:  
        print('check angle input')
        return 
        
    ## we need to normalize rotationAxis 
    mag = np.linalg.norm(rotationAxis, axis=1)
    rotationAxis= rotationAxis/np.expand_dims(mag, axis=1)

    ## check angle unit  
    if degree: angle = np.deg2rad(angle)
  
    cosAngle = cos(angle)        ;
    sinAngle = sin(angle)        ;

    ## 1st term
    firstTerm = oldVector * cosAngle ;
  
    ## 2nd term
    secondTerm =  np.cross(oldVector, rotationAxis) * sinAngle 
      
    ## 3rd term 
    temp= np.einsum('ij,ij->i', oldVector, rotationAxis)
    temp= np.expand_dims(temp, axis=1)
    thirdTerm = (temp * rotationAxis) * ( 1.0 - cosAngle) ;
    newVector = firstTerm - secondTerm + thirdTerm ;
    
    return newVector 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]    
        
#####################################################################################       
def buildFovLosInSSMF(FlipFOV=False): 

    
    ## from SBF_To_IAR
    MatrixP =  rotationMatrixY (para['SBFtoIAR_pitch']*1e-6)
    MatrixR =  rotationMatrixX (para['SBFtoIAR_roll']*1e-6)
    MatrixY =  rotationMatrixZ (para['SBFtoIAR_yaw']*1e-6)
    Mat_SBFtoIAR = matrix_matrix_product(MatrixR, MatrixP)
    Mat_SBFtoIAR = matrix_matrix_product(Mat_SBFtoIAR, MatrixY)

    ## from IAR_To_SSMR
    MatrixP =  rotationMatrixY (para['IARtoSSMR_pitch']*1e-6)
    MatrixR =  rotationMatrixX (para['IARtoSSMR_roll']*1e-6)
    MatrixY =  rotationMatrixZ (para['IARtoSSMR_yaw']*1e-6)
    Mat_IARtoSSMR = matrix_matrix_product(MatrixR, MatrixP)
    Mat_IARtoSSMR = matrix_matrix_product(Mat_IARtoSSMR, MatrixY)

    ## from SSMR_To_SSMF
    MatrixP =  rotationMatrixY (para['SSMRtoSSMF_pitch']*1e-6)
    MatrixR =  rotationMatrixX (para['SSMRtoSSMF_roll']*1e-6)
    MatrixY =  rotationMatrixZ (para['SSMRtoSSMF_yaw']*1e-6)
    Mat_SSMRtoSSMF = matrix_matrix_product(MatrixR, MatrixP)
    Mat_SSMRtoSSMF = matrix_matrix_product(Mat_SSMRtoSSMF, MatrixY)

    Mat_SBFtoSSMF = matrix_matrix_product(matrix_matrix_product(Mat_SBFtoIAR, Mat_IARtoSSMR), Mat_SSMRtoSSMF)

    ## nominal optical axis direction
    opticalAxis = np.array([1.0, 0, 0], dtype=np.float64)

    LW_FOV5_inTrk = para['lw_losRelativePitch'] 
    LW_FOV5_crTrk  = para['lw_losRelativeYaw'] 

    fov_inTrackAngle = np.asarray(para['lw_intrackOffsetAngle'], dtype=np.float64)
    fov_crTrackAngle = np.asarray(para['lw_crosstrackOffsetAngle'], dtype=np.float64)

    losInIOAR = np.zeros((fov_inTrackAngle.size, 3))
    losInSSMF = np.zeros((fov_inTrackAngle.size, 3))

    for i in range(0, fov_inTrackAngle.size):

        ifov = i 
        if FlipFOV: 
            if i==0: ifov = 2
            if i==2: ifov = 0
            if i==3: ifov = 5
            if i==5: ifov = 3
            if i==6: ifov = 8
            if i==8: ifov = 6
        
        fovYaw   =  -1.0*(fov_crTrackAngle[ifov] + LW_FOV5_crTrk)*1.0e-6
        fovPitch =  (fov_inTrackAngle[ifov] + LW_FOV5_inTrk)*1.0e-6
    
        losInIOAR[i, :] = matrix_vector_product(rotationMatrixZ(fovYaw), opticalAxis)
        losInIOAR[i, :] = matrix_vector_product(rotationMatrixY(fovPitch), losInIOAR[i, :]) 
    
        # Apply IOAR-SSMF transformation to account for pitch and yaw 
        #     misalignments between interferometer and scan mechanism
        losInSSMF[i, :] =  matrix_vector_product(rotationMatrixZ(para['IFRboresighttoSSMF_yaw']*1e-6)  ,   losInIOAR[i, :])
        losInSSMF[i, :] =  matrix_vector_product(rotationMatrixY(para['IFRboresighttoSSMF_pitch']*1e-6),   losInSSMF[i, :])
        
    return losInSSMF, Mat_SBFtoSSMF
    
#####################################################################################   
def buildMatSCToSBF (): 
    
    ## from SC_To_SBF
    MatrixP =  rotationMatrixY (para['SCtoSBF_pitch']*1e-6)
    MatrixR =  rotationMatrixX (para['SCtoSBF_roll']*1e-6)
    MatrixY =  rotationMatrixZ (para['SCtoSBF_yaw']*1e-6)
    Mat_SCtoSBF = matrix_matrix_product(MatrixR, MatrixP)
    Mat_SCtoSBF = matrix_matrix_product(Mat_SCtoSBF, MatrixY)
    return Mat_SCtoSBF

#####################################################################################    

def buildLosInSBF(forIET, servoErrCrTrk=None, servoErrInTrk=None):

    forIET = np.asarray(forIET)

    nScan, nFor = forIET.shape  
    nFov = 9
    
    losInSSMF, Mat_SBFtoSSMF  = buildFovLosInSSMF() 

    ## unit vector of RMF 
    normalRMF = np.array([-1.0, 0, 1.0], dtype=np.float64)/sqrt(2.0)
    
    commandedCrTrk =  np.asarray(para['actualCrosstrackAngle'], dtype=np.float64)
    commandedInTrk =  np.asarray(para['actualIntrackAngle'], dtype=np.float64)
    
    commandedCrTrk = np.broadcast_to(commandedCrTrk, (nScan, nFor))
    commandedInTrk = np.broadcast_to(commandedInTrk, (nScan, nFor))
    
    if servoErrCrTrk is None and servoErrInTrk is None: 
        commandedCrTrk = commandedCrTrk*1e-6
        commandedInTrk = commandedInTrk*1e-6
    else: 
        commandedCrTrk = (commandedCrTrk + servoErrCrTrk)*1e-6
        commandedInTrk = (commandedInTrk + servoErrInTrk)*1e-6
        
    ## adding FOV elements 
    commandedCrTrk = np.broadcast_to(commandedCrTrk, (nFov, nScan, nFor))
    commandedInTrk = np.broadcast_to(commandedInTrk, (nFov, nScan, nFor))
    
    commandedCrTrk = np.transpose(commandedCrTrk, (1, 2, 0))
    commandedInTrk = np.transpose(commandedInTrk, (1, 2, 0))
    
    
    commandedCrTrk = commandedCrTrk.reshape(nScan*nFor*nFov)
    commandedInTrk = commandedInTrk.reshape(nScan*nFor*nFov)
    
    ## Compute Normal of SSMF 
    normalSSMF = matrix_vector_product(rotationMatrixX(commandedCrTrk), normalRMF)
    normalSSMF = matrix_vector_product(rotationMatrixY(commandedInTrk), normalSSMF)
    
    ## get FOV LOS in SSMF 
    losInSSMF, Mat_SBFtoSSMF  = buildFovLosInSSMF()
    
    losInSSMF = np.broadcast_to(losInSSMF, (nScan, nFor, nFov, 3))
    losInSSMF = losInSSMF.reshape((nScan*nFor*nFov, 3))
    
    
    dotProdAns = dot_product(losInSSMF, normalSSMF) 
    dotProdAns = dotProdAns.reshape(-1, 1) * normalSSMF
    
    losInSBF = matrix_vector_product(Mat_SBFtoSSMF, losInSSMF - 2*dotProdAns) 
    
    losInSBF = losInSBF.reshape(nScan, nFor, nFov, 3)
    
    return losInSBF

    
#####################################################################################   
def buildQuatMatrix(Qw, Qi, Qj, Qk): 
    """
    //-------------------------------------------------------------------
    // See the book edited by James R. Wertz, Library of Congress CIP Data:
    // Computer Science Corporation. Attitude Systems Operation.
    //    Spacecraft Attitude Determination and Control.
    //    (Astrophysics and space library ; v. 73)
    //    'Contract no. NAS 5-11999.'
    //    TL3260.C65        1978    629.47'42       78-23657
    //    ISBN 90-277-0959-9
    //    ISBN 90-277-1204-2 (pbk.)
    //    Published by D. Reidel Publishing Company. Copyright 1978.
    //    Last reprinted 1997.  Appendix E, particullarly Page 762, E-8.
    // See also:  http://mathworld.wolfram.com/EulerAngles.html
    //
    // The spacecraft attitude quaternions provide the rotation from J2000 ECI
    // coordinates to spacecraft coordinates.
    
    """
    
    qMat2eci = np.zeros((Qw.size, 3, 3))
    
    #If the sum of squares is 1, then the quaternion is already normalized.
    mag =  sqrt(Qi*Qi + Qj*Qj + Qk*Qk + Qw*Qw);
        
    Qw /= mag;
    Qi /= mag;
    Qj /= mag;
    Qk /= mag;

    # first row of the matrix
    qMat2eci[:, 0, 0] = (Qi*Qi) - (Qj*Qj) - (Qk*Qk) + (Qw*Qw);
    qMat2eci[:, 0, 1] = 2.e0*( (Qi*Qj) + (Qk*Qw) );
    qMat2eci[:, 0, 2] = 2.e0*( (Qi*Qk) - (Qj*Qw) );

    # second row of the matrix
    qMat2eci[:, 1, 0] = 2.e0*( (Qi*Qj) - (Qk*Qw) );
    qMat2eci[:, 1, 1] = -(Qi*Qi) + (Qj*Qj) - (Qk*Qk) + (Qw*Qw);
    qMat2eci[:, 1, 2] = 2.e0*( (Qj*Qk) + (Qi*Qw) );

    # third row of the matrix
    qMat2eci[:, 2, 0] = 2.e0*( (Qi*Qk) + (Qj*Qw) );
    qMat2eci[:, 2, 1] = 2.e0*( (Qj*Qk) - (Qi*Qw) );
    qMat2eci[:, 2, 2] = -(Qi*Qi) - (Qj*Qj) + (Qk*Qk) + (Qw*Qw);

    ## Return the spacecraft coordinates from J2000 ECI to SC matrix.
    Mat_SCtoECI = qMat2eci.squeeze()
    return Mat_SCtoECI
        
#####################################################################################
def LLA2ECEF(lonIn, latIn, altIn):
    """
    Transform lon,lat,alt (WGS84 degrees, meters) to  ECEF
    x,y,z (meters)
    """
    lonRad = deg2rad(np.asarray(lonIn, dtype=np.float64) ) 
    latRad = deg2rad(np.asarray(latIn, dtype=np.float64) )
    alt    = np.asarray(altIn, dtype=np.float64) 
    a, b, e2 = WGS84_A, WGS84_B, WGS84_E2

    ## N = Radius of Curvature (meters), defined as:
    N = a/sqrt(1.0-e2*(sin(latRad)**2.0))
            
    ##$ calcute X, Y, Z
    x=(N+alt)*cos(latRad)*cos(lonRad)
    y=(N+alt)*cos(latRad)*sin(lonRad)
    z=(b**2.0/a**2.0*N + altIn)*sin(latRad)

    return x, y, z 


def RAE2ENU(azimuthIn, zenithIn, rangeIn):
    """
    Transform azimuth, zenith, range to ENU x,y,z (meters)
    """
    azimuth = deg2rad(np.asarray(azimuthIn, dtype=np.float64))
    zenith  = deg2rad(np.asarray(zenithIn, dtype=np.float64))
    r       = np.asarray(rangeIn, dtype=np.float64)

    # up 
    up = r*cos(zenith)
  
    # projection on the x-y plane 
    p = r*sin(zenith)  
  
    # north 
    north = p*cos(azimuth)
 
    # east
    east = p*sin(azimuth)   

    return east, north, up

    
def ENU2RAE(east, north, up):
    """
    Transform ENU x,y,z (meters) to azimuth angle, zenith angle, and range 
    """

    p = sqrt(east**2 + north**2 + up**2)
    
    zenith  = rad2deg(arccos(up/p))
    azimuth = rad2deg(arctan2(east, north))
    
    return p, azimuth, zenith
    
    
    
#####################################################################################
def ENU2ECEF (east, north, up, lon, dlat):
    """
    Convert local East, North, Up (ENU) coordinates to the (x,y,z) Earth Centred Earth Fixed (ECEF) coordinates
    Reference is here:  
    http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
    Note that laitutde should be geocentric latitude instead of geodetic latitude 
    Note: 

    On June 16 2015
    This note from https://en.wikipedia.org/wiki/Geodetic_datum 
    Note: \ \phi is the geodetic latitude. A prior version of this page showed use of the geocentric latitude (\ \phi^\prime).
    The geocentric latitude is not the appropriate up direction for the local tangent plane. If the
    original geodetic latitude is available it should be used, otherwise, the relationship between geodetic and geocentric
    latitude has an altitude dependency, and is captured by ...
    """ 

    x0 = np.asarray(east, dtype=np.float64)
    y0 = np.asarray(north, dtype=np.float64)
    z0 = np.asarray(up, dtype=np.float64)

    lm = deg2rad(np.asarray(lon, dtype=np.float64))
    ph = deg2rad(np.asarray(dlat, dtype=np.float64))

    x=-1.0*x0*sin(lm)-y0*cos(lm)*sin(ph)+z0*cos(lm)*cos(ph)
    y= x0*cos(lm) -y0*sin(lm)*sin(ph)+z0*sin(lm)*cos(ph)
    z= x0*0       +y0*cos(ph)        +z0*sin(ph)   

    return x, y, z

def ECEF2ENU (x, y, z, lon, dlat):
    """
    From ECEF(x, y, z) to ENU (East, North, up) coordinates at a given location(lon, dlat)
    Reference is here:  
    http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates
    Note that laitutde should be geocentric latitude instead of geodetic latitude
    """
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    lm = deg2rad(np.asarray(lon, dtype=np.float64))
    ph = deg2rad(np.asarray(dlat, dtype=np.float64))
    
    east  =       -x*sin(lm)     +  y*cos(lm) 
    north =   -x*cos(lm)*sin(ph) -  y*sin(lm)*sin(ph)   + z*cos(ph)  
    up    =    x*cos(lm)*cos(ph) +  y*sin(lm)*cos(ph)   + z*sin(ph)

    return east, north, up  

def GEO2ECEF( lon, lat, satAzimuth, satRange, satZenith, height = None): 
    """
     from geolocation fields including lon, lat, satAzimuth, satRange, satZenith to compute 
        LOS and satPos vectors in ECEF 
    """ 
    
    
    lon = np.asarray(lon, dtype=np.float64)
    lat =  np.asarray(lat, dtype=np.float64)
    if height is None: height = np.zeros_like(lat)
    satAzimuth = np.asarray(satAzimuth, dtype=np.float64) 
    satRange = np.asarray(satRange, dtype=np.float64) 
    satZenith = np.asarray(satZenith , dtype=np.float64) 
    
    savShape = lon.shape
        
    # compute CrIS Pos Vector 
    pos_x, pos_y, pos_z = LLA2ECEF(lon, lat, height)
    pos_x = np.expand_dims(pos_x, axis=-1)
    pos_y = np.expand_dims(pos_y, axis=-1)
    pos_z = np.expand_dims(pos_z, axis=-1)

    # compute CrIS LOS Vector
    east, north, up = RAE2ENU(satAzimuth, satZenith, satRange)
    los_x, los_y, los_z = ENU2ECEF(east, north, up, lon, lat)
    los_x = np.expand_dims(los_x, axis=-1)
    los_y = np.expand_dims(los_y, axis=-1)
    los_z = np.expand_dims(los_z, axis=-1)
    
    pos = np.concatenate((pos_x, pos_y, pos_z), axis=-1)
    los = np.concatenate((los_x, los_y, los_z), axis=-1)
    
    satPos = pos + los 
    
    return pos, los, satPos 
    
#####################################################################################
def ECEF2LLA(xIn, yIn, zIn):

    """
        Transform ECEF x,y,z (meters) lon,lat,alt (WGS84 degrees, meters) to  
    """

    x = np.asarray(xIn, dtype=np.float64)
    y = np.asarray(yIn, dtype=np.float64)
    z = np.asarray(zIn, dtype=np.float64)

    if x.size != y.size or x.size != z.size or x.ndim > 1: 
        print(x, y, z)
        print(x.shape, y.shape, z.shape)
        
        print ("check input x, y, z's shape")
        return

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla  = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    geoLon, geoLat, geoAlt = pyproj.transform(ecef, lla, x, y, z, radians=False)

    return geoLon, geoLat, geoAlt
    
    

#####################################################################################   
def IET2ATIME(iet):
    """
    convert IET time into TIME 
    """
    aTime = Time('1958-01-01', scale='tai') + TimeDelta(iet*1e-6, format='sec')
    return aTime


def TIME2IET(datetime): 
    """
    convert a DATETIME varible into IET 
    """
    t1=  Time(datetime, scale='utc')
    t0 = Time('1958-01-01', scale='tai')
    dt = t1.tai-t0.tai
    dt.format = 'sec'
    iet = np.int64(dt.value*1e6)
    return iet

def ATIME2IET(aTime): 
    """
    convert a TIME varible into IET 
    """
    t0 = Time('1958-01-01', scale='tai')
    dt = aTime.tai-t0.tai
    dt.format = 'sec'
    iet = np.int64(dt.value*1e6)
    return iet  
          

#####################################################################################   
    
def ECR2ECI_NOVAS (inTime, inPos, inVel, tflag='IET'):

    """
    Transform Position and Velocity vector from ECR to ECI frame using NOVAS function.   
    """
    inPos = np.asarray(inPos, dtype=np.float64)
    inVel = np.asarray(inVel, dtype=np.float64)
    
    
    oneFlag = 0 
    if inTime.size == 1: 
        inTime = np.repeat(inTime, 2)
        oneFlag = 1
    if inPos.size ==3 : 
        inPos = np.broadcast_to(inPos, (2, 3))
    if inVel.size == 3: 
        inVel = np.broadcast_to(inVel, (2, 3))
    
    
    if tflag == 'IET': 
        inTime = np.asarray(inTime)
        aTime = Time('1958-01-01', scale='tai') + TimeDelta(inTime*1e-6, format='sec')
    if tflag == 'aTime':  aTime = inTime
    
    # set Polar Motion data     
    pmx, pmy = dat.pm_xy(aTime)
    pmx=pmx.value
    pmy=pmy.value
    
    # set TT-UT1 data
    delta_t = map(timedelta.total_seconds, aTime.tt.datetime-aTime.ut1.datetime)
    
    # set jd_ut1 data 
    jd_ut1_high=aTime.ut1.jd
    jd_ut1_low = np.zeros_like(jd_ut1_high)
    
    inPos_List = inPos.tolist()
    
    mapfunc = partial(ter2cel, method=1)    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    s = map(mapfunc, jd_ut1_high, jd_ut1_low, delta_t, pmx, pmy, inPos_List)
    outPos = np.asarray(s)
    
    if inVel.size ==1  : 
        outVel = 0
    else: 
    
        # counting Earth Rotation velocity
        tempVec = np.cross(np.array([0,0, ANGVEL]), inPos, axisa=0, axisb=1)
        vectorIn = inVel + tempVec
        
        vectorIn_List = vectorIn.tolist() 
        s = map(mapfunc, jd_ut1_high, jd_ut1_low, delta_t, pmx, pmy, vectorIn_List)
        outVel = np.asarray(s)
        
    if oneFlag == 1: 
        outPos = outPos[0, :]
        if isinstance(outVel, np.ndarray): outVel = outVel[0, :]
    
    return outPos, outVel

def ECI2ECR_NOVAS (inTime, inPos, inVel, tflag='IET'):

    """
    Transform Position and Velocity vector from ECI to ECR frame using NOVAS function.   
    """
    
    inPos = np.asarray(inPos, dtype=np.float64)
    inVel = np.asarray(inVel, dtype=np.float64)
    
    
    oneFlag = 0 
    if inTime.size == 1: 
        inTime = np.repeat(inTime, 2)
        oneFlag = 1
    if inPos.size ==3 : 
        inPos = np.broadcast_to(inPos, (2, 3))
    if inVel.size == 3: 
        inVel = np.broadcast_to(inVel, (2, 3))
    

    if tflag == 'IET': 
        inTime = np.asarray(inTime)
        aTime = Time('1958-01-01', scale='tai') + TimeDelta(inTime*1e-6, format='sec')
    if tflag == 'aTime':  aTime = inTime
    
    # set Polar Motion data     
    pmx, pmy = dat.pm_xy(aTime)
    pmx=pmx.value
    pmy=pmy.value
    
    # set TT-UT1 data
    delta_t = map(timedelta.total_seconds, aTime.tt.datetime-aTime.ut1.datetime)
        
    # set jd_ut1 data 
    jd_ut1_high=aTime.ut1.jd
    jd_ut1_low = np.zeros_like(jd_ut1_high) 
    inPos_List = inPos.tolist()

    mapfunc = partial(cel2ter, method=1)    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    s = map(mapfunc, jd_ut1_high, jd_ut1_low, delta_t, pmx, pmy, inPos_List)
    
    outPos = np.asarray(s)
    
    if inVel.size == 1 : 
        vectorOut = 0
    else: 
        vectorIn_List = inVel.tolist()
        s = map(mapfunc, jd_ut1_high, jd_ut1_low, delta_t, pmx, pmy, vectorIn_List)
        vectorOut = np.asarray(s)
    
        # counting Earth Rotation velocity
        tempVec = np.cross(np.array([0,0, ANGVEL]), inPos, axisa=0, axisb=1)
        vectorOut = vectorOut - tempVec
            
    if oneFlag == 1: 
        outPos = outPos[0, :]
        if isinstance(vectorOut, np.ndarray): vectorOut = vectorOut[0, :]
    
    return outPos, vectorOut

    
##################################################################################### 
def buildECIOrbFrame (inTime, posECR, velECR, tflag='IET'): 

    """
    NOTE: The "orbit frame" is a coordinate system describing the perfect
    attitude.  If the spacecraft frame(coordinate system) is exactly aligned
    with the orbit frame, then the roll, pitch, and yaw would all be zero and
    spacecraft attitude would be perfect.  The orbit frame Z axis points down
    to geodetic nadir, the Y axis is at a right angle to the Z axis, and the
    spacecraft velocity vector (which means the Y axis is nearly at a right
    angle to the orbit plane).  The X axis completes a right handed cartesian
    coordinate system, and is less than one degree away from the direction of
    the spacecraft velocity vector.  The X and Y axis of the orbit frame
    form a plane which is geodetically horizontal.
    """

    posECR = np.asarray(posECR, dtype=np.float64)
    velECR = np.asarray(velECR, dtype=np.float64)
    inTime = np.asarray(inTime)
    
    if posECR.ndim ==1: posECR = np.expand_dims(posECR, axis=0)
    if velECR.ndim ==1: velECR = np.expand_dims(velECR, axis=0)
    if inTime.size == 1 and inTime.ndim == 0: inTime = np.expand_dims(inTime, axis=0)
        
    
    if tflag == 'IET': aTime = Time('1958-01-01', scale='tai') + TimeDelta(inTime*1e-6, format='sec')
    
    # compute satellite lon, geodetic lat, and altitude 
    sc_Lon, sc_dLat, sc_Alt = ECEF2LLA(posECR[:, 0], posECR[:, 1], posECR[:, 2])
    
    # compute gedetic nadir position vector in ECR  
    nadir_p = np.zeros_like(posECR)
    nadir_p[:, 0], nadir_p[:, 1], nadir_p[:, 2] =LLA2ECEF(sc_Lon, sc_dLat, np.zeros_like(sc_Lon))
    
    # the vector from satellite to geodetic nadir 
    nadirVecECR = nadir_p - posECR 
    nadirVecECR = normalize_vector(nadirVecECR)
    
    # convert all the vector from ECR to ECI
    
    posECI, velECI = ECR2ECI_NOVAS(inTime, posECR, velECR)
    
    nadirVecECI, tmpVel    = ECR2ECI_NOVAS(inTime, nadirVecECR, 0)
    
    orbFrameZ = nadirVecECI 
    orbFrameY = np.cross(orbFrameZ, velECI)
    orbFrameY = normalize_vector(orbFrameY)
    orbFrameX = np.cross(orbFrameY, orbFrameZ)
    
    orbFrameY = normalize_vector(orbFrameY)
    orbFrameX = normalize_vector(orbFrameX)
    
    Mat_ECItoOrb = np.zeros((int(posECR.size/3), 3, 3), dtype=np.float64)
    
    Mat_ECItoOrb[:,:, 0] = orbFrameX
    Mat_ECItoOrb[:,:, 1] = orbFrameY
    Mat_ECItoOrb[:,:, 2] = orbFrameZ
    
    # return Rotation Matrix from Orbital Frame to ECI(J2000). 
    return np.squeeze(Mat_ECItoOrb) 
#####################################################################################   
    
def orb2sc(rollIn, pitchIn, yawIn): 
    """
    Now form the direction cosine matrix from the roll, pitch, and yaw.
    This is a 3-1-2 matrix (i.e.:  yaw, then roll, then pitch).  This
    forms a matrix which will rotate a vector in Orbit Frame Coordinates
    to spacecraft coordinates.
    
    input roll,pitch, yaw in radium
    """ 
    roll = np.asarray(rollIn, dtype=np.float64)
    pitch = np.asarray(pitchIn, dtype=np.float64)
    yaw = np.asarray(yawIn, dtype=np.float64)
    
    if np.asarray(roll).shape != np.asarray(pitch).shape or \
        np.asarray(roll).shape != np.asarray(yaw).shape: 
        return None

    matDC = np.zeros((np.asarray(roll).size,  3, 3), dtype=np.float64)

    sin_roll = sin(roll)
    cos_roll = cos(roll)

    sin_pitch = sin(pitch)
    cos_pitch = cos(pitch)

    sin_yaw = sin(yaw)
    cos_yaw = cos(yaw)

    # Noted that it is from orbit to spacecraft
    
    # First row of the matrix.
    matDC[:, 0, 0] = (cos_yaw * cos_pitch) - (sin_yaw * sin_roll * sin_pitch)
    matDC[:, 0, 1] = (sin_yaw * cos_pitch) + (cos_yaw * sin_roll * sin_pitch)
    matDC[:, 0, 2] = -cos_roll * sin_pitch

    # Second row of matrix
    matDC[:, 1, 0] = -sin_yaw * cos_roll
    matDC[:, 1, 1] = cos_yaw * cos_roll
    matDC[:, 1, 2] = sin_roll

    #Third row of matrix.
    matDC[:, 2, 0] = (cos_yaw * sin_pitch) + (sin_yaw * sin_roll * cos_pitch)
    matDC[:, 2, 1] = (sin_yaw * sin_pitch) - (cos_yaw * sin_roll * cos_pitch)
    matDC[:, 2, 2] = cos_roll * cos_pitch
    
    Mat_SPCtoOrb = np.squeeze(matDC)
    
    # return Rotation Matrix from Orbital Frame to Spacecraft. 
    return Mat_SPCtoOrb 
    
#####################################################################################   
    
def calcGDRollPitchYaw(matDC): 

  
  ## MatDC is Rotation Matrix from Orbital Frame to Spacecraft.   
  if matDC.ndim ==2 and matDC.size==9: matDC = np.expand_dims(matDC, axis=0)    

  roll = asin(matDC[:, 1, 2]);

  cos_roll = cos(roll);

  sin_pitch = (-matDC[:, 0, 2]) / cos_roll;
  cos_pitch = matDC[:, 2, 2] / cos_roll;

  pitch = atan2(sin_pitch, cos_pitch);

  sin_yaw = (-matDC[:, 1, 0]) / cos_roll;
  cos_yaw = matDC[:, 1, 1] / cos_roll;

  yaw = atan2(sin_yaw, cos_yaw);
  
  if matDC.size==9: return roll[0], pitch[0], yaw[0]
  else: return roll, pitch, yaw 
    
#####################################################################################
def conVec2LatLonAlt (Vec):  

    inVec = np.asarray(Vec, dtype=np.float64)

    ### WGS84 Parameters
    a = WGS84_A
    f = WGS84_F 
    b = WGS84_A
    e2= WGS84_E2
    ep = sqrt((a**2.0-b**2.0)/b**2.0)

    MAX_LAT_DIFF = 1.0e-10

    ## Calculate square of radius from the Earth axis to the position.
    ## Get radius to position by adding the Z squared.

    ## Distance from axis to position.
    xyRadius = (inVec[0]*inVec[0]) + (inVec[1]*inVec[1])

    ## Radius to position.
    posRadius = sqrt(xyRadius + (inVec[2]*inVec[2])) 

    xyRadius = sqrt(xyRadius)

    ## Calculate the longitude.
    lon = arctan2(inVec[1], inVec[0]);

    ## Calculate the geocentric latitude.
    cLat = arcsin( inVec[2] / posRadius )
  
    dLat = cLat
    while True: 
  
        tmpGDLat = dLat
        sinGDLat = sin(tmpGDLat)

        convNum = 1.0 / sqrt(1.0 - e2 * sinGDLat * sinGDLat)

        adjZ = inVec[2] + (a * convNum * e2 * sinGDLat)
        dLat = arctan2(adjZ, xyRadius)
        
        if np.abs(dLat - tmpGDLat) <  MAX_LAT_DIFF: break

    RN = a/sqrt(1-e2*(sin(dLat)**2))    
    alt = xyRadius/cos(dLat) - RN 
    
    return rad2deg(lon), rad2deg(dLat), alt         
#####################################################################################

def earth_radius_D (dLat): 
    """
    DESCRIPTION:  Computes the radius of the earth, in meters, from the
    geodetic latitude.
    """
    
    rlat = deg2rad(dLat)        
   
    #***************************************************************************
    #convert input geodetic latitude to geocentric
    #save the sine of the geocentric latitude
    #**********************/

    clat = arctan( detic2centric * tan(rlat) );
    sine_lat = sin(clat)

    #/***************************************************************************
    #calculate the radius of earth, WGS 84 ellipsoid
    #**********************/
    radius = eq_radm / sqrt(1.e0 + (delta * sine_lat * sine_lat));

    return radius
        
##################################################################################### 
def match_cris_viirs(crisLos, crisPos, viirsPos, viirsMask):
    """
    Match crisLos with viirsPos using the method by Wang et al. (2016)
    Wang, L., D. A. Tremblay, B. Zhang, and Y. Han, 2016: Fast and Accurate 
      Collocation of the Visible Infrared Imaging Radiometer Suite 
      Measurements and Cross-track Infrared Sounder Measurements. 
      Remote Sensing, 8, 76; doi:10.3390/rs8010076.     
    """
    
    # Derive Satellite Postion 
    crisSat = crisPos - crisLos 
    
    # using KD-tree to find best matched points 
    # build kdtree to find match index 
    pytree_los = KDTree(viirsPos.reshape(-1, 3))
    dist_los, idx_los = pytree_los.query(crisPos.reshape(-1, 3) , sqr_dists=False)
    
    my, mx = np.unravel_index(idx_los, viirsPos.shape[0:2])
    
    idy, idx  = find_match_index(crisLos.reshape(-1, 3),crisSat.reshape(-1, 3), viirsPos, viirsMask, mx, my)
    
    if np.array(idy).size ==0: 
        idy = None 
        idx = None
    else: 
        idy = np.array(idy).reshape(crisLos.shape[0:crisLos.ndim-1])
        idx = np.array(idx).reshape(crisLos.shape[0:crisLos.ndim-1])

    return idy, idx

def match_cris_viirs_pert(crisLos, crisPos, viirsPos, viirsMask, crisLosPert):
    """
    Match crisLos with viirsPos using the method by Wang et al. (2016)
    Wang, L., D. A. Tremblay, B. Zhang, and Y. Han, 2016: Fast and Accurate 
      Collocation of the Visible Infrared Imaging Radiometer Suite 
      Measurements and Cross-track Infrared Sounder Measurements. 
      Remote Sensing, 8, 76; doi:10.3390/rs8010076.     
    """
    
    bsize, asize, nScan, nFor, nFov, nVec =  crisLosPert.shape
    
    # Derive Satellite Postion 
    crisSat = crisPos - crisLos 
    crisSatPert = np.broadcast_to(crisSat, (bsize, asize, nScan, nFor, nFov, nVec))
    crisPosPert = compute_geolocation(crisSatPert.reshape(-1, 3), crisLosPert.reshape(-1,3), flag=1)
     
    
    # using KD-tree to find best matched points 
    # build kdtree to find match index
    start_time = time.time()    
    pytree_los = KDTree(viirsPos.reshape(-1, 3))
    dist_los, idx_los = pytree_los.query(crisPosPert.reshape(-1, 3) , sqr_dists=False)
    
    my, mx = np.unravel_index(idx_los, viirsPos.shape[0:2])
    my = my.reshape(bsize, asize, -1)
    mx = mx.reshape(bsize, asize, -1)
    
    print("K-D Tree <---> %s seconds ---" % (time.time() - start_time))
    
    
    s = [find_match_index(crisLosPert[i, j].reshape(-1, 3),crisSatPert[i, j].reshape(-1, 3), viirsPos, viirsMask, mx[i, j], my[i, j]) \
        for i, j in np.ndindex(crisLosPert.shape[0:2])]
    
    s = np.array(s) 
    if s.size == 0: 
        iidy = None 
        iidx = None 
    else: 
        iidy = s[:, 0, :].reshape(crisLosPert.shape[0:crisLosPert.ndim-1])
        iidx = s[:, 1, :].reshape(crisLosPert.shape[0:crisLosPert.ndim-1])  
        
    return iidy, iidx   
    
def angle (v_pos, v_Qa, c_sat, c_los, x0, y0, cos_half_fov): 
        v_los = v_pos - c_sat
        cos_angle = dot_product(v_los, c_los)/mag_vector(v_los)  - cos_half_fov
        iy, ix = np.where ( (v_Qa == 0) & (cos_angle > 0) )
        return np.asarray(iy)+y0, np.asarray(ix)+x0 
        

def find_match_index (cris_los, cris_sat, viirs_pos_in, viirs_sdrQa_in, \
              mx, my, fovDia=0.963): 
          
    nLine, nPixel = viirs_pos_in.shape[0:2]
    
    # setup parameters 
    cos_half_fov=cos(deg2rad(fovDia/2.0)) 
    if nPixel == 3200: nc = np.round(deg2rad(fovDia/2)*833.0/0.75*4).astype(np.int)
    if nPixel == 6400: nc = np.round(deg2rad(fovDia/2)*833.0/0.375*4).astype(np.int)
    
    # return list 
    xb = mx-nc
    xb = xb.clip(0, nPixel-1)
    xe = mx+nc
    xe = xe.clip(0, nPixel-1)
    
    yb = my-nc
    yb = yb.clip(0, nLine-1)
    ye = my+nc
    ye = ye.clip(0, nLine-1)
    
    cris_los = normalize_vector(cris_los)
    
    viirs_pos_list = [viirs_pos_in[y0:y1, x0:x1] for y0, y1, x0, x1 in izip(yb, ye, xb, xe)]
    viirs_Qa_list  = [viirs_sdrQa_in[y0:y1, x0:x1] for y0, y1, x0, x1 in izip(yb, ye, xb, xe)]  
    
    
    # start_time = time.time()
    # viirs_los_list = [viirs_pos_in[yb[i]:ye[i], xb[i]:xe[i]] - cris_sat[i] for i in range(0, len(xb))]
    # cos_angle_list = [dot_product(v_los, cris_los[i])/mag_vector(v_los)  - cos_half_fov for i, v_los in   enumerate(viirs_los_list)]
    # iiy, iix     = zip(*[np.where((v_Qa == 0) & (cos_angle > 0)) for cos_angle, v_Qa in zip(cos_angle_list, viirs_Qa_list)])
    # iix = [ix + x0 for ix, x0 in zip(iix, xb)]
    # iiy = [iy + y0 for iy, y0 in zip(iiy, yb)]
    # print("Method --- %s seconds ---" % (time.time() - start_time))
    
    #start_time = time.time()   
    anglefunc = partial(angle, cos_half_fov=cos_half_fov)
    
    res = map(anglefunc,  viirs_pos_list, viirs_Qa_list, cris_sat.tolist(), cris_los.tolist(), xb.tolist(), yb.tolist())
    index_y, index_x = zip(*res)
    #print("angle <---> %s seconds ---" % (time.time() - start_time))
    
    return index_y, index_x

    #####################################################################################       

    
def compute_geolocation (satellitePosition, lineOfSight, flag=None):
    """
    geoLat, geoLon = compute_geolocation(satellitePosition, lineOfSight)

    Given the satellite position and the line-of-sight (or lookup vector), this function returns
    the geodetic (or geographic) latitude and longitude of the geolocation point on the Earth
    ellipsoid characterized by the Earth radius and flattening factor. 
    
    Input:
        SatellitePosition: Three elements vector of the satellite position in the Earth Centered
                Reference (ECR) frame (float or double)
        lineOfSight: Three elements unit vector of lookup vector ( viewing vector toward the Earth surface).
    Output:
        geoLat  :  Geodetic (geographic) latitude of the Earth geolocation point. 
        geoLon  :  Longitude of the Earth geolocation point.    
    """

    satellitePosition = np.asarray(satellitePosition, dtype=np.float64) 
    lineOfSight = np.asarray(lineOfSight, dtype=np.float64)  
    
    earthRadius = WGS84_A
    flatFact = WGS84_F
    
    if satellitePosition.ndim == 1: 
        satellitePosition = np.expand_dims(satellitePosition, axis=0)
        
    if lineOfSight.ndim == 1: 
        lineOfSight = np.expand_dims(lineOfSight, axis=0)
    
    if satellitePosition.shape != lineOfSight.shape or \
        satellitePosition.ndim !=2  or \
        lineOfSight.ndim !=2 : 
        print ('check input array ... return')
        return
        
    
    # ;;  The basic equations are:
    # ;;
    # ;;  P + lambda LOS = G  ( P = satellite position, LOS = line of sight, G = geolocation point).
    # ;;  and lambda is the slant range.
    # ;;
    # ;;  and
    # ;;
    # ;;  Gx^2 / a^2 + Gy^2 / a^2 + Gz^2 / c^2 = 1    Earth ellipsoid equation, a = equatorial radius.
    # ;;  c is polar radius ( c = ( 1-f) * a) where f is flattening factor.
        
    polarRadius = earthRadius * ( 1.0 - flatFact) ;
    
    # ;;  The geolocation vector position is the solution of
    # ;;  a quadratic equation where  A lambda^2 + B lambda + C = 0, here x is the slant range.
  
    termA = (lineOfSight[:, 0] / earthRadius)**2 + \
            (lineOfSight[:, 1] / earthRadius)**2 + \
            (lineOfSight[:, 2] / polarRadius)**2

    termB = satellitePosition[:, 0]* lineOfSight[:, 0]/ (earthRadius**2) + \
            satellitePosition[:, 1]* lineOfSight[:, 1]/ (earthRadius**2) + \
            satellitePosition[:, 2]* lineOfSight[:, 2]/ (polarRadius**2)  
    termB *= 2.0         

    termC =  (satellitePosition[:, 0]/earthRadius)**2 + \
             (satellitePosition[:, 1]/earthRadius)**2 + \
             (satellitePosition[:, 2]/polarRadius)**2 - 1.0;

    radical = termB**2.0 - (4.0 * termA * termC)
    radical = radical.ravel()
    
    ## define the output avriables
    geoLat = np.zeros(int(lineOfSight.size/3))
    geoLon = np.zeros(int(lineOfSight.size/3))
    geoAlt = np.zeros(int(lineOfSight.size/3))
    slantRange = np.zeros(int(lineOfSight.size/3))
    slantRange1 = np.zeros(int(lineOfSight.size/3))
    slantRange2 = np.zeros(int(lineOfSight.size/3))
    geolocationPoint = np.zeros_like(lineOfSight)
    
    # using proj to convert ECEF to LLA 
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla  = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
    
    # ;;  The line of sight does not intercept the Earth ellipsoid.
    index,  = np.where(radical <0) 
    if index.size > 0: 
        geoLat[index] = -9999.0
        geoLon[index] = -9999.0
  
    # ;;The line of sight does not intercept the Earth ellipsoid tangentially.
    index,  = np.where(radical == 0) 
    if index.size > 0 : 
        slantRange[index] = -termB[index] / (2.0 * termA[index]) ;
        geolocationPoint[index, :] = satellitePosition[index, :] + np.expand_dims(slantRange[index], axis=1) * lineOfSight[index,:]
        
        geoLon[index], geoLat[index], geoAlt[index] = pyproj.transform(ecef, lla, geolocationPoint[index, 0], \
                                            geolocationPoint[index, 1], geolocationPoint[index, 2],  radians=False)
    
    # ;;  The line of sight intercepts the Earth ellipsoid at 2 point, the solution
    # ;;  is the shorter slant range.
    index,  = np.where(radical > 0) 
    if index.size > 0 : 
    
        slantRange1[index] = (-1.0*termB[index] - np.sqrt(radical[index]))  / (2.0 * termA[index]);
        slantRange2[index] = (-1.0*termB[index] + np.sqrt(radical[index]))  / (2.0 * termA[index]);
      
        #;;;; find the minimum value
        slantRange[index] = np.minimum(slantRange2[index], slantRange1[index])
        
        geolocationPoint[index, :] = satellitePosition[index, :] +  np.expand_dims(slantRange[index], axis=1) * lineOfSight[index, :] 
    
        geoLon[index], geoLat[index], geoAlt[index] = pyproj.transform(ecef, lla, geolocationPoint[index, 0], \
            geolocationPoint[index, 1], geolocationPoint[index, 2],  radians=False)

    index, = np.where(slantRange <= 0.0)
    if index.size > 0 :     
        geoLat[index] = -9999.0
        geoLon[index] = -9999.0
        slantRange[index] =0.0
        
    if flag is None: 
        return geoLon, geoLat, slantRange
    else: 
        return geolocationPoint
    


def fov_shape (losVec, satVec, fovDia=0.963, degree=True):  

    losVec = np.asarray(losVec, dtype=np.float64)
    satVec = np.asarray(satVec, dtype=np.float64)
    fovDia = np.asarray(fovDia, dtype=np.float64) 

    if degree==True: 
        fovDia = np.deg2rad(fovDia)
        
    if losVec.size % 3 !=0 or losVec.size != satVec.size: 
        print("please check input ... vector size must be 3 times")
        return 
    
    if losVec.ndim == 1:
        losVec = np.expand_dims(losVec, axis=0)
    
    if satVec.ndim == 1:
        satVec = np.expand_dims(satVec, axis=0) 
        
    
    nLos, nVec = losVec.shape
    
    curFovVector =  np.zeros((37, nLos, nVec), dtype=np.float64)
    curSatVector =  np.broadcast_to(satVec, (37, nLos, nVec))
        
    
    # step 1: cross product of LOS 
    orthoVectorLOS = np.cross(losVec, np.array([[0,0,1]]))
    
    # Step 2: Rotatate the orthoVector to LOS by the FOV radius ( in radians).
    fovVector = rotate_vec(orthoVectorLOS, 0.5*fovDia, losVec, degree=False) 
    
    
    for angle in np.arange(0,37):
         curFovVector[angle, :, :] = rotate_vec(losVec, angle*10, fovVector, degree=True)

    
    curFovLon, curFovLat, curRange = compute_geolocation(curSatVector.reshape(37*nLos,3), \
                                               curFovVector.reshape(37*nLos,3))
         
    # using proj to convert ECEF to LLA 
    curFovLon = curFovLon.reshape(37, nLos)
    curFovLat = curFovLat.reshape(37, nLos)
    
    
    return curFovLon, curFovLat
        
###########################################################################
        
def interpolate_sat_vector(sat_p, sat_v, sat_att, sat_time, out_time): 
    
    ts = sat_time.size
    os = out_time.size
    vs = 3
    
    p_out   = np.zeros((os, vs), dtype=np.float64)
    v_out   = np.zeros((os, vs), dtype=np.float64)
    att_out = np.zeros((os, vs), dtype=np.float64)
    
    idx = np.where(sat_time > TIME2IET(datetime(2012, 1, 1)))
    
    for i in range(0, vs): 
        
        p_out[:, i] = np.interp(out_time, sat_time[idx], sat_p[idx, i].ravel())
        v_out[:, i] = np.interp(out_time, sat_time[idx], sat_v[idx, i].ravel())
        att_out[:, i] = np.interp(out_time, sat_time[idx], sat_att[idx, i].ravel())
        
    return p_out, v_out, att_out
        
        
    
    
             
    

