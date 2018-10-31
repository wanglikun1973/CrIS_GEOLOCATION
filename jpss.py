# -*- coding: utf-8 -*-
# jpss.py

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

""" JPSS file reader and instrument parameter setup.

A collection of functions to read CrIS and VIIRS data as well as CrIS instrument 
parameter setup 

:Author:
  `Likun Wang <Likun.Wang@noaa.gov>`_

:Organization:
  Earth System Science Interdisciplinary Center/Univ. of Maryland 

:Version: 2018.9.5

"""


import h5py
import numpy as np 
import xml.etree.ElementTree as etree
import xmltodict
import os

##############################################################################################
## Generally used hdf5 file check function 

def obj_info(name, obj):
    """Print information for each object in HDF5 file."""
    print((obj.name, ":", obj))

def obj_info2(name, obj):
    """Print information for each object in HDF5 file.""" 
    if isinstance(obj, h5py.Group):
        print(("Group: %s (members: %d; attrs: %d)"
                % (obj.name, len(obj), len(obj.attrs)))) 
    elif isinstance(obj, h5py.Dataset):
        print(("\tDataset: %s (dims: %s; type: %s; attrs: %d)"
               % (os.path.basename(obj.name), obj.shape, obj.dtype, len(obj.attrs))))

    
####################################################################################
def cris_sensor_info(EngPktFile=None): 
    """
    Return a dictionary contain CrIS sensor information. 
    """
    para = dict(normBins= [717, 437, 163], \
                normRes = [0.625, 1.25, 2.5], \
                wvLow   = [650.0, 1210.0, 2155.0], \
                wvHigh  = [1095.0, 1750.0, 2550.0], \
                fullBins= [717, 869, 637], \
                fullRes = [0.625, 0.625, 0.625])    
    
    wvNorm = []
    wvFull = []
    wvNormReal = []
    wvFullReal = []
    
    ## produce wavenumber for CrIS spectra 
    for i in np.arange(0,3): 
        wv=np.linspace(para['wvLow'][i], para['wvHigh'][i], num=para['normBins'][i]-4)
        wvNorm.append(wv)

        wv=np.linspace(para['wvLow'][i], para['wvHigh'][i], num=para['fullBins'][i]-4)
        wvFull.append(wv)

        wv=np.linspace(para['wvLow'][i]-2*para['normRes'][i], \
                       para['wvHigh'][i]+2*para['normRes'][i], \
                       num=para['normBins'][i])
        wvNormReal.append(wv)
        
        wv=np.linspace(para['wvLow'][i]-2*para['normRes'][i], \
                       para['wvHigh'][i]+2*para['normRes'][i], \
                       num=para['normBins'][i])
        wvFullReal.append(wv)

    
    para['wvNorm'] = wvNorm
    para['wvFull'] = wvFull
    para['wvNormReal'] = wvNormReal
    para['wvFullReal'] = wvFullReal
    
    
    if EngPktFile is None: EngPktFile = './EngPkt/JPSS1_side1_V115_EngPkt.xml'
    
    if isinstance(EngPktFile, str): 
    
        with open(EngPktFile) as f: 
            xml = f.read()
 
        x = xmltodict.parse(xml)
        
        InstrumentId = int(x['EngPkt']['InstrumentId'])
        PktVersion   = int(x['EngPkt']['PktVersion'])
        
        lw_crosstrackOffsetAngle = np.asarray(x['EngPkt']['FovParam']['Lw']['CrosstrackOffsetAngle'].split(), dtype=np.float64)
        lw_intrackOffsetAngle    = np.asarray(x['EngPkt']['FovParam']['Lw']['IntrackOffsetAngle'].split(), dtype=np.float64)
        lw_losRelativeYaw        = float(x['EngPkt']['FovParam']['Lw']['LosRelativeYaw'])
        lw_losRelativePitch      = float(x['EngPkt']['FovParam']['Lw']['LosRelativePitch'])
        lw_fovSize               = np.asarray(x['EngPkt']['FovParam']['Lw']['Size'].split(), dtype=np.float64)

        mw_crosstrackOffsetAngle = np.asarray(x['EngPkt']['FovParam']['Mw']['CrosstrackOffsetAngle'].split(), dtype=np.float64)
        mw_intrackOffsetAngle    = np.asarray(x['EngPkt']['FovParam']['Mw']['IntrackOffsetAngle'].split(), dtype=np.float64)
        mw_losRelativeYaw        = float(x['EngPkt']['FovParam']['Mw']['LosRelativeYaw'])
        mw_losRelativePitch      = float(x['EngPkt']['FovParam']['Mw']['LosRelativePitch'])
        mw_fovSize               = np.asarray(x['EngPkt']['FovParam']['Mw']['Size'].split(), dtype=np.float64)
        
        sw_crosstrackOffsetAngle = np.asarray(x['EngPkt']['FovParam']['Sw']['CrosstrackOffsetAngle'].split(), dtype=np.float64)
        sw_intrackOffsetAngle    = np.asarray(x['EngPkt']['FovParam']['Sw']['IntrackOffsetAngle'].split(), dtype=np.float64)
        sw_losRelativeYaw        = float(x['EngPkt']['FovParam']['Sw']['LosRelativeYaw'])
        sw_losRelativePitch      = float(x['EngPkt']['FovParam']['Sw']['LosRelativePitch'])
        sw_fovSize               = np.asarray(x['EngPkt']['FovParam']['Sw']['Size'].split(), dtype=np.float64)
        
        actualCrosstrackAngle = np.asarray(x['EngPkt']['MappingParameters']['ActualCrosstrackAngleRoll'].split(), dtype=np.float64)
        actualIntrackAngle    = np.asarray(x['EngPkt']['MappingParameters']['ActualIntrackAnglePitch'].split(), dtype=np.float64)
        
        SsmrToSsmf = x['EngPkt']['MappingParameters']['SsmrToSsmf']
        SSMRtoSSMF_roll, SSMRtoSSMF_pitch, SSMRtoSSMF_yaw = [float(v) for k, v in SsmrToSsmf.items()]
        
        IarToSsmr = x['EngPkt']['MappingParameters']['IarToSsmr']
        IARtoSSMR_roll , IARtoSSMR_pitch, IARtoSSMR_yaw = [float(v) for k, v in IarToSsmr.items()]
        
        IfrBoresightToSsmf = x['EngPkt']['MappingParameters']['IfrBoresightToSsmf']
        IFRboresighttoSSMF_yaw, IFRboresighttoSSMF_pitch = [float(v) for k, v in IfrBoresightToSsmf.items()]
        
        SbfToIar = x['EngPkt']['MappingParameters']['SbfToIar']
        SBFtoIAR_roll, SBFtoIAR_pitch, SBFtoIAR_yaw =  [float(v) for k, v in SbfToIar.items()]
        
        ### millisecond == > microsecond 
        TimeStampBias = int(x['EngPkt']['MappingParameters']['TimeStampBias'])*1000
        
            
        # PCT mounting matrix
        ### NPP Case 
        if     InstrumentId == 1: SCtoSBF_roll, SCtoSBF_pitch, SCtoSBF_yaw = [-518.45683, -77.760702, 46.109524]
        if     InstrumentId == 4: SCtoSBF_roll, SCtoSBF_pitch, SCtoSBF_yaw = [ -145.84994, 267.42417,  594.61832]
        ### J1
    
    
    # putting into dictionary
    para['InstrumentId'] = InstrumentId
    para['PktVersion']   = PktVersion 
    
    para['lw_crosstrackOffsetAngle'] = lw_crosstrackOffsetAngle
    para['mw_crosstrackOffsetAngle'] = mw_crosstrackOffsetAngle
    para['sw_crosstrackOffsetAngle'] = sw_crosstrackOffsetAngle
    
    para['lw_intrackOffsetAngle'] = lw_intrackOffsetAngle
    para['mw_intrackOffsetAngle'] = mw_intrackOffsetAngle
    para['sw_intrackOffsetAngle'] = sw_intrackOffsetAngle
    
    para['lw_losRelativeYaw'] = lw_losRelativeYaw
    para['mw_losRelativeYaw'] = mw_losRelativeYaw
    para['sw_losRelativeYaw'] = sw_losRelativeYaw
    
    para['lw_losRelativePitch'] = lw_losRelativePitch
    para['mw_losRelativePitch'] = mw_losRelativePitch
    para['sw_losRelativePitch'] = sw_losRelativePitch
    
    para['lw_fovSize'] = lw_fovSize
    para['mw_fovSize'] = mw_fovSize
    para['sw_fovSize'] = sw_fovSize
    
    para['actualCrosstrackAngle'] = actualCrosstrackAngle
    para['actualIntrackAngle'] = actualIntrackAngle
    
    para['SSMRtoSSMF_roll']  = SSMRtoSSMF_roll
    para['SSMRtoSSMF_pitch'] = SSMRtoSSMF_pitch
    para['SSMRtoSSMF_yaw']   = SSMRtoSSMF_yaw
    
    para['IARtoSSMR_roll']  = IARtoSSMR_roll
    para['IARtoSSMR_pitch'] = IARtoSSMR_pitch
    para['IARtoSSMR_yaw']   = IARtoSSMR_yaw
    
    para['IFRboresighttoSSMF_yaw']    =   IFRboresighttoSSMF_yaw
    para['IFRboresighttoSSMF_pitch']  =   IFRboresighttoSSMF_pitch
    
    para['SBFtoIAR_roll']   = SBFtoIAR_roll
    para['SBFtoIAR_pitch']  = SBFtoIAR_pitch
    para['SBFtoIAR_yaw']    = SBFtoIAR_yaw
    
    para['SCtoSBF_roll']  = SCtoSBF_roll
    para['SCtoSBF_pitch'] = SCtoSBF_pitch
    para['SCtoSBF_yaw']   = SCtoSBF_yaw
    
    para['TimeStampBias'] = TimeStampBias
    
    return para
    
##############################################################################################
def read_eng_pkt (EngPktFile):
    """
    Read the XML file of CrIS ENGPKT 
    """    
    print(EngPktFile)
    with open(EngPktFile) as f: 
        xml = f.read()
        x = xmltodict.parse(xml)
    
    return x     
            
##############################################################################################
# Satellite data reader 
# read CrIS SDR files 
def read_cris_sdr (filelist, sdrFlag='Real'):

    """
    Read JPSS CrIS SDR and return LW, MW, SW Spectral. Note that this method
    is very fast but can't open too many files (<1024) simultaneously.  
    """
    
    if type(filelist) is str: filelist = [filelist]
    if len(filelist) ==0: return None
    
    # Open user block to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with open(filelist[0], 'rU') as fn:
            ub_text = fn.read(user_block_size)

    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'
    
    # read the data
    sdrs = [h5py.File(filename, 'r') for filename in filelist]
    real_lw = np.concatenate([f['All_Data'][CollectionName]['ES_RealLW'][:] for f in sdrs])
    real_mw = np.concatenate([f['All_Data'][CollectionName]['ES_RealMW'][:] for f in sdrs])
    real_sw = np.concatenate([f['All_Data'][CollectionName]['ES_RealSW'][:] for f in sdrs])
    
    QF1_SCAN_CRISSDR = np.concatenate([f['All_Data'][CollectionName]['QF1_SCAN_CRISSDR'][:] for f in sdrs])
    QF2_CRISSDR = np.concatenate([f['All_Data'][CollectionName]['QF2_CRISSDR'][:] for f in sdrs])
    QF3_CRISSDR = np.concatenate([f['All_Data'][CollectionName]['QF3_CRISSDR'][:] for f in sdrs])
    QF4_CRISSDR = np.concatenate([f['All_Data'][CollectionName]['QF4_CRISSDR'][:] for f in sdrs])

    #sdrQa = shift(shift(qf3,-6),6)
    sdrQa = QF3_CRISSDR & 0b00000011
    
    #GeoQa = shift(shift(shift(qf3, 2),-7), 7)
    geoQa = (QF3_CRISSDR & 0b00000100) >> 2

    # dayFlag = shift(shift(qf4, -7), 7)
    dayFlag = QF4_CRISSDR & 0b00000001
    
    # moonFlag 
    moonFlag = QF2_CRISSDR
        
    if sdrFlag == 'Real': return real_lw, real_mw, real_sw, sdrQa, geoQa, dayFlag
    
    if sdrFlag == 'Apod': 
    
        spcLW, spcMW, spcSW = (apodize(real_lw), apodize(real_mw), apodize(real_sw))
        return spcLW, spcMW, spcSW, sdrQa, geoQa, dayFlag
    
    if sdrFlag == 'Qa': 
        return sdrQa, geoQa, dayFlag, moonFlag
        
####################################################################################    
## read CrIS GOE files     
def read_cris_geo (filelist, ephemeris = False):
    
    """
    Read JPSS CrIS Geo files and return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle.
        if ephemeris=True, then return forTime, midTime, satellite position, velocity, attitude 
    """
    
    if type(filelist) is str: filelist = [filelist]
    if len(filelist) ==0: return None
    
    geos = [h5py.File(filename, 'r') for filename in filelist]
    
    if ephemeris == False:  
        Latitude  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['Latitude'] [:] for f in geos])
        Longitude = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['Longitude'][:] for f in geos])
        SatelliteAzimuthAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteAzimuthAngle'][:] for f in geos])
        SatelliteRange = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteRange'][:] for f in geos])
        SatelliteZenithAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SatelliteZenithAngle'][:] for f in geos])
        return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle
    if ephemeris == True:
        FORTime  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['FORTime'] [:] for f in geos])
        MidTime  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['MidTime'] [:] for f in geos])
        SCPosition  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCPosition'] [:] for f in geos])
        SCVelocity  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCVelocity'] [:] for f in geos])
        SCAttitude  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SCAttitude'] [:] for f in geos])
        return FORTime, MidTime, SCPosition, SCVelocity, SCAttitude
    if ephemeris == 'Solar':
        SolarZenithAngle  = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SolarZenithAngle'] [:] for f in geos])
        SolarAzimuthAngle = np.concatenate([f['All_Data']['CrIS-SDR-GEO_All']['SolarAzimuthAngle'] [:] for f in geos])
        return SolarAzimuthAngle, SolarZenithAngle

####################################################################################        
## READ VIIRS Geofiles 
def read_viirs_geo (filelist, ephemeris=False, hgt = False):

    """
    Read JPSS VIIRS Geo files and return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle.
        if ephemeris=True, then return midTime, satellite position, velocity, attitude 
    """
        
    if type(filelist) is str: filelist = [filelist]
    if len(filelist) ==0: return None
    
    # Opne userbloack to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with open(filelist[0], 'rU') as fs:
            ub_text = fs.read(user_block_size)
    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))
    
    #print(ub_text)
    #print(etree.tostring(ub_xml))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'    
    #print(CollectionName)

    # read the data
    geos = [h5py.File(filename, 'r') for filename in filelist]
    
    if not ephemeris:
        Latitude  = np.concatenate([f['All_Data'][CollectionName]['Latitude'][:]  for f in geos])
        Longitude = np.concatenate([f['All_Data'][CollectionName]['Longitude'][:] for f in geos])
        SatelliteAzimuthAngle = np.concatenate([f['All_Data'][CollectionName]['SatelliteAzimuthAngle'][:] for f in geos])
        SatelliteRange = np.concatenate([f['All_Data'][CollectionName]['SatelliteRange'][:] for f in geos])
        SatelliteZenithAngle = np.concatenate([f['All_Data'][CollectionName]['SatelliteZenithAngle'][:] for f in geos])
        Height = np.concatenate([f['All_Data'][CollectionName]['Height'][:] for f in geos])
        if hgt: 
            return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle, Height
        else: 
            return Longitude, Latitude, SatelliteAzimuthAngle, SatelliteRange, SatelliteZenithAngle
    
    if ephemeris: 
        MidTime  = np.concatenate([f['All_Data'][CollectionName]['MidTime'] [:] for f in geos])
        SCPosition  = np.concatenate([f['All_Data'][CollectionName]['SCPosition'][:] for f in geos])
        SCVelocity  = np.concatenate([f['All_Data'][CollectionName]['SCVelocity'][:] for f in geos])
        SCAttitude  = np.concatenate([f['All_Data'][CollectionName]['SCAttitude'][:] for f in geos])
        return MidTime, SCPosition, SCVelocity, SCAttitude 


## READ VIIRS SDR files
def read_viirs_sdr (filelist):
    """
    READ VIIRS SDR files
    """
        
    if type(filelist) is str: filelist = [filelist]
    if len(filelist) == 0: return None
    
    # Opne userbloack to read Collection_Short_Name
    with h5py.File(filelist[0], 'r') as fn:
            user_block_size = fn.userblock_size
 
    with open(filelist[0], 'rU') as fn:
            ub_text = fn.read(user_block_size)

    ub_xml = etree.fromstring(ub_text.rstrip('\x00'))

    
    #print(etree.tostring(ub_xml, pretty_print=True))
    CollectionName = ub_xml.find('Data_Product/N_Collection_Short_Name').text+'_All'    
    #print(CollectionName)
    
    s='All_Data/'+CollectionName+'/'

    # Read datasets
    sdrs = [h5py.File(filename, 'r') for filename in filelist]
    
    if 'BrightnessTemperature' in sdrs[0][s].keys(): 
        BrightnessTemperature = np.concatenate([f[s+'BrightnessTemperature'] for f in sdrs])
        BT = BrightnessTemperature
        
        if 'BrightnessTemperatureFactors' in sdrs[0][s].keys(): 
            BrightnessTemperatureFactors=np.concatenate([f[s+'BrightnessTemperatureFactors'] for f in sdrs])
            BT = BrightnessTemperature * BrightnessTemperatureFactors[0] + BrightnessTemperatureFactors[1]
        
    if 'Reflectance' in sdrs[0][s].keys(): 
    
        Reflectance = np.concatenate([f[s+'Reflectance'] for f in sdrs])
        ReflectanceFactors=np.concatenate([f[s+'ReflectanceFactors'] for f in sdrs])
        BT = Reflectance * ReflectanceFactors[0] + ReflectanceFactors[1]    
    
    Radiance = np.concatenate([f[s+'Radiance'] for f in sdrs])
    
    if 'RadianceFactors' in sdrs[0][s].keys(): 
        RadianceFactors=np.concatenate([f[s+'RadianceFactors'] for f in sdrs])
        RAD = Radiance * RadianceFactors[0] + RadianceFactors[1]
    else: 
        RAD = Radiance
    
    if CollectionName.find('VIIRS-I') >= 0:
        qaStr = 'QF1_VIIRSIBANDSDR' 
    else:   qaStr = 'QF1_VIIRSMBANDSDR' 
    QF1_VIIRSBANDSDR = np.concatenate([f[s+qaStr] for f in sdrs])
        
    return BT, RAD, QF1_VIIRSBANDSDR    
    
## READ VIIRS SDR files
def get_viirs_rsr(band, sat='NPP', grid = 'wavenumber', dir = None ): 

    """
    READ VIIRS Spectral response funcation (SRF) from IDL processed files. 
        sat = ('NPP', 'N20')
        band = ('I5', 'M13', 'M15', 'M16')
    """
    
    from scipy.io.idl import readsav
    
    
    if dir is None: 
        dir = './JPSS_VIIRS_NG_SRFs/'

    if sat is None: 
        sat = 'NPP'    
    
    if grid == 'wavenumber': 
        if band == 'M13': 
            file = sat + '.VIIRS.SRFM13.NG.sav'
        if band == 'M14': 
            file = sat + '.VIIRS.SRFM14.NG.sav'    
        if band == 'M15': 
            file = sat + '.VIIRS.SRFM15.NG.sav'
        if band == 'M16': 
            file = sat + '.VIIRS.SRFM16.NG.sav'
        if band == 'I5' or band == 'I05': 
            file = sat + '.VIIRS.SRFI5.NG.sav'    
    
    if grid == 'wavelength':
        if band == 'M13': 
            file = sat + '.VIIRS.SRFM13.wl.NG.sav' 
        if band == 'M15': 
            file = sat + '.VIIRS.SRFM15.wl.NG.sav'
        if band == 'M16': 
            file = sat + '.VIIRS.SRFM16.wl.NG.sav'
        if band == 'I5' or band == 'I05': 
            file = sat + '.VIIRS.SRFI5.wl.NG.sav'        
            
    s = readsav(dir+file)
    s['sat'] = sat
    return s    

def rsr_on_grid(srf_w, srf_v, wv):
    
    from scipy.interpolate import InterpolatedUnivariateSpline    
    
    idx=np.argsort(srf_w)
    srf_w = srf_w[idx]
    srf_v = srf_v[idx]
        
    spl = InterpolatedUnivariateSpline(srf_w, srf_v)
    return spl(wv, ext=1)

    
##############################################################################################

def rad2temp (wv, rad): 

    from pyspectral.blackbody import blackbody_wn_rad2temp 

    wavenumber = wv*100
    radiance = rad*1e-5 

    results = blackbody_wn_rad2temp([wavenumber], radiance)
    
    if  'numpy.ma.core.MaskedArray' in str(type(results)):
        return results.data
    else: 
        return results
    
    
def temp2rad (wv, temp): 

    from pyspectral.blackbody import blackbody_wn 

    wavenumber = wv*100.0
    
    results=blackbody_wn([wavenumber], temp)
    results *= 1e5 
    
    if  'numpy.ma.core.MaskedArray' in str(type(results)):    
        return results.data
    else: 
        return results
        
def rad2temp_wl (wl, rad): 
    
    from pyspectral.blackbody import blackbody_rad2temp
    
    wavelength = wl * 1e-6 # from micron ==> meter 
    radiance = rad*1e6
    
    results = blackbody_rad2temp([wavelength], radiance)
    
    if  'numpy.ma.core.MaskedArray' in str(type(results)):
        return results.data
    else: 
        return results
    
        
def apodize(realLW): 
    """
    Hamming apodization function for CrIS spectra. See CrIS SDR ATBD 3.7    
    """
    # apodization parameters
    Hanming_a = 0.23
    w0=Hanming_a
    w1=1.-2*Hanming_a
    w2=Hanming_a
    
    realLW = np.asarray(realLW, dtype=np.float64)
    
    shapeLW = realLW.shape
    realLW = realLW.reshape(-1, shapeLW[-1])
    
    apLW = np.zeros_like(realLW)
    apLW[:, 0] = w1*realLW[:, 0] + w0*realLW[:, 1]
    apLW[:, 1:shapeLW[-1]-1] = w0*realLW[:, 0:shapeLW[-1]-2] + w1*realLW[:, 1:shapeLW[-1]-1] + w2*realLW[:, 2:shapeLW[-1]]
    apLW[:, shapeLW[-1]-1] = w1*realLW[:, shapeLW[-1]-1] + w0*realLW[:, shapeLW[-1]-2]
    
    apLW = apLW[:, 2:shapeLW[-1]-2]
    apLW = apLW.reshape(shapeLW[:-1]+(-1, ))
            
    return apLW
    
def compute_band_rad(wv, spc, srf):

    eqw = np.trapz(srf, wv)
    #print(eqw, np.trapz(spc * srf , wv), np.trapz(spc * srf , wv) / eqw)
    return np.trapz(spc * srf , wv) / eqw
    
##############################################################################################    
 
