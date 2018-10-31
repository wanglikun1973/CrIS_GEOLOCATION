Python Geolocation Module for CrIS Geolocation evaluation using VIIRS data 

==The following papers describe the algorithm: 

Wang, L., B. Zhang, D. Tremblay, and Y. Han (2017), Improved scheme for Cross‐track Infrared Sounder geolocation assessment and optimization, J. Geophys. Res. Atmos., 122, 519–536, doi:10.1002/2016JD025812.

Wang, L., D. A. Tremblay, Y. Han, M. Esplin, D. E. Hagan, J. Predina, L. Suwinski, X. Jin, and Y. Chen (2013), Geolocation assessment for CrIS sensor data records, J. Geophys. Res. Atmos., 118, 12,690–12,704, doi:10.1002/2013JD020376.

Wang, L., D. Tremblay, B. Zhang, and Y. Han (2016b), Fast and accurate collocation of the visible infrared imaging radiometer suite measurements with Cross‐Track Infrared Sounder, Remote Sens., 8(1), 76, doi:10.3390/rs8010076.


==Example codes: 
pass_time_n20.py: Collocating CrIS and VIIRS data through perturbing LOS vectors
check_pert.py: Analyzing the collocation results 
j1_mapping_angle.py: Deriving CrIS mapping angle 

==Modules:  
geolocaiton.py: All the geolocation related functions
jpss.py       : All the functions related to JPSS instrument and data processing
collocation.py: Collocate CrIS and VIIRS data together
 
== Directory
./Data: Testing example data 
./output: Output directory (using ./output.old to compare)
./JPSS_VIIRS_NG_SRFs/:  VIIRS RSR files (IDL format)
./ EngPkt/: N20 CrIS ENGPKT V115

== Notes

1) Python 2.7 is suggested to be installed from Anaconda Python Distribution at  https://www.continuum.io/downloads, which include several basic python scientific packages.  
e.g. Numpy, scipy, matplotlib, 
check "environment.yml" to create identical environment through conda. 

2) “geolocation” package needs following additional packages. 

pykdtree: for fast search to match CrIS and VIIRS pixels.
https://pypi.python.org/pypi/pykdtree 

pyproj: Python interface to PROJ.4 library, which performs cartographic transformations and geodetic computations.
https://pypi.python.org/pypi/pyproj 

novas: The United States Naval Observatory NOVAS astronomy library
https://pypi.python.org/pypi/novas/ 

astropy: A community Python Library for Astronomy. Astropy is included by default in the Anaconda Python Distribution
http://www.astropy.org/ 


3) “jpss” package needs following additional packages.

xmltodict is a Python module that makes working with XML feel like you are working with JSON,.
https://github.com/martinblech/xmltodict

pyspectral: used for blackbody radiances and temperature conversion, which will be removed from “jpss” in the future. 
https://pypi.python.org/pypi/pyspectral 


