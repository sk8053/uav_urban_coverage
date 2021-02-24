"""
tilting.py:  Plots the SNR behaviour at different tilting angles
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import tensorflow.keras.backend as K

from tqdm import tqdm
import seaborn as h_map
path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)

from mmwchanmod.datasets.download import load_model
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import dir_path_loss, dir_path_loss_multi_sect

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the SNR distribution')
parser.add_argument( \
    '--plot_dir', action='store', \
    default='plots', help='directory for the output plots')
parser.add_argument( \
    '--plot_fn', action='store', \
    default='snr_coverage.png', help='plot file name')
parser.add_argument( \
    '--mod_name', action='store', \
    default='uav_lon_tok', help='model to load')

args = parser.parse_args()
plot_dir = args.plot_dir
plot_fn = args.plot_fn
mod_name = args.mod_name

# Paramters
bw = 400e6  # Bandwidth in Hz=
nf = 6  # Noise figure in dB
kT = -174  # Thermal noise in dBm/Hz
tx_pow = 23  # TX power in dBm

aer_height = 60  # Height of the aerial cell in meterss

fc = 28e9  # carrier frequency in Hz
nant_gnb = np.array([8, 8])  # gNB array size
nant_ue = np.array([4, 4])  # UE/UAV array size
nsect_t = 3  # number of sectors for terrestrial gNBs

# Number of x and z  ins
nx = 1
nz = 1

"""
Create the arrays
"""
# Terrestrial gNB.
# We downtilt the array and then replicate it over three sectors
elem_gnb = Elem3GPP(thetabw=65, phibw=65)
arr_gnb0 = URA(elem=elem_gnb, nant=nant_gnb, fc=fc)
arr_gnb_list_t = multi_sect_array(arr_gnb0, sect_type='azimuth', nsect=1, theta0=-12)

# UE array.  Array is pointing down.
elem_ue = Elem3GPP(thetabw=65, phibw=65)
arr_ue0 = URA(elem=elem_ue, nant=nant_ue, fc=fc)
arr_ue = RotatedArray(arr_ue0, theta0=-90)

"""
Load the pre-trained model
"""

# Construct and load the channel model object
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(mod_name)
# Get types of RX
rx_types = chan_mod.rx_types

nplot = len(chan_mod.rx_types)

"""
Main simulation loop
"""


def get_gains(bs_type, tilt_t, dvec):
   # npts = 10000  # number of points for each (x,z) bin
    # dvec = np.repeat(dvec[None, :], npts, axis=0)
    arr_gnb_list = multi_sect_array( \
        arr_gnb0, sect_type='azimuth', theta0=tilt_t, nsect=1)
    if bs_type == 'Terrestrial':
        rx_type_vec = np.repeat([1], len(dvec), axis=0)
    else:
        rx_type_vec = np.repeat([0],len(dvec), axis=0)
    # Loop over multiple trials
    snr = np.array([])
    el_gain = np.array([])
    aod = np.array([])
    aoa = np.array([])
    # rx_elem_gain, tx_elem_gain = [],[]
    #link_type_to_save = np.zeros(npts)
    chan_list, link_state = chan_mod.sample_path(dvec, rx_type_vec)

    for i in tqdm(range(len(chan_list))):
        # Generate random channels
        chan = chan_list[i]
        data = dir_path_loss_multi_sect( \
            arr_gnb_list, [arr_ue], chan)
        pl_gain = data[0]
        # Compute the effective SNR
        snri = tx_pow - pl_gain - kT - nf - 10 * np.log10(bw)
        #link_type_to_save[i] = link_state[i]
        snr = np.append(snr, snri)
        if link_state[i] ==1:
            # 11: rx_el_gain, 10: tx_element gain, 2: element gain
            # 5 aoa theta, 7 aoa phi, 4 aod theta
            # 12: rotated_aod_theta, 13: rotated_aoa_theta
            # 14: rotated_aod_phi, 15: rotated_aoa_phi
            el_gain = np.append(el_gain, data[2])
            aod = np.append(aod, data[4])
            aoa = np.append(aoa, data[5])
        else:
            el_gain = np.append(el_gain, data[2])

    return snr, el_gain, link_state, aod,aoa


# place terrestrial BS (0,0,0)
# place aerial BS (0,0,10)
# then drop UAVs at 30m height
pt = 100
xx = np.linspace(-100, 100, pt)
zz = np.linspace (0, 140, pt)
x, z_a = np.meshgrid(xx,zz-15)
x, z_t = np.meshgrid(xx,zz)
'''
d = np.linspace(0,700, pt)
t = np.linspace(-np.pi, np.pi, pt)
x, y = np.array([]), np.array([])
for dd in d:
    for tt in t:
        x = np.append(x,dd*np.cos(tt))
        y = np.append (y, dd*np.sin(tt))
'''
aerial_height = 20
terrestrial_height =5
bs_height = aerial_height
x = x.reshape(-1,)
z_t =z_t.reshape (-1,)
z_a =z_a.reshape (-1,)

#z_t = np.array([terrestrial_height]*len(x))
#z_30 = np.array([30-bs_height]*len(x))
y = np.array([0]*len(x))
#z_120 = np.array([120-bs_height]*len(x))
#dist_2d = np.linalg.norm(np.column_stack((x,z)),axis =1)
height = 30 # set UAV height as 120m
d = np.linspace(0, 100, 1001)
xx = d
yy = np.zeros_like(d)
zz = np.array([height]*len(xx))

#dvec_t30 = np.column_stack((x,y,z_30))
dvec_a = np.column_stack((x,y,z_a))
dvec_t = np.column_stack((xx,yy,zz))
#dvec_t120 = np.column_stack((x,y,z_120))

#dvec_t30 = np.array([100, 100, 30])
#dvec_t120 = np.array([100, 100, 120])

# for i in range (len(dvec_t)):
#print('terrestrial BS at 60m')
SNR_t60, ele_gain_t, link_state,aod,aoa = get_gains(bs_type='Terrestrial', dvec=dvec_t, tilt_t=-12)
SNR_t60, ele_gain_a, link_state,aod,aoa = get_gains(bs_type='Aerial', dvec=dvec_a, tilt_t=45)
ele_gain = ele_gain_a
#SNR_t30, ele_gain_30, link_state,aod,aoa = get_gains(bs_type='Aerial', dvec=dvec_t30, tilt_t=45)
#SNR_t120, ele_gain_120, link_state,aod,aoa = get_gains(bs_type='Aerial', dvec=dvec_t120, tilt_t=45)

ele_gain_t = ele_gain_t.reshape (len(xx),len(zz))
ele_gain_t = np.flip(ele_gain_t, axis = 0)
ele_gain_a = ele_gain_a.reshape (len(xx),len(zz))
ele_gain_a = np.flip(ele_gain_a, axis = 0)

#ele_gain[ele_gain<0.0] = np.nan
cmap = 'plasma'
plt.figure (1)
plt.plot (np.sort (SNR_t60), np.linspace(0,1,len(SNR_t60)))
plt.show()
#plt.imshow(ele_gain_t,
#           extent=[np.min(xx),np.max(xx),np.min(zz),np.max(zz)],\
#               aspect='auto',vmin=-29, vmax=16)
#plt.plot ([np.min(xx),np.max(xx)], [30,30], 'r')
#plt.plot ([np.min(xx),np.max(xx)], [60,60], 'r')
#plt.plot ([np.min(xx),np.max(xx)], [120,120], 'r')

#h_map.heatmap(ele_gain_t, cmap=cmap, xticklabels=np.round(xx),
#              yticklabels=np.flip(np.round(zz)),
#              annot=False, cbar=True)
#plt.colorbar()
#plt.figure (2)
#h_map.heatmap(ele_gain_a, cmap=cmap, xticklabels=np.round(xx),
#              yticklabels=np.flip(np.round(zz)),
#              annot=False, cbar=True)
#plt.imshow(ele_gain_a, extent=[np.min(xx),np.max(xx),np.min(zz),np.max(zz)],\
#               aspect='auto',vmin=-29, vmax=16)
#plt.plot ([np.min(xx),np.max(xx)], [30,30], 'r')
#plt.plot ([np.min(xx),np.max(xx)], [60,60], 'r')
#plt.plot ([np.min(xx),np.max(xx)], [120,120], 'r')
#plt.colorbar()

#ele_gain_t, ele_gain_a = ele_gain_t.reshape(-1,), ele_gain_a.reshape(-1,)
#plt.figure(3)
#plt.plot (np.sort(ele_gain_t),np.linspace(0,1,len(ele_gain_t)), 'r',label = 'Terrestrial BS')
#plt.plot (np.sort(ele_gain_a), np.linspace(0,1,len(ele_gain_a)),'b', label ='Aerial BS' )
#plt.grid()
#plt.legend()

#print('terrestrial BS at 30m')
#SNR_t30 = get_gains(bs_type='Terrestrial', dvec=dvec_t30, tilt_t=-12)
#print('terrestrial BS at 120m')
#SNR_t120 = get_gains(bs_type='Terrestrial', dvec=dvec_t120, tilt_t=-12)

#plt.figure (2)
#ele_gain = ele_gain.reshape(-1,) #[link_state==1]
#ele_gain = ele_gain[~np.isnan(ele_gain)]
#ele_gain  =SNR_t60.reshape(-1,)
#plt.plot(np.sort(SNR_t30), np.linspace(0, 1, len(SNR_t30)), label='UAV height, 30m')
#plt.plot(np.sort(ele_gain), np.linspace(0, 1, len(ele_gain)), label='UAV height, 60m')
#plt.grid()
#plt.title ('CDF of total element gain in the case of LOS')
#plt.xlabel('total element gain')
#plt.ylabel ('CDF')
#plt.plot(np.sort(SNR_t120), np.linspace(0, 1, len(SNR_t120)), label='UAV height, 120m')
#plt.figure (3)
#aod = aod[ele_gain>1.0]
#aoa = aoa[ele_gain>1.0]
#plt.scatter(aod,aoa)
#plt.xlabel ('AOD')
#plt.ylabel ('AOA')

#plt.figure(4)
#plt.scatter (dist_2d, ele_gain_30, label= "UAV height, 30 m")
#plt.scatter(dist_2d, ele_gain_60, label ='UAV height, 60 m')
#plt.scatter(dist_2d, ele_gain_120, label = "UAV height, 120 m")
#plt.xlabel('2D distance (m)')
#plt.ylabel ('Antenna element gain (dB)')
#plt.plot (np.sort(aod),np.linspace(0,1,len(aod)))
#plt.xlabel('SNR (dB) ')
#plt.ylabel('CDF')
#plt.title('CDF of SNR for one distance vector at each height')
#plt.grid()
plt.legend()
plt.show()

