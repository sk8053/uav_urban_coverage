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
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss, dir_path_loss_multi_sect
np.set_printoptions(precision=6, suppress=True)

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
def writeOneLine(f, data, l):
  for j in range(l):
    f.write(str(data[j]) + ",")
  f.write("\n")


def makeNS3File(channels, angles_set):
  f = open("rayTracing_elem_gain.txt", "w")
  for i, chan in enumerate(channels):
    l = len(chan.pl)

    f.write(str(l) + "\n")
    #aod_theta = 90 - chan.ang[:, MPChan.aod_theta_ind]
    #aod_phi = chan.ang[:, MPChan.aod_phi_ind]
    #aoa_theta = 90 - chan.ang[:, MPChan.aoa_theta_ind]
    #aoa_phi = chan.ang[:, MPChan.aoa_phi_ind]
    dly = chan.dly
    pl = chan.pl
    angles = angles_set[i]

    writeOneLine(f, dly*10e9, l)
    writeOneLine(f, -1 * pl,  l)
    writeOneLine(f, np.zeros_like(pl),l)
    writeOneLine(f, angles['aod_theta_loc'],  l)
    writeOneLine(f, angles['aod_phi_loc'],  l)
    writeOneLine(f, angles['aoa_theta_loc'], l)
    writeOneLine(f, angles['aoa_phi_loc'],  l)
  f.close()


def get_gains(bs_type, tilt_t, dvec):

    arr_gnb_list = multi_sect_array( \
        arr_gnb0, sect_type='azimuth', theta0=tilt_t, nsect=1)
    if bs_type == 'Terrestrial':
        rx_type_vec = np.repeat([1], len(dvec), axis=0)
    else:
        rx_type_vec = np.repeat([0],len(dvec), axis=0)
    # Loop over multiple trials
    snr = np.array([])

    chan_list, link_state = chan_mod.sample_path(dvec, rx_type_vec)
    angles_set = list()
    for i in tqdm(range(len(chan_list))):
        # Generate random channels
        chan = chan_list[i]
        data, angles = dir_path_loss_multi_sect( \
            arr_gnb_list, [arr_ue], chan)
        angles_set.append(angles)
        #pl_gain = data['pl_min'] #+ data['rx_elem_gain'] + data['tx_elem_gain']
        pl_gain = data['pl_eff'] - data['tx_bf'] - data['rx_bf']
        # Compute the effective SNR
        snri = tx_pow - pl_gain - kT - nf - 10 * np.log10(bw)
        snr = np.append(snr, snri)
    makeNS3File(chan_list, angles_set)

    return snr


# place terrestrial BS (0,0,0)
# then drop UAVs at (0,0,30)m height and move along x axis
height = 30
d = np.linspace(-100, 100, 2001)
xx = d
yy = np.zeros_like(d)
zz = np.array([height]*len(xx))


dvec_t = np.column_stack((xx,yy,zz))

SNR_t60 = get_gains(bs_type='Terrestrial', dvec=dvec_t, tilt_t=-12)
np.savetxt('snr_with_antenna_gain.txt', SNR_t60)

plt.figure (1)
plt.plot (np.sort (SNR_t60), np.linspace(0,1,len(SNR_t60)))

plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel ('CDF')
plt.show()


