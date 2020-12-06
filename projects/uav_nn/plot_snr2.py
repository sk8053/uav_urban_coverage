"""
plot_snr.py:  Plots the SNR distribution in a single cell environment.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import tensorflow.keras.backend as K

from tqdm import trange

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
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='snr_coverage.png', help='plot file name')        
parser.add_argument(\
    '--mod_name',action='store',\
    default='uav_lon_tok', help='model to load') 
    
args = parser.parse_args()
plot_dir = args.plot_dir
plot_fn = args.plot_fn
mod_name = args.mod_name


# Paramters
bw = 400e6   # Bandwidth in Hz
nf = 6  # Noise figure in dB
kT = -174   # Thermal noise in dBm/Hz
tx_pow = 23  # TX power in dBm
npts = 100    # number of points for each (x,z) bin
aer_height=30  # Height of the aerial cell in meterss
downtilt_t = 10  # downtilt on terrestrial cells in degrees

fc = 28e9  # carrier frequency in Hz
nant_gnb = np.array([8,8])  # gNB array size
nant_ue = np.array([4,4])   # UE/UAV array size
nsect_t = 3  # number of sectors for terrestrial gNBs
nsect_a = 3  # number of sectors for aerial gNBs.  Set to 1 or 3

# uptilt on the aerial gNBs
if nsect_a == 1:
    uptilt_a = 90
else:
    uptilt_a =45
    
# Number of x and z bins
nx = 40
nz = 20

# Range of x and z distances to test
xlim = np.array([0,500])
zlim = np.array([0,130])    

"""
Create the arrays
"""
"""
 Load the pre-trained model
 """
# Construct and load the channel model object
K.clear_session()
chan_mod = load_model(mod_name)

# Get types of RX
rx_types = chan_mod.rx_types

# Terrestrial gNB.
# We downtilt the array and then replicate it over three sectors
elem_gnb = Elem3GPP(thetabw=30, phibw=30)
arr_gnb0 = URA(elem=elem_gnb, nant=nant_gnb, fc=fc)
# UE array.  Array is pointing down.
elem_ue = Elem3GPP(thetabw=30, phibw=30)
arr_ue0 = URA(elem=elem_ue, nant=nant_ue, fc=fc)
arr_ue = RotatedArray(arr_ue0, theta0=180)

SNR_median =[]
uptilt_a_list = [-10,0, 5,10,30, 40, 45,60,80,90]
#ptilt_a_list = [45,60,80,90]
for uptilt_a in uptilt_a_list:
    # Aerial gNB
    # First create a single uptilted array
    arr_gnb_a = RotatedArray(arr_gnb0,theta0=uptilt_a)
    # For multi-sector, create a list of arrays
    if (nsect_a > 1):
        arr_gnb_list_a = multi_sect_array(
            arr_gnb0, sect_type='azimuth', theta0=uptilt_a, nsect=nsect_a)

    # for terrestrial BS, we make antennas  down-tilted
    # so we don't expect any variation of SNR in this loop
    arr_gnb_t = RotatedArray(arr_gnb0, theta0=-downtilt_t)
    arr_gnb_list_t = multi_sect_array( \
        arr_gnb_t, sect_type='azimuth', nsect=nsect_t)

    SNR = np.array([])
    #rx_types =['Terrestrial']
    rx_types = ['Aerial']
    # dvec = UAV location - BS loacation = (300,0,300) - (0,0,0)
    dvec = np.column_stack(([200],[10],[100]))

    rx_type0 = rx_types[0]
    if rx_type0 == 'Terrestrial':
        rx_type_vec = np.array([0])
    elif rx_type0 == 'Aerial':
        rx_type_vec = np.array([1])

    for i in trange(npts):
        # Generate random channels
        chan_list, link_state = chan_mod.sample_path(dvec, rx_type_vec)

        # Compute the directional path loss of each link
        n = len(chan_list)
        pl_gain = np.zeros(n)
        for j, c in enumerate(chan_list):
            if (rx_type0 == 'Aerial') and (nsect_a == 1):
                pl_gain[j] = dir_path_loss(arr_gnb_a, arr_ue, c)[0]
            elif (rx_type0 == 'Aerial'):
                pl_gain[j] = dir_path_loss_multi_sect(\
                    arr_gnb_list_a, [arr_ue], c)[0]
            else:
                pl_gain[j] = dir_path_loss_multi_sect(\
                    arr_gnb_list_t, [arr_ue], c)[0]

        # Compute the effective SNR
        snri = tx_pow - pl_gain - kT - nf - 10*np.log10(bw)
        SNR = np.append(SNR, snri)
    print(f'uptilt angle:  {uptilt_a}, median SNR value: { np.median(SNR)}')
    SNR_median.append(np.median(SNR))
plt.scatter(uptilt_a_list, SNR_median)
plt.xlabel('uptilted angle')
plt.ylabel('SNR (dB)')
plt.grid()
plt.show()

'''      
# Plot the results
for iplot, rx_type0 in enumerate(rx_types):
                    
    plt.subplot(1,nplot,iplot+1)
    plt.imshow(snr_med[:,:,iplot],\
               extent=[np.min(xlim),np.max(xlim),np.min(zlim),np.max(zlim)],\
               aspect='auto', vmin=-20, vmax=60)   

    # Add horizontal line indicating location of aerial cell
    if (rx_type0 == 'Aerial'):
        plt.plot(xlim, np.array([1,1])*aer_height, 'r--')
        
    if (iplot > 0):
        plt.yticks([])
    else:
        plt.ylabel('Elevation (m)')
    plt.xlabel('Horiz (m)')
    plt.title(rx_types[iplot])
        

# Add the colorbar
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
cax = plt.axes([0.92, 0.1, 0.05, 0.8])
plt.colorbar(cax=cax)        
    
if 1:
    # Print plot
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    print('Figure saved to %s' % plot_path)
    plt.savefig('snr_dist.png', bbox_inches='tight')
            
'''


