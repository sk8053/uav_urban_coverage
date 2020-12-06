"""
tilting.py:  Plots the SNR behaviour at different tilting angles
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
import tqdm
import seaborn as sns
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
bw = 400e6   # Bandwidth in Hz=
nf = 6  # Noise figure in dB
kT = -174   # Thermal noise in dBm/Hz
tx_pow = 23  # TX power in dBm

aer_height = 30  # Height of the aerial cell in meterss

tilt_t = np.array([-52])

fc = 28e9  # carrier frequency in Hz
nant_gnb = np.array([8,8])  # gNB array size
nant_ue = np.array([4,4])   # UE/UAV array size
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

# UE array.  Array is pointing down.
elem_ue = Elem3GPP(thetabw=65, phibw=65)
arr_ue0 = URA(elem=elem_ue, nant=nant_ue, fc=fc)
arr_ue = RotatedArray(arr_ue0,theta0=-90)


"""
Load the pre-trained model
"""
    
# Construct and load the channel model object
print('Loading pre-trained model %s' % mod_name)
K.clear_session()
chan_mod = load_model(mod_name)
# Get types of RX     
rx_types = chan_mod.rx_types
print (rx_types)
nplot = len(chan_mod.rx_types)  

"""
Main simulation loop
"""

def getSNR(tilt_t, dx,dz):
    npts = 100  # number of points for each (x,z) bin
    SNR_list =dict()
    for itilt, tilting in enumerate(tilt_t):

        dvec = np.repeat([[dx, 0, dz]], npts, axis=0)
        arr_gnb_list_t = multi_sect_array( \
            arr_gnb0, sect_type='azimuth', theta0=tilting, nsect=nsect_t)

        rx_type_vec = np.repeat([1], npts, axis=0)

        # Loop over multiple trials
        snr = np.zeros(npts)
        link_type_to_save = np.zeros(npts)

        # Print cell type
        #print('')
        print("Simulating Elevation angle: %s" % tilting)
        print (f'distance vector  = {dx, 0, dz}')
        chan_list, link_state = chan_mod.sample_path(dvec, rx_type_vec)

        for i in trange(npts):
            # Generate random channels
            chan = chan_list[i]
            pl_gain = dir_path_loss_multi_sect( \
                arr_gnb_list_t, [arr_ue], chan)[0]

            # Compute the effective SNR
            snri = tx_pow - pl_gain - kT - nf - 10 * np.log10(bw)

            link_type_to_save[i] = link_state[i]
            snr[i] = snri

        snr = np.sort(snr)
        SNR_list[str(tilting)] = np.median(snr)
    return SNR_list

#for x_position in [0]:

#x, z = np.meshgrid(np.linspace(1,10,5), np.linspace(10,150, 11));
x_f = np.linspace(0,130,10)
z_f = np.linspace(30,160, 10)
s_x, s_z = len(x_f), len(z_f)
snr_12 = np.zeros ((s_x,s_z))
snr_45 = np.zeros((s_x,s_z))

for i,dx in enumerate(x_f):
    for j, dz in enumerate(z_f):
        SNR_list = getSNR(tilt_t, dx, dz)
        #print (SNR_list)
        snr_12[i][j] = SNR_list[str(tilt_t[0])]
        #snr_45[i][j] = SNR_list[str(tilt_t[1])]

snr_12 = np.flipud(snr_12.T)
#snr_45 = np.flipud(snr_45.T)
tilt_types = ['tilting angle = '+ str(tilt_t[0])] #, 'tilting angle = '+str(tilt_t[1])]

cmap = plt.get_cmap('plasma')

for i, snr in enumerate([snr_12]):
    #plt.subplot(1, 2, i + 1)

    sns.heatmap(snr,cmap = cmap, xticklabels = np.round(x_f,2),yticklabels = np.flip(np.round(z_f,2)))


    #plt.title ("SNR difference ($SNR_{-12~ tilted}-SNR_{45~tilted}$)")
    plt.title(tilt_types[i])
    plt.xlabel ("distance along X axis (m)")
    plt.ylabel ("distance along Z axis (m)")

plt.tight_layout()
plt.show()





'''
    #print (snr)
    #plt.subplot(1, 2, i + 1)
    #plt.imshow(snr,\
    #           extent=[np.min(x_f),np.max(x_f),np.min(z_f),np.max(z_f)],\
    #           aspect='auto', vmin=-20, vmax=60, cmap = cmap)
    #plt.subplots_adjust(bottom=0.1, right=0.97, top=0.9)
#cax = plt.axes([0.92, 0.1, 0.05, 0.8])


#plt.colorbar(sm,  ticks = np.linspace(0,3, 10), boundaries = np.arange(-0.05,2.1,0.1), cax = cax)
'''