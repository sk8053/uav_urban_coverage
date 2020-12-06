from models import ChanMod
import numpy as np
from mmchanmod import MmwaveChanel


channel_args = {'uav_number': 1,
                'frequency': 28e9, 'bw': 400e6, 'tx_power': 23, 'downtilt': 10}
mmWave_channel = MmwaveChanel(channel_args)
mmWave_channel.set_tx_rx_antenna(downtilt=10)

ue_loc = np.array([[0,0,1.5]])
bs_loc = np.array([[50,50,10]])
bs_loc2 = np.array([[-30,-20,10]])

dist_matrices = ue_loc - bs_loc
dist_matrices2 = ue_loc - bs_loc2

cell_type =np.array([ChanMod.terr_cell])

bf_gain, pl_gain,link_type = mmWave_channel.get_beamforming_gain(dist_matrices, cell_type)
w_gnb = np.random.uniform(-1,1,size = (1,64)) + 1j*np.random.uniform(-1,1, size=(1,64))
w_gnb = w_gnb / np.abs(w_gnb)
#print (w_gnb)
bf_gain2, pl_gain2,link_type2 = \
    mmWave_channel.get_beamforming_gain(dist_matrices2, cell_type,
    optimal_beaforming_gnb=False, w_gnb= w_gnb)


print (bf_gain,pl_gain, link_type)
print (bf_gain2, pl_gain2, link_type2)