import sys
sys.path.append('../')
from mmwchanmod.sim.antenna import Elem3GPP
#from mmwchanmod.learn.models import ChanMod
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss, dir_path_loss_multi_sect
from mmwchanmod.datasets.download import load_model
import tensorflow.keras.backend as K
import numpy as np
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
import pandas as pd
import csv
import os

elem_ue = Elem3GPP(thetabw=65, phibw=65)
elem_gnb_t = Elem3GPP(thetabw=65, phibw=65)
elem_gnb_a = Elem3GPP(thetabw=65, phibw=65)

frequency = 28e9
nsect = 3
arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=frequency)
arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([8, 8]), fc=frequency)
arr_ue0 = URA(elem=elem_ue, nant=np.array([4, 4]), fc=frequency)

arr_gnb_list_a = multi_sect_array( arr_gnb_a, sect_type= 'azimuth', nsect=nsect, theta0=45 )
arr_gnb_list_t = multi_sect_array(arr_gnb_t, sect_type = 'azimuth', nsect= nsect,theta0=-12)

K.clear_session()
chan_mod = load_model('uav_lon_tok', src='remote')
#self.npath_max = self.chan_mod.npaths_max
# Load the learned link classifier model
chan_mod.load_link_model()
# Load the learned path model
chan_mod.load_path_model()

bs_loc = np.array([0,0,0])
uav_loc = np.column_stack((np.linspace(0,300,10), np.repeat([0],10), np.repeat([30],10)))
dist_matrices = uav_loc - bs_loc

rx_type = np.array(chan_mod.rx_types)
terr_cell = np.argwhere(rx_type == 'Terrestrial')[0]
aerial_cell = np.argwhere(rx_type == 'Aerial')[0]

cell_type = np.repeat([terr_cell], len(dist_matrices))
channels, link_types = chan_mod.sample_path(dist_matrices, cell_type)
data_all = {'SP_BF':[],'LT_BF':[]}
pl_min_s,pl_eff_s,pl_eff_l,pl_min_l = [], [],[],[]
pl_list = np.empty(shape=[0,25])
for channel in channels:
    data = dir_path_loss_multi_sect(arr_gnb_list_t, [arr_ue0], channel, long_term_bf=False,instantaneous_bf=False)
    data_all['SP_BF'].append(data['rx_bf']+data['tx_bf'])
    pl_min_s.append(data['pl_min'])
    pl_eff_s.append(data['pl_eff'])
    data = dir_path_loss_multi_sect(arr_gnb_list_t, [arr_ue0], channel, long_term_bf=True, instantaneous_bf=False)
    data_all['LT_BF'].append(data['rx_bf'])
    pl_min_l.append(data['pl_min'])
    pl_eff_l.append(data['pl_eff'])

    pl = channel.pl
    pl = np.append(pl, np.repeat(np.NaN, 25-len(pl)))
    pl_list = np.vstack((pl_list, pl))

f = open ('../data/toy_data.csv','wt', encoding='utf-8', newline="")
writer = csv.writer(f)
total_data = np.column_stack((data_all['SP_BF'], data_all['LT_BF'],pl_min_s, pl_eff_s,pl_min_l, pl_min_l, pl_list))
writer.writerow(['SP_BF', 'LT_BF','PL_min_S', 'PL_eff_S', 'PL_min_L', 'PL_eff_L',  'Path Loss (dB)'])
for data in total_data:
    writer.writerow(data)
f.close()