import sys
sys.path.append('../')
from mmwchanmod.sim.antenna import Elem3GPP
#from mmwchanmod.learn.models import ChanMod
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss, dir_path_loss_multi_sect
from mmwchanmod.datasets.download import load_model
import tensorflow.keras.backend as K
import numpy as np
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

elem_ue = Elem3GPP(thetabw=65, phibw=65)
elem_gnb_t = Elem3GPP(thetabw=65, phibw=65)
elem_gnb_a = Elem3GPP(thetabw=65, phibw=65)

frequency = 28e9
nsect = 1
arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=frequency)
arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([8, 8]), fc=frequency)
arr_ue0 = URA(elem=elem_ue, nant=np.array([4, 4]), fc=frequency)

#arr_gnb_list_a = multi_sect_array( arr_gnb_a, sect_type= 'azimuth', nsect=nsect, theta0=45 )
arr_gnb_list_t = RotatedArray(arr_gnb_t, theta0=90)
#arr_gnb_list_t = multi_sect_array(arr_gnb_t, sect_type = 'azimuth', nsect= nsect,theta0=90)
arr_ue = RotatedArray(arr_ue0, theta0=-90, drone = True)
K.clear_session()
chan_mod = load_model('uav_lon_tok', src='remote')
#self.npath_max = self.chan_mod.npaths_max
# Load the learned link classifier model
chan_mod.load_link_model()
# Load the learned path model
chan_mod.load_path_model()

bs_loc = np.array([0,0,0])
R  = 20
height = 10
n = 21
drone = True
if drone is True:
    type = 'E_Gain_Drone_Side_of_Measured_Patterns_using_new_rotation'
else:
    type = 'E_Gain_Drone_Side_of_3GPP_Patterns_using_new_rotation'

theta = np.linspace(-np.pi, np.pi, n)
X, Y = R*np.cos(theta), R*np.sin(theta)
uav_loc = np.column_stack((X, Y, np.repeat([height],n)))
dist_matrices = uav_loc - bs_loc

rx_type = np.array(chan_mod.rx_types)
terr_cell = np.argwhere(rx_type == 'Terrestrial')[0]
aerial_cell = np.argwhere(rx_type == 'Aerial')[0]

cell_type = np.repeat([terr_cell], len(dist_matrices))
channels, link_types = chan_mod.sample_path(dist_matrices, cell_type)
data_all = []
pl_min_s,pl_eff_s,pl_eff_l,pl_min_l = [], [],[],[]
pl_list = np.empty(shape=[0,25])
for channel in channels:
    data = dir_path_loss_multi_sect([arr_gnb_list_t], [arr_ue], channel, long_term_bf=False,instantaneous_bf=False, drone_pattern= drone)
    data_all.append(data['rx_elem_gain'])
    #print (channel.link_state)
    #print (channel.link_state, data['aoa_theta'], data['im']) #90-channel.ang[:,MPChan.aoa_theta_ind][0])
plt.scatter (X, Y, edgecolors='r', facecolors = 'r', marker='*', label = 'UAV')
plt.scatter (0, 0, edgecolors='b', facecolors = 'b', marker='^', label = 'BS')

r = np.round(data_all,2)

for i, (x, y, r0) in enumerate(zip (X[:-1], Y[:-1], r[:-1])):
    plt.text(x,y, str(r0), fontsize = 12.5)
plt.xlim(min(X)-2.5,max(X)+2.5)
plt.grid()
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend()
plt.title ('R = '+ str(R)+'m'+ ', height = '+str(height) )
plt.savefig('/home/sk8053/Downloads/'+ type +'_'+str(R)+'m.png')
theta  = np.linspace(0,180,20)

#plt.figure(3)
#phi = np.linspace(-180,180,20)
#theta = np.repeat(0, len(phi))
#elem =  Elem3GPP()
#gain = elem.response(phi,theta)

#plt.plot (phi, gain)
plt.show()

#print (data_all)
