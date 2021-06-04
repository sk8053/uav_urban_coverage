import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import pandas as pd
sys.path.append('../')
from mmwchanmod.sim.antenna import Elem3GPP
elem = Elem3GPP()

fig = plt.figure()
#ax = Axes3D(fig)

#from mmwchanmod.sim.drone_antenna_field import drone_antenna_gain

p = np.loadtxt('/home/sk8053/uavchanmod/mmwchanmod/sim/azi_ele_angles.txt')
az, ele = p.T
v = np.loadtxt('/home/sk8053/uavchanmod/mmwchanmod/sim/values.txt')
az, ele = np.array(az.T, dtype=int), np.array(ele.T, dtype=int)
v = v[ele<91]
az, ele = az[ele<91], ele[ele<91]
ele = 90- ele

gain_3gpp = elem.response(az, ele)

gain_3gpp = gain_3gpp.reshape(361,181)
gain = v.reshape(181,361)

def plot_snr(dir, h = 60, l_t = '', label = '', feature = 'SNR'):
    data200 = pd.read_csv('../'+dir +'DATA_ISD_t_200_ISD_a_200_height_'+ str(h)+'.txt', delimiter = '\t')[feature]
    data0 = pd.read_csv('../'+dir +'DATA_ISD_t_200_ISD_a_0_height_'+str(h)+'.txt', delimiter = '\t')[feature]
    data400 = pd.read_csv('../'+dir+'DATA_ISD_t_200_ISD_a_400_height_'+str(h)+'.txt', delimiter = '\t')[feature]
    data800 = pd.read_csv('../'+dir+'DATA_ISD_t_200_ISD_a_800_height_'+str(h)+'.txt', delimiter = '\t')[feature]
    plt.plot(np.sort(data0), np.linspace(0,1,len(data0)),'r'+l_t, label = 'ISD, $\infty$, '+ label)
    plt.plot(np.sort(data200), np.linspace(0, 1, len(data200)),'g'+l_t, label='$ISD_a$ = 200 m '+ label)
    plt.plot(np.sort(data400), np.linspace(0,1,len(data400)),'b'+l_t, label = 'ISD, 400 m '+ label)
    plt.plot(np.sort(data800), np.linspace(0, 1, len(data800)),'k'+l_t, label='ISD, 800 m '+ label)

height = 120
feature = 'tx_elem_gain'
plot_snr('data/long_term_bf_100/', h = height,label = 'previous results', feature=feature)
plot_snr('data/', l_t = '-.', label = 'new results', h = height, feature = feature)
plt.grid()
plt.title('UAV height = ' + str(height)+ ' m')
plt.legend()
plt.xticks(fontsize= 12)
plt.yticks (fontsize = 12)
#plt.xlabel('Beamforming Gain (dB)', fontsize = 13)
plt.xlabel('SNR (dB)', fontsize = 13)
plt.ylabel ('CDF', fontsize = 13)

#plt.colorbar()
#plt.plot( gain_3gpp, color = 'r', ls = 'none')
dir = 'data/'
h = 120
data200 = pd.read_csv('../' + dir + 'DATA_ISD_t_200_ISD_a_200_height_' + str(h) + '.txt', delimiter='\t')
plt.figure (3)
tx_elem_gain_tr = np.array(data200[data200['BS_type']==1]['pl_min'])
tx_elem_gain_ar = np.array(data200[data200['BS_type']==0]['pl_min'])
plt.plot(np.sort(tx_elem_gain_tr), np.linspace(0,1,len(tx_elem_gain_tr)), label = 'Tr')
plt.plot(np.sort(tx_elem_gain_ar), np.linspace(0,1,len(tx_elem_gain_ar)), label = 'Ar')
plt.legend()


plt.figure(2)
ax=  plt.subplot(projection="polar")
#print (gain[:,0])
print (np.max(gain))
plt.pcolormesh(gain, cmap='jet' , vmin = np.min(gain), vmax = 8)
#plt.pcolor(az.reshape(181,361), ele.reshape(181,361), gain, cmap ='jet')

#plt.imshow(gain, cmap = 'jet')
plt.colorbar()
#plt.plot( gain_3gpp+42, color = 'r', ls = 'none')
#cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
#plt.colorbar(cax = cbaxes)
plt.legend(loc= 'upper left')
plt.grid()
#plt.xlabel ('Beamforming gain (dB)', fontsize = 13)
#plt.ylabel ('CDF', fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize =13)
plt.show()
