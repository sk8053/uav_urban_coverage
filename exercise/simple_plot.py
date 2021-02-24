import  numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/sk8053/mmwchanmod2/mmwchanmod')
from mmwchanmod.sim.chanmod import MPChan
from mmwchanmod.common.spherical import sph_to_cart, spherical_add_sub
import argparse
from mmwchanmod.sim.antenna import Elem3GPP
import pickle as p
tx_pow= 23
kT = -174
nf =-6
bw = 400e6

f = open ('train_test_data.p', 'rb')
raw_data = p.load(f)
angles = np.append(raw_data[1]['los_ang'] , raw_data[0]['los_ang'], axis = 0)

aod_theta = 90 - angles[:, MPChan.aod_theta_ind]
aod_phi = angles[:, MPChan.aod_phi_ind]
aoa_theta = 90- angles[:, MPChan.aoa_theta_ind]
aoa_phi = angles[:, MPChan.aoa_phi_ind]

elem_ue = Elem3GPP(thetabw=65, phibw=65)
g1 = elem_ue.response(90-aod_theta, aod_phi)
g2 = elem_ue.response(90-aoa_theta, aoa_phi)
g = g1+g2


parser = argparse.ArgumentParser(description='')
parser.add_argument('--height',action='store',default=60,type= int,\
    help='uav_height')
args = parser.parse_args()
uav_height = args.height
h = str(uav_height)

def plot_cdf(d, key):
    plt.plot(np.sort(d), np.linspace(0,1,len(d)), label = key)

#data = np.loadtxt('distance_vector_snr_60single_sector.txt')
data= {'60': np.loadtxt('distance_vector_snr_60.txt'), '30': np.loadtxt('distance_vector_snr_30.txt'),
       '120':np.loadtxt('distance_vector_snr_120.txt'), '120_p_':np.loadtxt('distance_vector_snr_120_p.txt')}
#print (data['120'][9]-data['120p'][8])
for  h in ['120','120_p_']:
    d = data[h]
    index = np.where(d[:,6]==1)[0]
    #print (h,index)
    pl = -d[:,3]#[index] #+ d[:,5]
    plot_cdf(pl,h+"m")
    #plt.scatter(d[:,1], pl, label = h)
plt.legend()
plt.grid()
plt.xlabel('Path Loss (dB) (- Path Loss Gain)')
#plt.xlabel ("Antenna Gain (dB)")
#plt.xlabel ('SNR (dB)')
#plt.ylabel ('CDF')
plt.title("PL All (dB)")
plt.tight_layout()
plt.show()
'''
data2 = np.loadtxt('distance_vector_snr_60_phi.txt')
data_a= {'100': np.loadtxt('distance_vector_60m_with_aerial_100.txt'), '200': np.loadtxt('distance_vector_60m_with_aerial_200.txt'),
       '400':np.loadtxt('distance_vector_60m_with_aerial_400.txt')}
def get_distance(data):
    dis_x= data[:,0]
    dis_y = data[:,1]
    dis_z = data[:,2]
    distance = np.column_stack((dis_x, dis_y, dis_z))
    distance = np.linalg.norm(distance, axis = 1)
    return distance, dis_x, dis_y,dis_z
d_30, x_30, y_30, z_30 = get_distance(data['30'])
d_60, x_60, y_60, z_60 = get_distance(data['60'])
d_120, x_120, y_120, z_120 = get_distance(data['120'])

pl = data[h][:,3] #-data[h][:,3]
rot_aoa_ang = data[h][:,-1]
rot_aod_ang = data[h][:,-2]
aoa_ang = data[h][:,-3]
aod_ang = data[h][:,-4]

rot_aoa_ang_phi = data[h][:,-5]
rot_aod_ang_phi = data[h][:,-6]
aoa_ang_phi = data[h][:,-7]
aod_ang_phi = data[h][:,-8]

#aod_ang = aod_ang[aod_ang<500]
#aoa_ang = aoa_ang[aoa_ang<500]
#rot_aoa_ang = rot_aoa_ang[rot_aoa_ang<500]
#rot_aod_ang = rot_aod_ang[rot_aod_ang<500]

#aod_ang_phi = aod_ang_phi[aod_ang_phi<500]
#aoa_ang_phi = aoa_ang_phi[aoa_ang_phi<500]
#rot_aoa_ang_phi = rot_aoa_ang_phi[rot_aoa_ang_phi<500]
#rot_aod_ang_phi = rot_aod_ang_phi[rot_aod_ang_phi<500]


rx_el_gain = data[h][:,-9]
tx_el_gain = data[h][:,-10]
SNR  = data[h][:,-11]
n_ele_gain = data[h][:,-12]
el_gain = data[h][:,-13]
tx_ind = data[h][:,-15]
tx_ind = tx_ind[tx_ind<500]

n_ele_gain = n_ele_gain[n_ele_gain>-100]
el_gain = el_gain[el_gain>-100]
el_gain  = np.append(el_gain, n_ele_gain)

uniq, count = np.unique(tx_ind, return_counts=True)
d_2d_100 = np.sqrt(data_a['100'][:,0] **2 - 60**2)
d_2d_200 =  np.sqrt(data_a['200'][:,0] **2 - 60**2)
d_2d_400 =  np.sqrt(data_a['400'][:,0] **2 - 60**2)

plt.scatter(d_2d_400,data_a['400'][:,2], label = '$ISD_a$, 400 m' )
plt.scatter(d_2d_200,data_a['200'][:,2], label = '$ISD_a$, 200 m' )
plt.scatter(d_2d_100,data_a['100'][:,2], label = '$ISD_a$, 100 m' )

#plt.scatter(data_a['30'][:,0],data_a['30'][:,2], label = '30 m' )
#plt.scatter(data_a['60'][:,0],data_a['60'][:,2], label = '60 m' )
#plt.scatter(data_a['120'][:,0],data_a['120'][:,2], label = '120 m' )
plt.grid()
plt.legend(fontsize = 12)
plt.xlabel ('2D distance between UAVs and serving BSs', fontsize = 13)
plt.ylabel ('SNR (dB) ', fontsize = 13)
plt.xticks(fontsize = 12)
plt.yticks(fontsize =12)
plt.tight_layout()

plt.show()

plt.figure(11,figsize=(10,8))
plt.subplot(121)
th =10.1
#aod_ang, aod_ang_phi = aod_ang[el_gain>0.7], aod_ang_phi[el_gain>0.7]
#aod_ang, aod_ang_phi = aod_ang[el_gain<-1.6], aod_ang_phi[el_gain<-1.6]
plt.scatter (aod_ang, aod_ang_phi, label = 'sector 0')
#plt.scatter (aod_ang[tx_ind==1], aod_ang_phi[tx_ind==1], label = 'sector 1')
#plt.scatter (aod_ang[tx_ind==2], aod_ang_phi[tx_ind==2], label = 'sector 2')

plt.xlabel('elevation angle of departure')
plt.ylabel('azimuth angle of departure')
plt.title ('Angle of departure before rotation')
plt.legend()
plt.grid()

plt.subplot(122)

plt.scatter (rot_aod_ang, rot_aod_ang_phi, label = 'sector 0')
#plt.scatter (rot_aod_ang[tx_ind==1], rot_aod_ang_phi[tx_ind==1], label = 'sector 1')
#plt.scatter (rot_aod_ang[tx_ind==2], rot_aod_ang_phi[tx_ind==2], label = 'sector 2')
plt.xlabel('elevation angle of departure')
#plt.ylabel('azimuth angle of departure')
plt.title ('After rotation by -12 degree')
plt.grid()

plt.figure(12,figsize=(10,8))
plt.subplot(121)
#aoa_ang, aoa_ang_phi= aoa_ang[el_gain<-1.6], aoa_ang_phi[el_gain<-1.6]

plt.scatter (aoa_ang, aoa_ang_phi, label = 'sector 0')
#plt.scatter (aoa_ang[tx_ind==1], aoa_ang_phi[tx_ind==1], label = 'sector 1')
#plt.scatter (aoa_ang[tx_ind==2], aoa_ang_phi[tx_ind==2], label = 'sector 2')
plt.legend()
plt.xlabel('elevation angle of arrival')
plt.ylabel('azimuth angle of arrival')
plt.title ('Angle of arrival before rotation')
plt.grid()

plt.subplot(122)

plt.scatter (rot_aoa_ang, rot_aoa_ang_phi, label = 'sector 0')
#plt.scatter (rot_aoa_ang[tx_ind==1],rot_aoa_ang_phi[tx_ind==1], label = 'sector 1')
#plt.scatter (rot_aoa_ang[tx_ind==2], rot_aoa_ang_phi[tx_ind==2], label = 'sector 2')
plt.legend()
#plt.scatter(rot_aoa_ang, rot_aoa_ang_phi)
plt.xlabel('elevation angle of arrival')
plt.title ('Angle of arrival after rotation')
plt.title ('After rotation by -90 degree')
plt.grid()
'''
#el_gain = n_ele_gain
#el_gain = n_ele_gain
#d_100  = data[h][:,-14]

#d_200 = data[h][:, -13]
#d_400 = data[h][:, -12]

'''
plt.figure(9)
plt.plot(np.sort(d_30), np.linspace(0,1,len(d_30)), label = 'UAV height,30m')
plt.plot(np.sort(d_60), np.linspace(0,1,len(d_60)), label = 'UAV height, 60m')
plt.plot(np.sort(d_120), np.linspace(0,1,len(d_120)), label = 'UAV height, 120m')
#plt.plot(np.sort(d_400), np.linspace(0,1,len(d_400)), label = '$ISD_a$, 400m')
plt.legend()
plt.grid()
plt.xlabel('2D distance (m)')
plt.ylabel ('CDF')
plt.title ('CDF of 2D distance at each height')


plt.figure(10)
plt.plot(np.sort(d_60), np.linspace(0,1,len(d_60)), label = '$ISD_a,\infty$')
plt.plot(np.sort(d_100), np.linspace(0,1,len(d_100)), label = '$ISD_a$, 100m')
plt.plot(np.sort(d_200), np.linspace(0,1,len(d_200)), label = '$ISD_a$, 200m')
plt.plot(np.sort(d_400), np.linspace(0,1,len(d_400)), label = '$ISD_a$, 400m')
plt.legend()
plt.grid()
plt.xlabel('Distance (m)')
plt.ylabel ('CDF')
plt.title ('CDF of distance at 60m')




plt.figure(1)

plt.plot (np.sort(SNR), np.linspace(0,1, len(SNR)))
plt.xlabel('SNR (dB)')
plt.ylabel ('CDF')
plt.grid()
plt.title ('CDF of SNR at ' + h+'m')


plt.figure (2)
plt.plot (np.sort(el_gain), np.linspace(0,1,len(el_gain)))
plt.xlabel ('element gain (dB)')
plt.ylabel ('CDF')
plt.grid()
plt.title ('CDF of Element Gain at ' +h +'m')

plt.figure(5)
#aod_ang = aod_ang[20:200]
plt.plot (np.sort(rot_aod_ang), np.linspace(0,1, len(rot_aod_ang)), label ='angle of departure')
#plt.xlabel('elevation angle of departure')
plt.xlabel('elevation angle of departure')
plt.ylabel ('CDF')
#plt.title ('CDF of elevation angle of departure at ' + h+'m')
plt.title ('CDF of elevation angle of departure at ' + h+'m after rotation')
plt.grid()

plt.figure (6)
plt.plot (np.sort(rot_aoa_ang), np.linspace(0,1, len(rot_aoa_ang)),label = 'angle of arrival')
#plt.xlabel('elevation angle of arrival')
plt.xlabel('elevation angle of arrival')
plt.ylabel ('CDF')
#plt.title ('CDF of elevation angle of arrival at ' +h+'m')
plt.title ('CDF of elevation angle of arrival at ' +h+'m after rotation')
plt.grid()
plt.show()

data_60 = np.loadtxt ('distance_vector_snr_30.txt')

dist_, x_60,y_60,_ =get_distance(data_60)

data_30 = np.loadtxt ('2d_distance_vector_30m_with_aerial_200.txt')
data_60 = np.loadtxt ('2d_distance_vector_60m_with_aerial_200.txt')
data_120 = np.loadtxt ('2d_distance_vector_120m_with_aerial_200.txt')
dist_30, el_gain_30 = data_30[:,0], data_30[:,1]
dist_60, el_gain_60 = data_60[:,0], data_60[:,1]
dist_120, el_gain_120 = data_120[:,0], data_120[:,1]

plt.figure (15)
#plt.plot (np.sort(dist_), np.linspace(0,1, len(dist_)), label = '$ISD_a$, $\infty$')
#plt.plot (np.sort(dist_100), np.linspace(0,1, len(dist_100)), label = '$ISD_a$, $100$ m')
#plt.plot (np.sort(dist_200), np.linspace(0,1, len(dist_200)), label = '$ISD_a$, $200$ m')
#plt.plot (np.sort(dist_400), np.linspace(0,1, len(dist_400)), label = '$ISD_a$, $400$ m')
plt.scatter (dist_30, el_gain_30, label = 'UAV height, 30 m', marker='.')
plt.scatter (dist_60, el_gain_60, label = 'UAV height, 60 m', marker= '.')
plt.scatter (dist_120, el_gain_120, label = 'UAV height, 120 m',marker='.')
plt.legend()

plt.xlabel('3D distance to serving BS (m)', fontsize = 14)
plt.ylabel ('CDF', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid()
plt.legend(fontsize = 13)
'''
'''
el_gain_30= np.loadtxt('distance_vector_snr_60_theta_30.txt')[:,-13]
el_gain_45= np.loadtxt('distance_vector_snr_60_theta_45.txt')[:,-13]
el_gain_65= np.loadtxt('distance_vector_snr_60_theta_65.txt')[:,-13]
el_gain_85= np.loadtxt('distance_vector_snr_60_theta_85.txt')[:,-13]
el_gain_30 = el_gain_30[el_gain_30>-100]
el_gain_45 = el_gain_45[el_gain_45>-100]
el_gain_65 = el_gain_65[el_gain_65>-100]
el_gain_85 = el_gain_85[el_gain_85>-100]

plt.figure (15)
plt.plot (np.sort(el_gain_30), np.linspace(0,1,len(el_gain_30)),label = '$\phi$ = 65, $\\theta$ = 30')
plt.plot (np.sort(el_gain_45), np.linspace(0,1,len(el_gain_45)),label = '$\phi$ = 65, $\\theta$ = 45')
plt.plot (np.sort(el_gain_65), np.linspace(0,1,len(el_gain_65)),label = '$\phi$ = 65, $\\theta$ = 65')
plt.plot (np.sort(el_gain_85), np.linspace(0,1,len(el_gain_85)),label = '$\phi$ = 65, $\\theta$ = 85')
plt.grid()
plt.xlabel('Element Gain (dB)')
plt.ylabel ('CDF')
plt.title ('CDF of element gain for different HPB angles')
plt.legend()

snr_30= np.loadtxt('distance_vector_snr_60_theta_30.txt')[:,-11]
snr_45= np.loadtxt('distance_vector_snr_60_theta_45.txt')[:,-11]
snr_65= np.loadtxt('distance_vector_snr_60_theta_65.txt')[:,-11]
snr_85= np.loadtxt('distance_vector_snr_60_theta_85.txt')[:,-11]
snr_30 = snr_30[snr_30>-100]
snr_45 = snr_45[snr_45>-100]
snr_65 = snr_65[snr_65>-100]
snr_85 = snr_85[snr_85>-100]

plt.figure (16)
plt.plot (np.sort(snr_30), np.linspace(0,1,len(snr_30)),label = '$\phi$ = 65, $\\theta$ = 35')
plt.plot (np.sort(snr_45), np.linspace(0,1,len(snr_45)),label = '$\phi$ = 65, $\\theta$ = 45')
plt.plot (np.sort(snr_65), np.linspace(0,1,len(snr_65)),label = '$\phi$ = 65, $\\theta$ = 65')
plt.plot (np.sort(snr_85), np.linspace(0,1,len(snr_85)),label = '$\phi$ = 65, $\\theta$ = 85')
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel ('CDF')
plt.title ('CDF of SNR for different HPB angles')
plt.legend()

data_60 = np.loadtxt ('2d_distance_vector_snr_60.txt')
el_gain = data_60[:,-13]
el_gain = el_gain[el_gain>-100]
dist_, x_60,y_60,_ =get_distance(data_60)
dis_ = dist_[el_gain>-100]
data_100 = np.loadtxt ('2d_distance_vector_60m_with_aerial_100.txt')
data_200 = np.loadtxt ('2d_distance_vector_60m_with_aerial_200.txt')
data_400 = np.loadtxt ('2d_distance_vector_60m_with_aerial_400.txt')
print (data_100.shape)

el_gain_100 = data_100[:,1]
el_gain_200 = data_200[:,1]
el_gain_400 = data_400[:,1]


dist_100 = data_100[:,0][el_gain_100>-100]
dist_200 = data_200[:,0][el_gain_200>-100]
dist_400= data_400 [:,0][el_gain_400>-100]

el_gain_100 = el_gain_100[el_gain_100>-100]
el_gain_200 = el_gain_200[el_gain_200>-100]
el_gain_400 = el_gain_400 [el_gain_400>-100]
plt.figure (15)
plt.plot (np.sort(dist_), np.linspace(0,1, len(dist_)), label = '$ISD_a$, $\infty$')
plt.plot (np.sort(dist_100), np.linspace(0,1, len(dist_100)), label = '$ISD_a$, $100$ m')
plt.plot (np.sort(dist_200), np.linspace(0,1, len(dist_200)), label = '$ISD_a$, $200$ m')
plt.plot (np.sort(dist_400), np.linspace(0,1, len(dist_400)), label = '$ISD_a$, $400$ m')

plt.xlabel('3D distance to serving BS (m)', fontsize = 14)
plt.ylabel ('CDF', fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid()
plt.legend(fontsize = 13)

plt.figure(16)
dist_2d_ = np.linalg.norm(np.column_stack((x_60,y_60)), axis=1)
print (dist_2d_.shape, el_gain.shape)

plt.scatter (dist_2d_, el_gain, label = '$ISD_a$, $\infty$', marker='.')
plt.scatter (dist_100, el_gain_100, label = '$ISD_a$, 100 m', marker='.')
plt.scatter (dist_200, el_gain_200, label = '$ISD_a$, 200 m', marker= '.')
plt.scatter (dist_400, el_gain_400, label = '$ISD_a$, 400 m',marker='.')
plt.legend()
'''

