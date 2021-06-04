import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')

ISD_t = 200
ISD_a = 0
uav_height = 120
l_p, h_p = 0.1, 0.98
dir = '/home/sk8053/uavchanmod/data/data_boston/'
file_name = dir + 'DATA_ISD_t_' + str(ISD_t) + '_ISD_a_' + str(ISD_a) + '_height_' + str(uav_height) + '.txt'

df = pd.read_csv(file_name, delimiter='\t', index_col=False)

SNR = df['SNR']

length = len(SNR)
snr_low = np.argsort(SNR)[:int(length*l_p)]
snr_high = np.argsort(SNR)[int(length*h_p):]
snr_mid = np.argsort(SNR)[int(length*l_p):int(length*h_p)]

data_low = df.take(snr_low)
data_high = df.take(snr_high)
data_mid = df.take(snr_mid)

def get_link_state(data):
    link_state = data['link_state']
    nlos_p = np.sum(link_state==2)/len(link_state)

    link_state_strong_nlos= data[data['im']!=0]['link_state']#[snr_low]
    link_state_without_nlos = data[data['im'] == 0]['link_state']  # [snr_low]
    los_state_with_strong_nlos = link_state_strong_nlos[link_state_strong_nlos==1]
    los_state_with_strong_nlos_p = len(los_state_with_strong_nlos)/len(link_state)
    los_state_without_nlos = link_state_without_nlos[link_state_without_nlos==1]
    los_state_without_nlos_p = len(los_state_without_nlos)/len(link_state)

    return nlos_p, los_state_with_strong_nlos_p, los_state_without_nlos_p

print (get_link_state(data_low))
print (get_link_state(data_mid))
print (get_link_state(data_high))
plt.plot(np.sort(SNR), np.linspace(0,1,len(SNR)))
SNR_range = np.arange(min(SNR),max(SNR))
plt.plot(SNR_range,np.repeat([l_p], len(SNR_range)),'r--')
plt.text(-10, l_p, str(l_p))
plt.plot(SNR_range,np.repeat([h_p], len(SNR_range)),'r--')
plt.text(-10, h_p, str(h_p))
plt.yticks(np.arange(0,1,step =0.1))
plt.title ('SNR distribution for UAVs at %sm'%str(uav_height))
plt.grid()
plt.show()