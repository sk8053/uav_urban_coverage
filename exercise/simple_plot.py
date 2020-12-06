import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/sk8053/mmwchanmod2/mmwchanmod')
from cdf_plots import snr_plot

single_sect_plotter = snr_plot(dir = '/home/sk8053/mmwchanmod2/mmwchanmod/snr_data/single_sect_dir_simple/', enabl_log=False)
multi_sect_plotter = snr_plot(dir = '/home/sk8053/mmwchanmod2/mmwchanmod/snr_data/multi_sect_dir_simple/', enabl_log=False)

data_all_single = single_sect_plotter.read_data(uav_height=60, dir='/home/sk8053/mmwchanmod2/mmwchanmod/snr_data/single_sect_dir_simple/')
data_all_multi = multi_sect_plotter.read_data(uav_height=60, dir='/home/sk8053/mmwchanmod2/mmwchanmod/snr_data/multi_sect_dir_simple/')

ISD_a = '200'
print (np.median(data_all_multi[0][ISD_a]))
print (np.median(data_all_single[0][ISD_a]))
L_s = len(data_all_single[0][ISD_a])
L_m = len(data_all_multi[0][ISD_a])

p_s = np.linspace(0,1,L_s)
p_m = np.linspace(0,1,L_m)

plt.plot (np.sort(data_all_single[0][ISD_a]), p_s,label = 'single sector')
plt.plot(np.sort(data_all_multi[0][ISD_a]),p_m, label = 'multi sector')
plt.xlabel ('SNR (dB)')
plt.ylabel ('CDF')
plt.grid()
plt.legend()
plt.title ('CDF comparison of single and multi sector cases\n UAV_height = 120, ISD_t = 200, ISD_a = 200')
plt.show()