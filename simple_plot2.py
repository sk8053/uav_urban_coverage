import numpy as np

from mmwchanmod.datasets.download import load_model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss, dir_path_loss_multi_sect

def writeOneLine(f, data, l):
  for j in range(l):
    f.write(str(data[j]) + ",")
  f.write("\n")


def makeNS3File(channels):
  f = open("rayTracing.txt", "w")
  for chan in channels:
    l = len(chan.pl)
    f.write(str(l) + "\n")
    aod_theta = 90 - chan.ang[:, MPChan.aod_theta_ind]
    aod_phi = chan.ang[:, MPChan.aod_phi_ind]
    aoa_theta = 90 - chan.ang[:, MPChan.aoa_theta_ind]
    aoa_phi = chan.ang[:, MPChan.aoa_phi_ind]
    dly = chan.dly
    pl = chan.pl
    writeOneLine(f,  dly, l)
    writeOneLine(f, -1 * pl,  l)
    writeOneLine(f, np.zeros_like(pl),l)
    writeOneLine(f, aod_theta,  l)
    writeOneLine(f, aod_phi,  l)
    writeOneLine(f, aoa_theta, l)
    writeOneLine(f, aoa_phi,  l)
  f.close()

# Construct the channel model object
K.clear_session()
chan_model = load_model('uav_lon_tok', src='remote')
height = 30 # set UAV height as 120m
d = np.linspace(0, 100, 1001)
xx = d
yy = np.zeros_like(d)
zz = np.array([height]*len(xx))
#xx, yy = np.meshgrid(d,d)
#xx, yy = np.array(xx.reshape(-1,)), np.array(yy.reshape(-1,))


dvec = np.column_stack((xx,yy,zz)) # this is distance vector: UAV_loc - BS_loc
dvec2 = np.column_stack((-xx,-yy,zz))
cell_type = np.array ([1]*len(zz))
# angles of channel
chan_negative, links_negative = chan_model.sample_path(dvec2, cell_type) # channels and link-states of terrestrial BSs
chan_positive, links_positive = chan_model.sample_path(dvec, cell_type) # channels and link-states of terrestrial BSs
makeNS3File(chan_positive)
pl_negative  =[]
pl_positive  =[]
for i,c in enumerate(chan_negative):
  if len(c.pl)!=0 : #and links_negative[i] ==1:
    pl_negative.append(min(c.pl))
for i, c in enumerate (chan_positive):
  if len(c.pl)!=0 : #and links_positive[i] ==1:
    pl_positive.append(min(c.pl))

plt.plot (np.sort(pl_negative), np.linspace(0,1,len(pl_negative)), label = 'negative sign')
plt.plot (np.sort(pl_positive), np.linspace(0,1,len(pl_positive)), label = 'positive sign')
plt.legend()
plt.grid()

pl_gain = np.array(pl_positive)
tx_pow = 23
kT = -174
nf = 6
bw = 400e6
snr = tx_pow - pl_gain - kT - nf - 10 * np.log10(bw)
np.savetxt('snr_with_no_gain_400.txt', snr)
plt.figure (2)
plt.plot(np.sort(snr), np.linspace(0,1,len(snr)))
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel ('CDF')
plt.show()