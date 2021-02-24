import numpy as np

from mmwchanmod.datasets.download import load_model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# Construct the channel model object
K.clear_session()
chan_model = load_model('uav_beijing', src='remote')
height = 120 # set UAV height as 120m
d = np.linspace(-300, 300, 200)
xx, yy = np.meshgrid(d,d)
xx, yy = np.array(xx.reshape(-1,)), np.array(yy.reshape(-1,))
zz = np.array([height]*len(xx))

dvec = np.column_stack((xx,yy,zz)) # this is distance vector: UAV_loc - BS_loc
dvec2 = np.column_stack((-xx,-yy,zz))
cell_type = np.array ([1]*len(zz))
# angles of channel
chan_negative, links_negative = chan_model.sample_path(dvec2, cell_type) # channels and link-states of terrestrial BSs
chan_positive, links_positive = chan_model.sample_path(dvec, cell_type) # channels and link-states of terrestrial BSs

pl_negative  =[]
pl_positive  =[]
for i,c in enumerate(chan_negative):
  if len(c.pl)!=0 and links_negative[i] ==1:
    pl_negative.append(min(c.pl))
for i, c in enumerate (chan_positive):
  if len(c.pl)!=0 and links_positive[i] ==1:
    pl_positive.append(min(c.pl))

plt.plot (np.sort(pl_negative), np.linspace(0,1,len(pl_negative)), label = 'negative sign')
plt.plot (np.sort(pl_positive), np.linspace(0,1,len(pl_positive)), label = 'positive sign')
plt.legend()
plt.grid()
plt.show()