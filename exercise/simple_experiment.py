import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mmwchanmod.datasets.download import  list_models
from mmwchanmod.datasets.download import load_model
from mmwchanmod.datasets.download import ds_gdrive_ids, model_gdrive_ids
from mmwchanmod.sim.chanmod import MPChan
from tqdm import  tqdm
from matplotlib import cm
#fromcdf_plots import snr_plot
def get_pathloss_from_one_point (dvec_t,dvec_a, cell_type_t, cell_type_a):
    #chan_model = load_model('uav_boston', src='remote')
    chan_model = load_model('uav_beijing', src='remote')
    # chan_model = load_model('uav_moscow', src='remote')
    # chan_model = load_model('uav_boston', src='remote')
    chan_a, links_a = chan_model.sample_path(dvec_a, cell_type_a)  # channels and link-states of aerial BSs
    chan_t, links_t = chan_model.sample_path(dvec_t, cell_type_t)  # channels and link-states of terrestrial BSs
    path_loss_t = []
    path_loss_a = []
    # calculate path loss  for aerial BSs
    for i, c in enumerate(chan_a):
        if len(c.pl) != 0:
            pl_min = np.min(c.pl)
            pl_lin = 10 ** (-0.1 * (c.pl - pl_min))
            pl_eff = pl_min - 10 * np.log10(np.sum(pl_lin))
            path_loss_a.append(pl_eff)  # take minimum value
    # calculate path loss  for terrestrial BSs
    for i, c in enumerate(chan_t):
        if len(c.pl) != 0:
            pl_min = np.min(c.pl)
            pl_lin = 10 ** (-0.1 * (c.pl - pl_min))
            pl_eff = pl_min - 10 * np.log10(np.sum(pl_lin))
            path_loss_t.append(pl_eff)  # take minimum value
    p_t = np.median(path_loss_t)
    p_a = np.median(path_loss_a)
    #p_t = path_loss_t [int (0.98*len(path_loss_t))]
    #p_a = path_loss_a[int(0.98 * len(path_loss_a))]

    return p_t, p_a
    #distance_t = np.delete(distance_t, ind, axis=0)  # remove distance values that do not have a connection
pt =8
d = np.linspace(-150,150, pt)
x, y = np.meshgrid(d,d)
x = x.reshape(-1,)
y = y.reshape(-1,)
z_t = np.array([3]*len(x))
z_a = np.array([20]*len(x))
uav_height = 120
dvec_t = np.column_stack((x,y,uav_height-z_t)) # this is distance vector: UAV_loc - BS_loc
dvec_a = np.column_stack((x,y,uav_height-z_a)) # aerial BS is located at (0,0,10)

# consider two cases: 1) terrestrial Bs only and 2) aerial BS only
cell_type_t = np.array([1]*len(x)) #  terrestrial cell:1 , aerial cell: 0
cell_type_a = np.array([0]*len(x)) #  terrestrial cell:1 , aerial cell: 0

path_loss_a, path_loss_t =[], []
for i in tqdm(range (len(dvec_t))):

    d_t = np.repeat(dvec_t[i][None,:], 100, axis = 0)
    d_a = np.repeat(dvec_a[i][None,:], 100, axis = 0)
    c_t = np.repeat([1],100)
    c_a = np.repeat([0], 100)
    p_t, p_a = get_pathloss_from_one_point(d_t, d_a, c_t, c_a)
    path_loss_t.append(p_t)
    path_loss_a.append (p_a)


# You can choose the channel models derived from different cities.

import matplotlib as mpl
import seaborn as h_map


x = x.reshape(pt,pt)
y = y.reshape (pt,pt)
path_loss_t =np.array (path_loss_t).reshape(pt,pt)
path_loss_a =np.array(path_loss_a).reshape(pt,pt)

plt.figure (1)
h_map.heatmap(path_loss_t, cmap='plasma', xticklabels=np.round(d, 1),
                          yticklabels=np.flip(np.round(d, 1)),
                            annot = True, cbar = True, fmt = '.0f')
plt.xlabel ('distance along x axis')
plt.ylabel ('distance along y axis')

plt.title ('Pathloss from Terrestrial BSs at ' + str (uav_height)+'m height')

plt.figure (2)
h_map.heatmap(path_loss_a, cmap='plasma', xticklabels=np.round(d, 1),
                          yticklabels=np.flip(np.round(d, 1)),
                            annot = True, cbar = True, fmt = '.0f')
plt.xlabel ('distance along x axis')
plt.ylabel ('distance along y axis')
plt.title ('Pathloss from Aerial BSs at ' + str (uav_height)+'m height')

plt.figure (3)
ax = plt.axes(projection = '3d')
ax.plot_surface(x,y, path_loss_t, color='b', alpha = 1, rstride=1, cstride=1)
ax.plot_surface(x,y, path_loss_a,color= 'r', alpha = 1, rstride=1, cstride=1)

fake2Dline = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
fake2Dline2 = mpl.lines.Line2D([0],[1], linestyle="none", c='r', marker = 'o')
#ax.legend([fake2Dline, fake2Dline2], ['terrestrial','aerial'], numpoints = 1)
#ax.legend([fake2Dline], ['terrestrial'], numpoints = 1)
ax.legend([fake2Dline,fake2Dline2], ['terrestrial','aerial'], numpoints = 1)

ax.set_xlabel('X')
ax.set_ylabel ('Y')
ax.set_zlabel ('Path Loss')
ax.text2D(0.05, 0.95, "Path Loss of UAVs at "+ str (uav_height)+'m height', transform=ax.transAxes)
#ax.set_zlim (95, 130)
plt.show()

'''
plt.scatter(d, path_loss_a, c='r', marker='*', label = 'aerial')
plt.scatter(d, path_loss_t, c='g', marker='*', label = 'terrestrial')
plt.legend()
plt.xlabel('distance')
plt.ylabel ('path loss (dB)')
plt.grid()
plt.title ('Path Loss')
plt.show()
'''