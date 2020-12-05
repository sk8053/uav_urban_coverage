import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import tensorflow.keras.backend as K

from tqdm import tqdm

path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)

from mmwchanmod.datasets.download import load_model
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import dir_path_loss, dir_path_loss_multi_sect

import seaborn as heat_map
import numpy as np

class Heat_Map(object):

    def __init__(self, mod_name='uav_beijing', bs_type = 'Terrestrial',  npts=100
                , nsect = 3, cdf_prob = 0.5, net_work_area_hori = np.linspace(0, 100, 20),
                net_work_area_verti = np.linspace(1, 150,20), cmap_name = 'plasma',
                 plan_type = 'xz', plan_shift = 0):
        """
        This class plots a heat map showing strength of SNR values using color bar
        We can observe how SNR values change in different plans (x-y, y-z, x-z)
        for terrestrial and aerial BSs changing antenna configurations: number of sectors and tilted angle

        Parameters
        ----------
        mod_name: model name depending on name of city
        bs_type: Either 'Terrestrial' or 'Aerial'
        npts: number of random channels generated at one point
        nsect: number of sectors for antenna array
        cdf_prob: the probability to take SNR value from its CDF
        net_work_area_hori: horizontal values consisting of network area ( observation plan)
        net_work_area_verti: vertical values consisting of network area (observation plan)
        cmap_name: types of color bar: https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
        plan_type: types of plan where we can observe data
        plan_shift: value for shifting observation plan
        """

        # Paramters
        self.bw = 400e6  # Bandwidth in Hz
        self.nf = 6  # Noise figure in dB
        self.kT = -174  # Thermal noise in dBm/Hz
        self.tx_pow = 23  # TX power in dBm
        self.net_work_area_hori = net_work_area_hori
        self.net_work_area_verti = net_work_area_verti
        self.cmap_name = cmap_name
        self.plan_type = plan_type
        self.plan_shift = plan_shift # value for shifting plane
        fc = 28e9  # carrier frequency in Hz
        nant_gnb = np.array([8, 8])  # gNB array size
        nant_ue = np.array([4, 4])  # UE/UAV array size

        # We downtilt the array and then replicate it over three sectors
        elem_gnb = Elem3GPP(thetabw=65, phibw=65)
        self.arr_gnb0 = URA(elem=elem_gnb, nant=nant_gnb, fc=fc)

        # UE array.  Array is pointing down.
        elem_ue = Elem3GPP(thetabw=65, phibw=65)
        arr_ue0 = URA(elem=elem_ue, nant=nant_ue, fc=fc)
        self.arr_ue = RotatedArray(arr_ue0, theta0=-90)

        # Construct and load the channel model object
        print('Loading pre-trained model %s' % mod_name)
        K.clear_session()
        self.chan_mod = load_model(mod_name)

        # Get types of cell; terrestrial or aerial
        cell_types = self.chan_mod.rx_types
        self.cell_type = np.where(np.array(cell_types) == bs_type)[0]
        self.npts = npts
        self.nsect = nsect
        self.cdf_prob = cdf_prob

    def plot_heat_map(self, bs_type = 'Terrestrial', tilt_angle = 90, nsect = 3, cdf_prob = 0.5):
        """
        this function plot the heat map using a SNR matrix
        Parameters
        ----------
        bs_type : Either Terrestrial or Aerial
        tilt_angle: tilted angles of antenna array

        Returns
        -------

        """
        # update BS type
        cell_types = self.chan_mod.rx_types
        self.cell_type = np.where(np.array(cell_types) == bs_type)[0]
        # update number of sectors for antenna arrays
        self.nsect = nsect
        # update probability of cdf from which we want to observe SNR values
        self.cdf_prob = cdf_prob

        cmap = plt.get_cmap(self.cmap_name)
        x = self.net_work_area_hori
        y = self.net_work_area_verti
        SNR_matrix, link_state_matrix = \
            self.get_snr_from_plane(x, y, tilt_angle= tilt_angle, plan_type=self.plan_type, plan_shift=self.plan_shift)
        SNR_matrix = np.flipud(SNR_matrix.T)

        heat_map.heatmap(SNR_matrix, cmap=cmap, xticklabels=np.round(x, 2), yticklabels=np.flip(np.round(y, 2)))

        plt.xlabel("distance along X axis (m)")
        plt.ylabel("distance along Z axis (m)")

        if self.chan_mod.rx_types[self.cell_type[0]] == 'Terrestrial':
            title_ = 'Terrestrial BS case and terrestrial angle is ' + str(tilt_angle)
        else:
            title_ = 'Aerial BS case and terrestrial angle is ' + str(tilt_angle)
        plt.title(title_)
        plt.savefig ('heat_map.png')

    def get_snr_from_plane(self, x, y, tilt_angle=45,plan_type = 'xz', plan_shift = 0):
        """
        This function will return SNR and link state matrix observed from the given plan
        """
        # x : horizontal values
        # y : vertial values
        # plane_type:   plane where we want to see the heatmap
        s_x, s_y = len(x), len(y)
        SNR_matrix = np.zeros((s_x, s_y))
        link_state_matrix = np.zeros((s_x, s_y))
        i, j = 0,0
        for dx in tqdm(x,position=0, leave=True, desc= 'horizontal axis'):
            j = 0
            for dy in tqdm(y, position=0, leave=True, desc = 'vertical axis'):
                if plan_type == 'xz':
                    SNR, link_state = self.get_snr_from_one_point(tilt_angle, dx, plan_shift, dy)
                elif plan_type == 'yz':
                    SNR, link_state = self.get_snr_from_one_point(tilt_angle,plan_shift, dx,  dy)
                elif plan_type == 'xy':
                    SNR, link_state = self.get_snr_from_one_point(tilt_angle, dx, dy, plan_shift)
                SNR_matrix[i][j] = SNR
                link_state_matrix[i][j] = link_state
                j += 1
            i += 1

        return SNR_matrix, link_state_matrix


    def get_snr_from_one_point(self,tilt_angle, dx,dy,dz):
        """
        This function will obtain SNR value at one point
        Parameters
        ----------
        tilt_angle: tilted angle
        dx : x vector
        dy: y vector
        dz: z vector

        Returns
        -------
        snr: SNR value taking a value with a probability, cdf_prob
        link_state[0]: a link state value representing LOS, NLOS and outage
        """

        npts = self.npts
        dvec = np.repeat([[dx, dy, dz]], npts, axis=0)
        arr_gnb_list_t = multi_sect_array( \
                self.arr_gnb0, sect_type='azimuth', theta0=tilt_angle, nsect=self.nsect)
        cell_type_vec = np.repeat([self.cell_type], npts, axis=0)
        cell_type_vec = cell_type_vec.reshape(-1,)
        snr = np.zeros(npts)
        chan_list, link_state = self.chan_mod.sample_path(dvec, cell_type_vec)
        for i in range(npts):
            # Generate random channels
            chan = chan_list[i]
            pl_gain = dir_path_loss_multi_sect( \
                arr_gnb_list_t, [self.arr_ue], chan)[0]
            # Compute the  SNR
            snri = self.tx_pow - pl_gain - self.kT - self.nf - 10 * np.log10(self.bw)
            snr[i] = snri
        snr = np.sort(snr)
        if self.cdf_prob == 0.5:
            snr = np.median(snr)
        else:
            snr = snr[int(npts*self.cdf_prob)]
        return snr, link_state[0]

'''
f = Heat_Map(mod_name='uav_beijing', bs_type = 'Aerial',
             npts=100, nsect = 3, cdf_prob = 0.5,
             net_work_area_hori = np.linspace(-50, 50, 10),
             net_work_area_verti = np.linspace(-50, 50,10), plan_type='xy', plan_shift=120)
f.plot_heat_map(bs_type="Aerial", tilt_angle= 90)
plt.show()
'''