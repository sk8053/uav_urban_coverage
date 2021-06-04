
import matplotlib.pyplot as plt
import os
import sys
import argparse

import tensorflow.keras.backend as K

from tqdm import tqdm

path = os.path.abspath('../')
if not path in sys.path:
    sys.path.append(path)

from mmwchanmod.datasets.download import load_model
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
from mmwchanmod.sim.chanmod import dir_path_loss, dir_path_loss_multi_sect

import seaborn as h_map
import numpy as np
import matplotlib.ticker as ticker

class Heat_Map(object):

    def __init__(self, mod_name='uav_beijing', bs_type = 'Terrestrial',  npts=100
                , nsect = 3, cdf_prob = 0.5, horizontal_axis_of_net = np.linspace(0, 100, 20),
                vertical_axis_of_net = np.linspace(1, 150,20), cmap_name = 'plasma',
                 plane_type = 'xz', plane_shift = 0):
        """
        This class plots a heat map showing strength of SNR values using color bar
        We can observe how SNR values change in different planes (x-y, y-z, x-z)
        for terrestrial and aerial BSs changing antenna configurations: number of sectors and tilted angle

        Parameters
        ----------
        mod_name: model name depending on name of city
        bs_type: Either 'Terrestrial' or 'Aerial'
        npts: number of random channels generated at one point
        nsect: number of sectors for antenna array
        cdf_prob: the probability to take SNR value from its CDF
        net_work_area_hori: horizontal values consisting of network area ( observation plane)
        net_work_area_verti: vertical values consisting of network area (observation plane)
        cmap_name: types of color bar: https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html
        plane_type: types of plane where we can observe data
        plan_shift: value for shifting observation plane
        """

        # Paramters
        self.bw = 400e6  # Bandwidth in Hz
        self.nf = 6  # Noise figure in dB
        self.kT = -174  # Thermal noise in dBm/Hz
        self.tx_pow = 23  # TX power in dBm
        self.horizontal_axis_of_net = horizontal_axis_of_net
        self.vertical_axis_of_net = vertical_axis_of_net
        self.cmap_name = cmap_name
        self.plane_type = plane_type
        self.plane_shift = plane_shift # value for shifting plane
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
        #print('Loading pre-trained model %s' % mod_name)
        K.clear_session()
        self.chan_mod = load_model(mod_name, src = 'remote')

        # Get types of cell; terrestrial or aerial
        cell_types = self.chan_mod.rx_types
        self.cell_type = np.where(np.array(cell_types) == bs_type)[0]
        self.npts = npts
        self.nsect = nsect
        self.cdf_prob = cdf_prob

        self.vmin = -10
        self.vmax = 60

    def plot_heat_map(self, bs_type = 'Terrestrial', tilt_angle = 45, nsect = 3, cdf_prob = 0.5, annot= False
                      , get_link_state = False, plane_type = 'xz', disable_plot = False, bs_height=0):
        """
        this function plot the heat map using a SNR matrix
        Parameters
        ----------
        bs_type : Either Terrestrial or Aerial
        tilt_angle: tilted angles of antenna array
        nsect: number sectors for BSs
        cdf_prob: probability to take SNR value from its CDF
        annot: if Ture, write data values in each cell of heatmap
        get_link_state: if True, we will see link state heat map with values or symbols in each cell
        disable_plot: if True, we only get data and don't plot heatmap
        bs_height: the height of BS, the value of z axis
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
        # update plane type
        self.plane_type = plane_type

        cmap = plt.get_cmap(self.cmap_name)
        x = self.horizontal_axis_of_net
        if plane_type == 'xz' or plane_type == 'yz':
            y = self.vertical_axis_of_net - bs_height
            plane_shift = self.plane_shift
        elif plane_type == 'xy':
            y = self.vertical_axis_of_net
            plane_shift = self.plane_shift - bs_height

        SNR_matrix, link_state_matrix, rx_gain_matrix = \
            self.get_snr_from_plane(x, y, tilt_angle= tilt_angle, plane_type=self.plane_type, plane_shift=plane_shift)
        SNR_matrix = np.flipud(SNR_matrix)
        fig, ax = plt.subplots()
        if disable_plot is not True:
            rx_gain_matrix = np.flipud(rx_gain_matrix)
            s = h_map.heatmap(rx_gain_matrix, cmap=cmap, xticklabels=np.round(self.horizontal_axis_of_net, 2),
                              yticklabels = np.flip(np.round(self.vertical_axis_of_net, 2)),
                            annot = annot, cbar = not annot)



            if self.chan_mod.rx_types[self.cell_type[0]] == 'Terrestrial':
                title_ = 'Terrestrial BS case and tilted angle is ' + str(tilt_angle)
                file_name = 'SNR of terrestrial BS HeatMap '+ str(tilt_angle) + ' n_sector = ' + str(nsect)
            else:
                title_ = 'Aerial BS case and tilted angle is ' + str(tilt_angle)
                file_name = 'SNR of aerial BS HeatMap ' + str(tilt_angle) + ' n_sector = ' + str(nsect)

            #plt.title(title_)
            plt.savefig (file_name+'.png')


            plt.xlabel("distance along "+ list(plane_type)[0]+ " axis (m)")
            plt.ylabel("distance along "+ list(plane_type)[1]+ " axis (m)")
           # plt.title ("AOD STD")

            plt.figure(50)
            ss = rx_gain_matrix.reshape(-1,)
            plt.plot(np.sort(ss), np.linspace(0,1,len(ss)))
            plt.grid()

            if get_link_state is True:
                plt.figure (30)
                link_state_matrix = np.flipud(link_state_matrix)

                annot_ = np.empty_like(link_state_matrix)
                annot_ = np.array (annot_, dtype = object)
                annot_[link_state_matrix == 1] = 'L'
                annot_ [link_state_matrix ==3] = 'LL'
                annot_[link_state_matrix == 2] = 'N'
                annot_[link_state_matrix == 0] = 'O'

                h_map.heatmap(link_state_matrix, cmap=cmap, xticklabels=np.round(x, 2), yticklabels=np.flip(np.round(y, 2)),
                             annot= annot_, cbar =False, fmt = '')
                plt.xlabel("distance along " + list(plane_type)[0] + " axis (m)")
                plt.ylabel("distance along " + list(plane_type)[1] + " axis (m)")

                if self.chan_mod.rx_types[self.cell_type[0]] == 'Terrestrial':
                    title_ = 'Link state of terrestrial BS case and tilted angle is ' + str(tilt_angle)
                    file_name = 'link_state_terrestrial BS heatmap ' + str(tilt_angle) + ' n_sector = ' + str(nsect)
                else:
                    title_ = 'Link state of aerial BS case and tilted angle is ' + str(tilt_angle)
                    file_name = 'link_state_aerial BS heatmap ' + str(tilt_angle) + ' n_sector = ' + str(nsect)

                plt.title(title_)
                plt.savefig(file_name + '.png')
            return rx_gain_matrix
        else:
            return SNR_matrix, link_state_matrix, rx_gain_matrix


    def get_snr_from_plane(self, x, y, tilt_angle=45,plane_type = 'xz', plane_shift = 0):
        """
        This function will return SNR and link state matrix observed from the given plane
        """
        # x : horizontal values
        # y : vertial values
        # plane_type:   plane where we want to see the heatmap
        xx,yy = np.meshgrid(x,y)
        xx = xx.reshape(-1,)
        yy = yy.reshape (-1,)
        SNR_matrix = []
        link_state_matrix = []
        rx_gain_matrix = []
        for dx, dy in tqdm(zip (xx,yy), total= len(xx), desc='total number of bins'): #, position= 0, leave= False):
            if plane_type == 'xz':
                SNR, link_state, rx_e_gain = self.get_snr_from_one_point(tilt_angle, dx, plane_shift, dy)
            elif plane_type == 'yz':
                SNR, link_state, rx_e_gain = self.get_snr_from_one_point(tilt_angle,plane_shift, dx,  dy)
            elif plane_type == 'xy':
                SNR, link_state, rx_e_gain = self.get_snr_from_one_point(tilt_angle, dx, dy, plane_shift)
            SNR_matrix.append(SNR)
            link_state_matrix.append(link_state)
            rx_gain_matrix.append(rx_e_gain)
        SNR_matrix = np.array(SNR_matrix).reshape(len(y), len(x))
        link_state_matrix = np.array(link_state_matrix).reshape(len(y), len(x))
        rx_gain_matrix = np.array(rx_gain_matrix).reshape(len(y), len(x))

        return SNR_matrix, link_state_matrix, rx_gain_matrix


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
        arr_gnb_list = multi_sect_array( \
                self.arr_gnb0, sect_type='azimuth', theta0=tilt_angle, nsect=self.nsect)
        cell_type_vec = np.repeat([self.cell_type], npts, axis=0)
        cell_type_vec = cell_type_vec.reshape(-1,)
        snr = np.zeros(npts)
        chan_list, link_state = self.chan_mod.sample_path(dvec, cell_type_vec)
        aod_thetas, aoa_thetas = np.array([]), np.array([])

        pl_gain_sum = 0
        link_types = np.array([])
        for i in range(npts):
            # Generate random channels
            chan = chan_list[i]
            data = dir_path_loss_multi_sect( \
                [self.arr_ue],arr_gnb_list,  chan)
            pl_gain = data['pl_eff']
            aod_theta = data['aod_theta']
            aoa_theta = data['aoa_theta']

            pl_gain_sum += 10 ** (-0.1 * (pl_gain))

            aod_thetas = np.append(aod_thetas, np.std(aod_theta))
            aoa_thetas = np.append(aoa_thetas, np.std(aoa_theta))
            # Compute the  SNR
            snri = self.tx_pow - pl_gain - self.kT - self.nf - 10 * np.log10(self.bw)
            snr[i] = snri
        pl_gain_sum /=npts

        pl_gain_sum = 10*np.log10(pl_gain_sum)

        snr_avg = self.tx_pow - pl_gain_sum - self.kT - self.nf - 10 * np.log10(self.bw)

        aoa_theta_return = np.median(aoa_thetas)

        return snr_avg, [1],aoa_theta_return#np.median(aoa_thetas)

    def get_association(self, annot = True, tilt_angel_t = -12, tilt_angle_a= 45,
                        bs_height_a =10, bs_height_t =0, plane_type = 'xy'):
        print ('Get data from UAVs for terrestrial BS')

        SNR_matrix_t, link_state_t, rx_gain_t = self.plot_heat_map(bs_type="Terrestrial", tilt_angle= tilt_angel_t,
                                                        plane_type=plane_type, nsect=3, cdf_prob=self.cdf_prob,
                                                        disable_plot=True, bs_height=bs_height_t)
        print ('Get data from UAVs for aerial BS')
        SNR_matrix_a, link_state_a, rx_gain_a = self.plot_heat_map(bs_type="Aerial", tilt_angle=tilt_angle_a, plane_type=plane_type,
                                                        nsect=3, cdf_prob=self.cdf_prob, disable_plot=True, bs_height= bs_height_a)
        maximum_SNR = np.maximum(SNR_matrix_a, SNR_matrix_t)
        associatioin_matrix = np.array((maximum_SNR==SNR_matrix_a), dtype= int)
        x = self.horizontal_axis_of_net
        y = self.vertical_axis_of_net

        annot_ = np.empty_like(associatioin_matrix)
        annot_ = np.array (annot_, dtype= object)
        annot_[associatioin_matrix == 1] = 'a'
        annot_[associatioin_matrix == 0] = 't'

        plt.figure(1)
        h_map.heatmap(associatioin_matrix, cmap=self.cmap_name, xticklabels=np.round(x, 2), yticklabels=np.flip(np.round(y, 2)),
                         annot=annot_, cbar=not annot, fmt = '')
        plt.xlabel("distance along " + list(self.plane_type)[0] + " axis (m)")
        plt.ylabel("distance along " + list(self.plane_type)[1] + " axis (m)")
        plt.title ('UAVs in bins after selecting either terrestrial or aerial BS')

        plt.figure (2)
        h_map.heatmap(SNR_matrix_a, cmap=self.cmap_name, xticklabels=np.round(x, 2),
                      yticklabels=np.flip(np.round(y, 2)),
                      annot=annot, cbar=not annot)
        plt.xlabel("distance along " + list(self.plane_type)[0] + " axis (m)")
        plt.ylabel("distance along " + list(self.plane_type)[1] + " axis (m)")
        plt.title('UAVs in bins associated with only aerial BS')

        plt.figure(3)
        h_map.heatmap(SNR_matrix_t, cmap=self.cmap_name, xticklabels=np.round(x, 2),
                      yticklabels=np.flip(np.round(y, 2)),
                      annot=annot, cbar=not annot)
        plt.xlabel("distance along " + list(self.plane_type)[0] + " axis (m)")
        plt.ylabel("distance along " + list(self.plane_type)[1] + " axis (m)")
        plt.title('UAVs in bins associated with only terrestrial BS')
        return  rx_gain_t, rx_gain_a

