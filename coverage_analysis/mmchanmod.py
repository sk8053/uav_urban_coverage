from mmwchanmod.sim.antenna import Elem3GPP
#from mmwchanmod.learn.models import ChanMod
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss, dir_path_loss_multi_sect
from mmwchanmod.datasets.download import load_model
import tensorflow.keras.backend as K
import numpy as np
import scipy.constants as const_values
from mmwchanmod.common.constants import LinkState
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
class MmwaveChanel():

    args_default = {'uav_number': 100,
            'frequency': 28e9, 'bw': 400e6, 'tx_power': 23, 'bs_elev_t':-12, 'bs_elev_a':45
                    , 'nsect_t':3, 'nsect_a':3, 'thetabw':65, 'phibw':65,
                    'uav_elev':180}

    def __init__(self, args= args_default):
        # set basic paramters
        self.frequency = args ['frequency']
        self.bw =args['bw']  # Bandwidth in Hz
        self.tx_pow = args['tx_power']  # TX power in dBm
        self.nsect_t = args['nsect_t']
        self.nsect_a = args['nsect_a']

        self.uav_number = args['uav_number']
        self.bs_elev_angle_t = args['bs_elev_t']  # downtilt on terrestrial cells
        self.bs_elev_angle_a = args['bs_elev_a']
        self.uav_elev_angle = args['uav_elev']
        self.thetabw = args['thetabw']
        self.phibw = args ['phibw']
        # set constant values
        self.lambda_ = const_values.speed_of_light / self.frequency
        self.pi = const_values.pi
        self.nf = 6  # Noise figure in dB
        self.kT = -174  # Thermal noise in dB/Hz



        ## let's get link state using the trained VAE model
        #model_dir = 'models/uav_beijing'
        self.pl_max = 200.0
        # Construct the channel model object
        K.clear_session()
        self.chan_mod = load_model('uav_boston', src = 'remote')
        self.npath_max = self.chan_mod.npaths_max
        # Load the learned link classifier model
        self.chan_mod.load_link_model()
        # Load the learned path model
        self.chan_mod.load_path_model()
        self.npaths_max = self.chan_mod.npaths_max
        self.arr_gnb_list_a = []
        self.arr_gnb_list_t = []
        self.arr_ue_list = []

    def set_tx_rx_antenna(self):
        # set antenna elements for UAVs and gNBs
        elem_ue = Elem3GPP(thetabw=self.thetabw, phibw=self.phibw )
        elem_gnb_t = Elem3GPP( thetabw=self.thetabw, phibw=self.phibw)
        elem_gnb_a = Elem3GPP(thetabw=self.thetabw, phibw=self.phibw)

        arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8,8]), fc= self.frequency)
        arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([8, 8]), fc=self.frequency)
        arr_ue0 =  URA(elem= elem_ue, nant=np.array([4,4]), fc = self.frequency)

        self.arr_ue_list.append(RotatedArray(arr_ue0, theta0=self.uav_elev_angle, drone = True))
        self.arr_gnb_list_a = multi_sect_array( arr_gnb_a, sect_type= 'azimuth', nsect=self.nsect_a, theta0=self.bs_elev_angle_a )
        self.arr_gnb_list_t = multi_sect_array(arr_gnb_t, sect_type = 'azimuth', nsect= self.nsect_t,theta0=self.bs_elev_angle_t)


    def compute_SNR(self,dist_matrices, cell_type):
        """
        compute SNR
        Parameters
        ----------
        dist_matrices: ( nlink, 3 (= X_Y_Z)) matrix, distance matrix between UAVs and BSs
        cell_type: ( nlink, ) array, cell type that indicates terrestrial or aerial BS

        Returns
        -------
        snr: SNR matrix, ( UAVs number, BSs number ), SNR values for all pairs of UAVs and BSs
        data_for_all_links : dictionry variable including all data such as pathloss, angles, bs type, link state, snr, and so on
        """


        rx_type = np.array(self.chan_mod.rx_types)
        terr_cell = np.argwhere(rx_type == 'Terrestrial')[0]
        aerial_cell = np.argwhere(rx_type == 'Aerial')[0]

        channels, link_types = self.chan_mod.sample_path(dist_matrices, cell_type)
        pl_gain = np.zeros(len(link_types))

        data_for_all_links = np.array([])

        for j, channel in enumerate(channels):
            #if channel.link_state != LinkState.no_link:
            #    channel.pl = np.array([channel.pl[0]])
            #    channel.ang =np.array([channel.ang[0]])

            if cell_type[j]== 1:  # 1 is terrestrial
                arr_gnb_list = self.arr_gnb_list_t
            else:
                arr_gnb_list = self.arr_gnb_list_a
            dist_3d = np.linalg.norm(dist_matrices[j].reshape(1, 3), axis=1)
            # compute free space loss
            fspl = 10 * np.log10(np.power(4 * self.pi * dist_3d / self.lambda_, 2))
            channel.pl [channel.pl<fspl] = fspl # in case that the generated path loss is less than FSPL
            data = dir_path_loss_multi_sect(arr_gnb_list, self.arr_ue_list,channel, long_term_bf=True, instantaneous_bf=False)
            #data = dir_path_loss_multi_sect(arr_gnb_list, self.arr_ue_list, channel, long_term_bf=True,instantaneous_bf=False)

            # below linnes are for logging all parameters per every link
            data['BS_type'] = cell_type[j]
            data['link_state'] = link_types[j]
            # compute SNR for every link and save it in the dictionary for only logging
            data['SNR'] = self.tx_pow - data['pl_eff'] - self.kT - self.nf - 10 * np.log10(self.bw)
            data_for_all_links = np.append(data_for_all_links, data)
            pl_gain[j] = data['pl_eff']

        # we separately save SNRs for associations
        snr = self.tx_pow - pl_gain - self.kT - self.nf - 10 * np.log10(self.bw)
        # we will consider terrestrial cell only case here
        snr = snr.reshape(self.uav_number, -1)
        data_for_all_links = data_for_all_links.reshape(self.uav_number, -1)
        return snr,  data_for_all_links


    def set_uav_number(self, uav_number):
        self.uav_number = uav_number
