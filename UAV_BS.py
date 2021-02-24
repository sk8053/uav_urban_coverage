import numpy as np
import scipy.stats
import scipy.constants
from mmwchanmod.sim.antenna import Elem3GPP
from mmwchanmod.sim.array import URA, RotatedArray, multi_sect_array

import random
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss_multi_sect
from mmwchanmod.common.constants import LinkState

class UAV(object):
    """
    Class for implementing UAV
    """
    def __init__(self, bs_info=None, channel_info=None, UAV_id=0,UAV_Heights =[10,30], enable_mobility=True):

        self.UAV_Height_MAX = UAV_Heights[1]
        self.UAV_Height_MIN = UAV_Heights[0]
        thetabw_, phibw_ = 65, 65
        self.time_keeping_velocity = 0
        self.velocity = [0, 0, 0]
        self.enable_mobility = enable_mobility
        self.UAV_id = UAV_id

        frequency = channel_info.frequency
        self.lambda_ = scipy.constants.speed_of_light / frequency
        self.SINRs = []
        self.channel_info = channel_info

        element_ue = Elem3GPP(thetabw=thetabw_, phibw=phibw_)

        arr_ue = URA(elem=element_ue, nant=np.array([4, 4]), fc=frequency)
        self.arr_ue = RotatedArray(arr_ue, theta0=-90)
        self.wrx = 0
        self.rx_sv = dict()
        self.bs_locations = np.empty(shape=[0, 3])
        self.SINR_servingBS = -20

        # obtain the information on all BSs
        self.cell_types = np.array([])
        self.arr_gnb_list = []


        ## intial connection with a random BS
        self.serving_BS = -1
        # set moving parameters
        self.Maximum_Velocity_UAVs = 10 # maximum velcoity of UAVs
        # maximum and minimum time to change velocity for random walk
        self.MAXIMUM_TIME = 80
        self.MINIMUM_TIME = 10

    def set_BS_info(self, bs_info):
        self.BS_set = bs_info
        for bs in bs_info:
            self.bs_locations = np.vstack((self.bs_locations, bs.get_location()))  # collect the location information
            self.cell_types = np.append(self.cell_types, bs.bs_type)  # get BS types, terrestrial or aerial
            self.arr_gnb_list.append(bs.get_antenna_array())  # get antenna arrays for all BSs

        # self._association()
    def set_locations(self, coordinates={'X_MAX':100, 'X_MIN':-100, 'Y_MAX':100, 'Y_MIN':-100,
                                       'Z_MAX':60, 'Z_MIN':0}):

        self.X_MAX, self.Y_MAX, self.Z_MAX = coordinates['X_MAX'], coordinates['Y_MAX'], coordinates['Z_MAX']
        self.X_MIN, self.Y_MIN, self.Z_MIN = coordinates['X_MIN'], coordinates['Y_MIN'], coordinates['Z_MIN']
        trajectory_2D = np.array((self.X_MAX - self.X_MIN) * scipy.stats.uniform.rvs(0, 1, ((1, 2))) + self.X_MIN)
        # uav_height = (UAV_Height_MAX-UAV_Height_MIN)* scipy.stats.uniform.rvs(0,1, ((1,))) + UAV_Height_MIN
        uav_height = np.array([self.UAV_Height_MAX])
        self.trajectory = np.append(trajectory_2D, uav_height[:, None], axis=1)

    def update_velocity(self):
        r = self.Maximum_Velocity_UAVs * scipy.stats.uniform.rvs(0, 1, (1,))
        theta = np.pi * scipy.stats.uniform.rvs(0, 2, (1,)) - np.pi
        x, y = r * np.cos(theta), r * np.sin(theta)
        z = scipy.stats.uniform.rvs(-1, 1, (1,))
        self.velocity = [x[0], y[0], 0]  # [x[0],y[0],z[0]]

    def _next_location(self):
        return self.trajectory[-1] + self.velocity

    def _record_trajectory(self, location):
        self.trajectory = np.append(self.trajectory, location[None, :], axis=0)

    def _association(self):

        # first, get channels for all BSs
        dist_vectors = self.get_current_location() - self.bs_locations
        channels, link_types = self.channel_info.channel.sample_path(dist_vectors, self.cell_types)
        pl_gain = np.zeros(len(link_types))
        wrx_s = dict()
        rx_sv_s, tx_sv_s = dict(), dict()
        # then compute path loss, channel matrices, and beamforming vectors for all links
        for j, channel in enumerate(channels):
            data, bf_channel = dir_path_loss_multi_sect(self.arr_gnb_list[j], [self.arr_ue], channel, bf_ch_return = True)
            pl_gain[j] = data['pl_eff']

            bs_id = str(j)
            rx_sv_s[bs_id] = bf_channel['rx_sv']
            tx_sv_s[bs_id] = bf_channel['tx_sv']
            wrx_s[bs_id] = bf_channel['wrx']
            if channel.link_state != LinkState.no_link:
                self.BS_set[j].set_beamforming_vector(self.UAV_id, bf_channel['wtx'])

        # we calculate the interference
        interference = np.array([])
        for serv_id, serv_BS in enumerate(self.BS_set):  # take one serving BS for a UAV
            inter_i = 0.0
            if channels[serv_id].link_state != LinkState.no_link:  # do not consider if no link
                wrx = wrx_s[str(serv_id)]
                for ind, bs in enumerate(self.BS_set):  # consider other BSs for one serving BSs
                    if ind != serv_id and channels[ind].link_state != LinkState.no_link:
                        # let's choose a transmit beamforming vector of BSs directed to a different UAV
                        wtx = self.BS_set[ind].get_random_beamforming_vector(self.UAV_id)
                        # get the channel between UAV and the BS that causes interference
                        tx_sv, rx_sv = tx_sv_s[str(ind)], rx_sv_s[str(ind)]
                        # compute the beamforming gain between them
                        tx_bf = np.abs(tx_sv.dot(wtx)) ** 2
                        rx_bf = np.abs(rx_sv.dot(wrx)) ** 2
                        # choose only one path having minimum path loss
                        opt_p = np.argmin(pl_gain[ind])
                        # covert the dB-scale to linear scale
                        pl_lin = 10 ** (0.1 * channels[ind].pl)  # = TX/RX
                        itf_gain = tx_bf[opt_p] * rx_bf[opt_p] * 10 ** (0.1 * self.channel_info.TX_power) / pl_lin[
                            opt_p]  # = TX/TX/RX = RX
                        # accumulate all interference values in linear-scale
                        inter_i += itf_gain

            if inter_i != 0:
                interference = np.append(interference, inter_i)
            else:  # if there is no link
                interference = np.append(interference, 0)
        # then add interference-power to noise power in linear scale
        KT_lin, NF_lin = 10 ** (0.1 * self.channel_info.KT), 10 ** (0.1 * self.channel_info.NF)
        Noise_Interference = KT_lin * NF_lin * self.channel_info.BW + interference
        # obtain SINR in dB-scale
        SINR = self.channel_info.TX_power - pl_gain - 10 * np.log10(Noise_Interference)
        # choose maximum SINR for association
        MAX_SINR = np.max(SINR)
        if MAX_SINR - self.SINR_servingBS > 3:  # if new SINR is larger than SINR of previous serving BS
            self.SINR_servingBS = MAX_SINR  # update the SINR value
            if self.serving_BS != -1:  # if this is not initial connetion
                self.BS_set[self.serving_BS].disconnect_to_UAV(self.UAV_id)  # disconnect with the previous serving BS
            self.serving_BS = np.argmax(SINR)  # save the ID of new serving BS
            self.BS_set[self.serving_BS].association_with_UAV(self.UAV_id)  # do association with new serving BS

        # fspl = 10 * np.log10(np.power(4*np.pi*min_distance/lambda_,2 ))
        self.SINRs.append(MAX_SINR)  # record all SINR value over simulation time

    def move(self):
        """
        this function implements the random movement of UAV in 3-D space
        """

        if self.time_keeping_velocity <= 0:
            # set the time to keep a certain velocity chosen randomly
            self.time_keeping_velocity = random.randrange(start=self.MINIMUM_TIME, stop=self.MAXIMUM_TIME)
            self.update_velocity()  # update a velocity by choosing it randomly
        else:
            self.time_keeping_velocity -= 1

        if self.enable_mobility == False:
            self.velocity = [0, 0, 0]

        next_location = np.squeeze(self._next_location())
        # check wheather UAVs locations exceed the vertial or horizontal boundaries
        if next_location[0] < self.X_MIN or next_location[0] > self.X_MAX:  # if an UAV goes beyond the horizontal boundaries.
            self.velocity[0] = - self.velocity[0]  # change the sign of horizontal velocity component
            next_location = np.squeeze(self._next_location())  # return original place

        elif next_location[1] < self.Y_MIN or next_location[1] > self.Y_MAX:  # if an UAV goes beyond the vertical boundaries.
            self.velocity[1] = - self.velocity[1]  # change the sign of vertial velocity component
            next_location = np.squeeze(self._next_location())  # return original place

        elif next_location[2] < self.UAV_Height_MIN or next_location[2] > self.Z_MAX:  # if an UAV goes beyond the vertical boundaries.
            self.velocity[2] = - self.velocity[2]  # change the sign of vertial velocity component
            next_location = np.squeeze(self._next_location())  # return original place

        self._record_trajectory(next_location)  # update trajectory of UAV
        self._association()  # do association

    def get_current_location(self):
        return self.trajectory[-1]

class BS(object):
    def __init__(self, bs_type=1, bs_id=0):
        # 1 is terrestrial, 0 is aerial
        thetabw, phibw = 65, 65
        n_sect = 3
        self.Tr_BS_Height_MIN = 5
        self.Tr_BS_Height_MAX = 10
        self.Ar_BS_Height_MIN = 15
        self.Ar_BS_Height_MAX = 30
        self.bs_type = bs_type
        frequency = 28e9
        self.bs_type = bs_type
        self.bs_id = bs_id
        elem_gnb_t = Elem3GPP(thetabw=thetabw, phibw=phibw)
        elem_gnb_a = Elem3GPP(thetabw=thetabw, phibw=phibw)

        arr_gnb_t = URA(elem=elem_gnb_t, nant=np.array([8, 8]), fc=frequency)
        arr_gnb_a = URA(elem=elem_gnb_a, nant=np.array([8, 8]), fc=frequency)

        arr_gnb_list_a = multi_sect_array(arr_gnb_a, sect_type='azimuth', nsect=n_sect, theta0=45)
        arr_gnb_list_t = multi_sect_array(arr_gnb_t, sect_type='azimuth', nsect=n_sect, theta0=-12)

        if bs_type == 1:
            self.arr_gnb_list = arr_gnb_list_t
        else:
            self.arr_gnb_list = arr_gnb_list_a


        self.wtx = dict()
        self.connected_UAVs = np.array([])

    def set_locations(self, coordinates={'X_MAX':100, 'X_MIN':-100, 'Y_MAX':100, 'Y_MIN':-100,
                                       'Z_MAX':60, 'Z_MIN':0}):

        self.X_MAX, self.Y_MAX, self.Z_MAX = coordinates['X_MAX'], coordinates['Y_MAX'], coordinates['Z_MAX']
        self.X_MIN, self.Y_MIN, self.Z_MIN = coordinates['X_MIN'], coordinates['Y_MIN'], coordinates['Z_MIN']
        init_loc = (self.X_MAX - self.X_MIN) * scipy.stats.uniform.rvs(0, 1, ((1, 2))) + self.X_MIN

        if self.bs_type == 1:  # if bs is terrestrial
            BS_Height_MAX, BS_Height_MIN = self.Tr_BS_Height_MAX, self.Tr_BS_Height_MIN
        else:
            BS_Height_MAX, BS_Height_MIN = self.Ar_BS_Height_MAX, self.Ar_BS_Height_MIN

        init_loc_z = (BS_Height_MAX - BS_Height_MIN) * scipy.stats.uniform.rvs(0, 1, ((1,))) + BS_Height_MIN
        # init_loc_z = np.array ([BS_Height_MAX])
        self.locations = np.append(init_loc, init_loc_z[:, None], axis=1)


    def association_with_UAV(self, uav_id):
        self.connected_UAVs = np.append(self.connected_UAVs, uav_id)

    def disconnect_to_UAV(self, uav_id):
        index = np.where(self.connected_UAVs == uav_id)[0]
        self.connected_UAVs = np.delete(self.connected_UAVs, index)

    def get_location(self):
        return np.squeeze(self.locations)

    def get_antenna_array(self):
        return self.arr_gnb_list

    def set_beamforming_vector(self, uav_id, wtx):
        self.wtx[str(uav_id)] = wtx

    def get_random_beamforming_vector(self, uav_id):
        # if a BS has no UAV connected with it
        if len(self.connected_UAVs) == 0:
            return np.zeros(shape=(64,))
        # choose a beamforming vector directed to a connected UAV
        else:
            ind = int(np.random.choice(self.connected_UAVs, 1)[0])  # choose random w_tx toward a random UAV connected
            return self.wtx[str(ind)]
