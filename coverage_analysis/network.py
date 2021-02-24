import numpy as np
import pickle, gzip
import pandas as pd
#from mmchanmod import MmwaveChanel
import matplotlib.pyplot as plt

class ConfigureNetwork():
    # set default argument
    args_default = {'x_max': 1000, 'y_max': 1000,'uav_number': 100}

    def __init__(self, args = args_default):
        self.args = args
        self.x_max = self.args['x_max']
        self.y_max = self.args['y_max']
        self.uav_number = self.args['uav_number']

    def set_deploy_model(self, deployment_model): # there would be other deployment models
        # we set PPP deployment
        self.deployment_model = deployment_model


    def set_channel_model(self, channel_model):
        self.channel_model = channel_model
        self.channel_model.set_tx_rx_antenna()

    def set_tr_BS_UAV(self, lambda_t= 4/(np.pi*200**2)):
        # we will fix the coordinates of UAVs and terrestrial BSs
        # get terrestrial BS coordinates firstly and then obtain the coordinates of UAVs
        UAV_loc_tr,BS_loc_tr \
            = self.deployment_model.get_terrestril_BS(lambda_t)
        # update number of UAV to consider inter-site distance
        self.uav_number = len(UAV_loc_tr[0])
        self.channel_model.set_uav_number(self.uav_number)
        # obtain distance matrix and cell type from the locations of UAVs and terrestrial BSs
        dist_matrix, cell_type, _ =self.get_dist_matrices(UAV_loc =UAV_loc_tr,
                                                       BS_loc =BS_loc_tr,  trOnly = True)
        # then save SNR matrix and data for association only with terrestrial BSs
        self.snr_tr, self.data_for_all_links_tr \
            = self.channel_model.compute_SNR(dist_matrix, cell_type)


    def get_dist_matrices(self, UAV_loc=(0,0,0), BS_loc=(0,0,0), trOnly = False, lambda_a = 0 ):
        # this function returns distance matrices
        if trOnly  == True:
            dist_matrix, cell_type = self.compute_distance_matrix(UAV_loc,BS_loc)
            self.dist_matrix_t = dist_matrix
            uav_indices_to_delete = np.array([])
        else:
            # get the coordinates for aerial BSs
            UAV_loc, BS_loc = self.deployment_model.get_aerial_BS(lambda_a)
            # update number of UAV to consider inter-site distance
            self.uav_number = len (UAV_loc[0])
            # if the number of UAV has changed, we have to save the indices of them
            # because we have to cut SNR matrix and and data matrix corresponding to terrestrial BS later
            uav_indices_to_delete = self.deployment_model.get_uav_to_delete()
            self.channel_model.set_uav_number(self.uav_number)
            # get distance matrices and cell types
            dist_matrix, cell_type = self.compute_distance_matrix(UAV_loc=UAV_loc, BS_loc = BS_loc)

        return dist_matrix, cell_type, uav_indices_to_delete

    def compute_distance_matrix(self, UAV_loc, BS_loc):
        """

        Parameters
        ----------
        UAV_loc = (xx_u, yy_u, zz_u)
        BS_loc = (xx_b, yy_b, zz_b)
        xx_u, yy_u, zz_u : (n_UAVs, ), UAVs' coordinates
        xx_b, yy_b, zz_b: (n_BSs, ), BSs' coordinates

        Returns
        -------
        dist_matrices: (n_link, 3), distance matrix for all links between UAVs and BSs
        cell_type: (n_link, ), cell types for all links between UAVs and BSs
        """
        (xx_u, yy_u, zz_u) = UAV_loc
        (xx_b, yy_b, zz_b) = BS_loc
        height_a_min = self.deployment_model.args['z_a_min']
        cell_type = np.array(np.array(zz_b) < (height_a_min-1) , dtype=int)  # terrestrial is 1 and aerial is 0
        dist_x = np.squeeze(xx_u[:, np.newaxis] - xx_b).reshape(-1, )
        # to implement wrap around, we only find UAVs which would be served
        # by nearest BSs that might be located in different cells
        ind_x = np.where(self.x_max - np.abs(dist_x) <= np.abs(dist_x))

        # change the values as X_max - |dist_x|
        # Also change the sign as the opposite one of original distance vector
        dist_x[ind_x] = (-1) * np.sign(dist_x[ind_x]) * (self.x_max - np.abs(dist_x[ind_x]))
        # in the same way, we will change dist_y
        dist_y = np.squeeze(yy_u[:, np.newaxis] - yy_b).reshape(-1, )
        ind_y = np.where(self.y_max - np.abs(dist_y) <= np.abs(dist_y))
        dist_y[ind_y] = (-1) * np.sign(dist_y[ind_y]) * (self.y_max - np.abs(dist_y[ind_y]))

        zz_u = zz_u[:, np.newaxis]
        dist_z = np.squeeze(zz_u[:, np.newaxis] - zz_b).reshape(-1, )

        # define distance tensors storing distance matrix
        # shape is (# of UAV * # of BS, 3 (= x & y & z ))
        dist_matrices = np.insert(dist_x[:, np.newaxis], 1, dist_y, axis=1)
        dist_matrices = np.insert(dist_matrices, 2, dist_z, axis=1)


        # we find the indices of links that have less distance than minimum distance
        # because we have change the ditance vectors after implementation of warp around
        distance = np.linalg.norm(dist_matrices,axis = -1)
        indices = np.where (distance <=self.deployment_model.args['min_dist'])

        #take big values for distance vectors which are less than minimum distance
        dist_matrices[indices] = np.array([self.x_max, self.y_max, 1000])
        cell_type = np.repeat(cell_type[:, np.newaxis], self.uav_number, axis=1).T
        cell_type = cell_type.reshape(-1, )

        return dist_matrices, cell_type

    def association(self,  tr_only = False, lambda_a = 4/(np.pi*200**2)):
        """

        Parameters
        ----------
        tr_only: True means that we only deploy terrestrial BSs
        lambda_a: deployment rate of aerial BSs

        Returns
        -------
        data_for_connected_links: dictionary variable, data of association links including all channel parameters
        """
        if tr_only == True:
            snr = self.snr_tr
            data_for_all_links = self.data_for_all_links_tr
        else:
            # if aerial BSs are added, SNR matrices should be added as well
            # get SNR matrices and link types for aerial BSs
            dist_matrix_a, cell_type_a, uav_ind_to_delete = self.get_dist_matrices(trOnly= False, lambda_a = lambda_a)
            if len(dist_matrix_a) != 0: # if we generate at least one aerial BS
                # compute SNR and data corresponding to the aerial BSs
                snr_a,  self.data_for_all_links_ar = \
                    self.channel_model.compute_SNR(dist_matrix_a, cell_type_a)
                # if the number of UAV has changed because of minimum-distance consideration
                if len(uav_ind_to_delete) >=1:
                    snr_tr = np.delete(self.snr_tr, uav_ind_to_delete, axis = 0)
                    data_for_all_links_tr = np.delete(self.data_for_all_links_tr, uav_ind_to_delete, axis=0)
                else:
                    snr_tr = self.snr_tr
                    data_for_all_links_tr =self.data_for_all_links_tr

                # add 'snr_a' and data_a to 'snr_tr' and data_tr
                snr = np.vstack ((snr_tr.T,snr_a.T)).T
                data_for_all_links = \
                    np.vstack((data_for_all_links_tr.T, self.data_for_all_links_ar.T)).T
            else: # if there is no aerial BS
                snr = self.snr_tr
                data_for_all_links = self.data_for_all_links_tr

        # UAVs will connect BSs having the highest SNR values
        snr_indices = np.argsort(snr, axis=1)
        # number of aerial BSs connected
        connected_BSs = snr_indices[:, -1]
        # cut data for the associated links
        data_for_connected_links = data_for_all_links[range(len(data_for_all_links)), connected_BSs]
        return data_for_connected_links

    def do_associations(self,scenarios = {'isd_t':200, 'isd_a':[400,200,100]}):
        """
        Compute SNR values of each case
        Return SNR dictionary containing all cases

        Parameters
        ----------
        scenarios: dictionary variable indicating each case
        {'isd_t: value, 'isd_a':values}
        isd_t : inter-site distance of terrestrial BSs
        isd_a: inter-site distance of aerial BSs

        Returns
        -------
        DATA: dictionary variable, {isd_t: link data, isd_a:link_data}
        data for all scenarios depending on inter-site distances
        """

        DATA = dict()
        # reset the number of UAV
        self.deployment_model.set_uav_num(self.args['uav_number'])
        self.channel_model.set_uav_number (self.args['uav_number'])
        # firstly drop UAVs
        self.deployment_model.set_UAV_cooridinates()
        # fix terrestrial BSs and UAVs every dro
        self.set_tr_BS_UAV(lambda_t = 4 /(np.pi* scenarios['isd_t']**2))
        # terrestrial only
        DATA['0'] = self.association(tr_only=True)
        if 'isd_a' in scenarios:
            for isd_a in scenarios['isd_a']:
               DATA[str(isd_a)] \
                    = self.association(tr_only=False, lambda_a=4 / (np.pi * isd_a ** 2))

        return DATA


    # this is the function saving all SNR values as the pikcle files
    def save_SNR_DATA (self,ISD_t, uav_height,  DATA_All, dir):
        for ISD_a in list(DATA_All[0].keys()):
            DATA_sorted = {key: [] for key in list(DATA_All[0]['0'][0].keys())}
            for key in list(DATA_All[0][ISD_a][0].keys()):
                for DATA in DATA_All:
                    for n_link in range (len(DATA[ISD_a])):
                        DATA_sorted[key].append(DATA[ISD_a][n_link][key])
            df = pd.DataFrame(DATA_sorted)
            file_name = dir + 'DATA_ISD_t_' + str(ISD_t) + '_ISD_a_' + ISD_a + '_height_' + str(
                    uav_height) + '.txt'
            df.to_csv(file_name, sep = '\t', index = False)




