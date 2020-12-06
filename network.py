import numpy as np
import pickle, gzip
#from mmchanmod import MmwaveChanel

class ConfigureNetwork():
    # set default argument
    args_default = {'x_max': 1000, 'y_max': 1000,'uav_number': 100}

    def __init__(self, args = args_default):
        self.args = args
        self.x_max = self.args['x_max']
        self.y_max = self.args['y_max']
        self.uav_number = self.args['uav_number']
        ## the below lists is for logging distance vectors and snr values
        self.distance_list = []
        self.snr_list =[]
        self.dist_vector_list = np.empty(shape=[0,3])
    def set_deploy_model(self, deployment_model): # there would be other deployment models
        # we set PPP deployment
        self.deployment_model = deployment_model


    def set_channel_model(self, channel_model):
        self.channel_model = channel_model
        self.link_type_values = self.channel_model.get_link_type_values()

    def set_tr_BS_UAV(self, lambda_t= 4/(np.pi*200**2)):
        # we will fix the coordinates of UAVs and terrestrial BSs
        self.xx_u, self.yy_u, self.zz_u = self.deployment_model.get_UAV_cooridinates()

        self.xx_t, self.yy_t, self.zz_t = self.deployment_model.get_terrestril_BS(lambda_t)
        # update number of UAV to consider inter-site distance
        original_uav_number = self.uav_number
        self.uav_number, uav_indices_to_delete = self.deployment_model.get_uav_num()
        self.tr_number = len(self.xx_t)

        # if number of UAV is changed by deploying aerial BS and cheking inter-site distance,
        # we should update the existing SNR matrix, link_type matrix, and UAV distance vectors
        if len(uav_indices_to_delete) >= 1:
            # update the existing SNR and Link Type matrices
            self.update_network_and_data(uav_indices_to_delete, original_uav_number)

        # then we also save snr matrices between UAVs and terrestrial BSs
        dist_matrix, cell_type =self.get_dist_matrices(trOnly = True)
        self.snr_tr, self.link_type_tr,self.bf_gain_tr = self.channel_model.compute_SNR(dist_matrix, cell_type)

    def get_dist_matrices(self, trOnly = False, lambda_a = 0 ):
        # this function returns distance matrices
        if trOnly  == True:
            dist_matrix, cell_type = self.compute_distance_matrix(self.xx_u, self.yy_u, self.zz_u,
                                                             self.xx_t, self.yy_t, self.zz_t)
            self.dist_matrix_t = dist_matrix
        else:
            # get the coordinates for aerial BSs
            xx_a, yy_a, zz_a = self.deployment_model.get_aerial_BS(lambda_a)
            # update number of UAV to consider inter-site distance
            original_uav_number = self.uav_number
            self.uav_number, uav_indices_to_delete = self.deployment_model.get_uav_num()
            #if number of UAV is changed by deploying aerial BS and cheking inter-site distance,
            # we should update the existing SNR matrix, link_type matrix, and UAV distance vectors
            if len(uav_indices_to_delete)>=1:
                self.update_network_and_data(uav_indices_to_delete, original_uav_number)

            # get distance matrices and cell types
            dist_matrix, cell_type = self.compute_distance_matrix(self.xx_u, self.yy_u, self.zz_u,
                                                                  xx_a, yy_a, zz_a)
        return dist_matrix, cell_type

    def compute_distance_matrix(self, xx_u, yy_u, zz_u, xx_b, yy_b, zz_b):
        """

        Parameters
        ----------
        xx_u, yy_u, zz_u : (n_UAVs, ), UAVs' coordinates
        xx_b, yy_b, zz_b: (n_BSs, ), BSs' coordinates

        Returns
        -------
        dist_matrices: (n_link, 3), distance matrix for all links between UAVs and BSs
        cell_type: (n_link, ), cell types for all links between UAVs and BSs
        """
        height_a_min = self.deployment_model.args['z_a_min']
        cell_type = np.array(np.array(zz_b) < (height_a_min-1) , dtype=int)  # terrestrial is 0 and aerial is 1
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

        #print ('minimum distance',np.min(np.linalg.norm(dist_matrices, axis=1)))
        return dist_matrices, cell_type

    def association(self,  tr_only = True, lambda_a = 4/(np.pi*200**2)):
        """

        Parameters
        ----------
        tr_only: True means that we only deploy terrestrial BSs
        lambda_a: deployment rate of aerial BSs

        Returns
        -------
        SNR: { 'los': snr_los array, 'nlos': snr_nlos array, 'outage': snr_outage array},
        SNR values connected to each UAVs
        """
        bf_gain = self.bf_gain_tr
        if tr_only == True:
            snr = self.snr_tr
            link_type = self.link_type_tr
            #snr_t = snr
            #snr_t = snr_t.reshape(-1,)
            #self.snr_list = np.append(self.snr_list, snr_t)
            #self.dist_vector_list = np.vstack((self.dist_vector_list, self.dist_matrix_t))
            #dvec_snr = np.append(self.dist_matrix_t, snr_t[:, None], axis=1)
            #ind = np.where(link_type == self.link_type_values['los_link'])

            #distance_t = np.linalg.norm(self.dist_matrix_t, axis=1)
            #snr_t = snr.reshape(-1,)
            #self.distance_list = np.append(self.distance_list, self.snr_tr)

            #print(len(self.distance_list))

        else:
            # if aerial BSs are added, SNR matrices should be added as well
            # get SNR matrices and link types for aerial BSs
            dist_matrix_a, cell_type_a = self.get_dist_matrices(trOnly= False, lambda_a = lambda_a)
            if len(dist_matrix_a) != 0:
                snr_a, link_type_a,self.bf_gain_a = self.channel_model.compute_SNR(dist_matrix_a, cell_type_a)
                # add 'snr_a' and 'link_type_a' to 'snr_tr' and 'link_type_tr'
                snr = np.vstack (( self.snr_tr.T,snr_a.T)).T
                link_type_tr = self.link_type_tr.reshape(self.uav_number, -1)
                link_type_a2 = link_type_a.reshape(self.uav_number, -1)
                link_type = np.vstack((link_type_tr.T, link_type_a2.T)).T
                bf_gain = np.append(self.bf_gain_tr, self.bf_gain_a)
            else:
                snr = self.snr_tr
                link_type = self.link_type_tr
                bf_gain = self.bf_gain_tr


        # UAVs will connect BSs having the highest SNR values
        snr_indices = np.argsort(snr, axis=1)
        connected_BSs = snr_indices[:, -1]

        # number of aerial BSs connected
        n_aerial_connected = len(connected_BSs[connected_BSs >= self.tr_number])
        snr = np.sort(snr, axis=1)  # shape ( # of UAV, # of BSs)
        best_snr = snr[:, -1]

        if tr_only is True:
            dist_matrix =self.dist_matrix_t
            dist_matrix = dist_matrix.reshape(self.uav_number, -1, 3)
            dist_matrix = dist_matrix [range(dist_matrix.shape[0]), connected_BSs]
            self.snr_list = np.append(self.snr_list, best_snr)
            self.dist_vector_list = np.vstack((self.dist_vector_list, dist_matrix))

        link_type = link_type.reshape(self.uav_number, -1)  # consider the link types per each UAV
        connected_link_type = link_type[range(len(link_type)), connected_BSs] # take link types for connected BSs

        # let's classify SNR values according to link types
        snr_los = best_snr[np.where(connected_link_type == self.link_type_values['los_link'])]

        snr_nlos = best_snr[np.where(connected_link_type == self.link_type_values['nlos_link'])]
        snr_outage = best_snr[np.where(connected_link_type == self.link_type_values['no_link'])]
        Connected_SNR = {'los':snr_los, 'nlos':snr_nlos, 'outage':snr_outage}

        return Connected_SNR,n_aerial_connected,bf_gain

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
        SNR: dictionary variable, {key = isd_a, value = SNR dictionary including LOS, NLOS and outage}
        n_tr_connected: dictionary variable,  {key = isd_a, value = # of aerial BSs connected}
        """
        SNR = dict()
        n_aerial_connected = dict()

        # reset the number of UAV
        self.deployment_model.set_uav_num(self.args['uav_number'])
        self.channel_model.set_uav_number (self.args['uav_number'])
        # fix terrestrial BSs and UAVs every dro
        self.set_tr_BS_UAV(lambda_t = 4 /(np.pi* scenarios['isd_t']**2))
        # terrestrial only
        snr,_, _ = self.association(tr_only=True)
        SNR['0'] = snr # save terrestrial-only case

        if 'isd_a' in scenarios:
            for isd_a in scenarios['isd_a']:
                SNR[str(isd_a)],n_aerial_connected[str(isd_a)], bf_gain \
                    = self.association(tr_only=False, lambda_a=4 / (np.pi * isd_a ** 2))
                n_aerial_connected[str(isd_a)] /= self.deployment_model.num_uav
        return SNR, n_aerial_connected, bf_gain

    def update_network_and_data(self,uav_indices_to_delete, original_uav_number ):
        # update the existing SNR and Link Type matrices
        self.snr_tr = np.delete(self.snr_tr, uav_indices_to_delete, axis=0)
        self.link_type_tr = np.delete(self.link_type_tr.reshape(original_uav_number, -1),
                                      uav_indices_to_delete, axis=0)
        self.link_type_tr = self.link_type_tr.reshape(-1, )
        # update UAV locations
        self.xx_u, self.yy_u, self.zz_u = self.deployment_model.update_UAVs()
        # update UAV number in channel class
        self.channel_model.set_uav_number(self.uav_number)
    def save_text_file(self):
        # print (SNR_list[0])
        #data = np.column_stack((self.distance_list, self.snr_list))
        #print (self.dist_vector_list)
        #print (self.snr_list)
        dvec_snr = np.append(self.dist_vector_list, self.snr_list[:, None], axis=1)
        np.savetxt('distance_vector_snr_' + str(self.deployment_model.args['uav_height']) + '.txt', dvec_snr)

    # this is the function saving all SNR values as the pikcle files
    def save_SNR (self,ISD_t, uav_height, SNR_list,connected_ar,  dir):
        #print (SNR_list[0])


        for ISD_a in list(SNR_list[0].keys()):
            SNR_los, SNR_nlos, SNR_outage = np.array([]), np.array([]), np.array([])

            for SNR in SNR_list:
                SNR_los = np.append(SNR_los, SNR[ISD_a]['los'])
                SNR_nlos = np.append(SNR_nlos, SNR[ISD_a]['nlos'])
                SNR_outage = np.append(SNR_outage, SNR[ISD_a]['outage'])

            if ISD_a == '0':
                data = {'los': SNR_los, 'nlos': SNR_nlos, 'outage': SNR_outage}
            else:
                data = {'los': SNR_los, 'nlos': SNR_nlos, 'outage': SNR_outage, 'connected_ar': connected_ar[ISD_a]}

            file_name = dir + 'SNR_ISD_t_' + str(ISD_t) + '_ISD_a_' + ISD_a + '_height_' + str(uav_height) + '.p'

            f = gzip.open(file_name, 'wb')
            pickle.dump(data, f)