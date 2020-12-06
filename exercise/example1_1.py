from tqdm import tqdm
import numpy as np
from network import ConfigureNetwork
import argparse

from mmchanmod import MmwaveChanel

from ppp_deployment import PPP_deployment
import multiprocessing as process
import pickle, gzip

parser = argparse.ArgumentParser(description='')

parser.add_argument('--height',action='store',default=30,type= int,\
    help='uav_height')

parser.add_argument('--isd_t',action='store',default=200,type= int,\
    help='inter site distance for terrestrial cells')

parser.add_argument('--isd_a', action = 'store', default=400, type = int,\
                    help = 'inter site distance for aerial cells')

parser.add_argument('--n_uav', action = 'store', default=100, type = int,\
                    help = 'number of UAVs deployed')

parser.add_argument('--max_iter', action = 'store', default=1000, type = int,\
                    help = 'number of drops to be randomized')

args = parser.parse_args()
uav_height = args.height
uav_number = args.n_uav
max_iter = args.max_iter

isd_t = args.isd_t   # Inter-site distance of terrestrial BSs
isd_a = args.isd_a   # Inter-site distance of aerial BSs (if selected)

# define parameters necessary for simulation

# network parameters
net_args = {'x_max': 1000, 'y_max': 1000, 'uav_number':uav_number}
# channel parameters
channel_args = {'uav_number':uav_number,
                'frequency': 28e9, 'bw':400e6, 'tx_power':23, 'downtilt':10 }
# deployment parameters
deploy_args = {'x_max': 1000,'x_min':0,  'y_max': 1000, 'y_min':0,
               'z_t_min': 2.0,'z_t_max': 5.0, 'z_a_min':30, 'z_a_max': 60,
                'uav_height': uav_height, 'uav_number': uav_number,
               'lambda_t': 4/(np.pi*isd_t**2), 'lambda_a': 0.0, 'min_dist':10.0 }



print ('UAV height: ', uav_height)

# set cases that we want to know
scenarios = {'isd_t':200, 'isd_a':[400,200,100]}


# do multiple drops
def Iteration (n):
    # configure network using parameters
    net = ConfigureNetwork(net_args)
    # set channel model
    channel_model = MmwaveChanel(channel_args)
    # set deployment model
    deployment_model = PPP_deployment(deploy_args)
    # install channel and deployment models into the configured network model
    net.set_channel_model(channel_model)
    net.set_deploy_model(deployment_model)
    SNR_list = []
    for iteration in tqdm(range (2500), desc='# of iterations'):
        # do association for the given cases
        SNR,_ = net.do_associations(scenarios)
        SNR_list.append(SNR)
    f = gzip.open('test_file_' + str(n) + '.p', 'wb')
    pickle.dump(SNR_list, f)


n_prc = 2
pool = process.Pool(processes = n_prc)
pool.map(Iteration, [j+1 for j in range(n_prc)])
pool.close()
pool.join()

# configure network using parameters
net = ConfigureNetwork(net_args)
# set channel model
channel_model = MmwaveChanel(channel_args)
# set deployment model
deployment_model = PPP_deployment(deploy_args)
# install channel and deployment models into the configured network model
net.set_channel_model(channel_model)
net.set_deploy_model(deployment_model)
SNR_list = np.array([])
for n in range(n_prc):
    f = gzip.open('test_file_'+str(n+1)+'.p')
    snr_list = pickle.load(f)
    SNR_list = np.append(SNR_list, snr_list)

# save the SNR values for each case as pickle files
ISD_t = scenarios['isd_t']
dir = 'snr_data/'
net.save_SNR(ISD_t, uav_height, SNR_list, dir)
