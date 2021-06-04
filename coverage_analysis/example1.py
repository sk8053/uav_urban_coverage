from tqdm import tqdm
import numpy as np
import sys
sys.path.append('../')
from network import ConfigureNetwork
import argparse
from mmchanmod import MmwaveChanel
from ppp_deployment import PPP_deployment
from collections import Counter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')

parser.add_argument('--height',action='store',default=30,type= int,\
    help='uav_height')

parser.add_argument('--isd_t',action='store',default=200,type= int,\
    help='inter site distance for terrestrial cells')

parser.add_argument('--isd_a', action = 'store', default=400, type = int,\
                    help = 'inter site distance for aerial cells')

parser.add_argument('--n_uav', action = 'store', default=1000, type = int,\
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
channel_args ={'uav_number': 100,
            'frequency': 28e9, 'bw': 400e6, 'tx_power': 23, 'bs_elev_t':-12,
               'bs_elev_a':45, 'uav_elev':-90 , 'nsect_t':3, 'nsect_a':3,
               'thetabw':65, 'phibw':65, 'uav_number':uav_number}
# deployment parameters
deploy_args = {'x_max': 1000,'x_min':0,  'y_max': 1000, 'y_min':0,
               'z_t_min': 2.0,'z_t_max': 5.0, 'z_a_min':10, 'z_a_max': 30,
                'uav_height': uav_height, 'uav_number': uav_number,
               'lambda_t': 4/(np.pi*isd_t**2), 'lambda_a': 0.0, 'min_dist':10.0 }

# configure network using parameters
net = ConfigureNetwork(net_args)
# set channel model
channel_model = MmwaveChanel(channel_args)
# set deployment model
deployment_model = PPP_deployment(deploy_args)
# install channel and deployment models into the configured network model
net.set_channel_model(channel_model)
net.set_deploy_model(deployment_model)

print ('UAV height: ', uav_height)
print ("UAV number:", uav_number)
# set cases that we want to know

scenarios = {'isd_t':200, 'isd_a':[800, 400, 200]}

SNR_list=[]
DATA_All= []
# do multiple drops
connected_ar = dict()
for iteration in tqdm(range (max_iter), desc='# of iterations'):
    # do association for the given cases
    DATA= net.do_associations(scenarios)
    DATA_All.append(DATA)


# save the SNR values for each case as pickle files
ISD_t = scenarios['isd_t']
#print (DATA_All)
dir = '../data/'
net.save_SNR_DATA(ISD_t, uav_height,DATA_All, dir)

#deployment_model.plot()
#plt.show()
