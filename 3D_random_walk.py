import random
import matplotlib.pyplot as plt

import numpy as np
from tqdm.auto import tqdm
from UAV_BS import  UAV, BS
from network_channel import Channel_Info, Network

# set some parameters for network area
random.seed(100)
UAV_Height = 50
N_UAV = 10
UAV_Height_MAX = 60
UAV_Height_MIN = 15
N_BS = 8
# the limits of three axis
# running time
run_time = 20


bs_types = np.random.randint(low=0, high=2, size=(N_BS,))
BS_set = [BS(bs_type=bs_types[a], bs_id=a) for a in range(N_BS)]
channel_Info = Channel_Info()
UAV_set = {str(a): UAV(channel_info=channel_Info, UAV_id=a, enable_mobility=True) for a in range(N_UAV)}
# UAV_set['0'].enable_mobility = True
network = Network()
UAV_set, BS_set = network.drop (UAVs = UAV_set, BSs = BS_set)
data_t = {str(t): {'UAV_locations': dict(), 'serving_BS': dict()} for t in range(run_time)}

for t in tqdm(np.arange(run_time), desc= 'simulation time'):
    serving_BS_set = []
    UAV_locations = np.empty(shape=[0, 3])
    for k in np.arange(N_UAV):
        u = str(k)
        # UAVs moves by the number of counter with random velocity
        UAV_set[u].move()
        ## collect the locations for all mobile UAVs
        UAV_locations = np.vstack((UAV_locations, UAV_set[u].get_current_location()))
        serving_BS_set.append(UAV_set[u].serving_BS)

    data_t[str(t)]['UAV_locations'] = UAV_locations
    data_t[str(t)]['serving_BS'] = serving_BS_set

network.visualization(run_time = run_time, data_t = data_t)
plt.show()






