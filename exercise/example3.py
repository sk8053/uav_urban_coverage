from tqdm import tqdm
import numpy as np
from network import ConfigureNetwork
import argparse
from mmchanmod import MmwaveChanel
from ppp_deployment import PPP_deployment
from collections import Counter
from mmwchanmod.datasets.download import get_dataset
import pickle, gzip

#d= get_dataset(ds_name='uav_boston', src = 'remote')

f = open('/home/sk8053/mmwchanmod2/mmwchanmod/data/uav_boston.p','rb')
data = pickle.load(f)
print (data[0])
print (data[1].keys())
print (data[1]['dvec'][5])
ind = np.where(data[1]['rx_type'] ==1)
print (np.min(data[1]['dvec'][ind][:,2]))
print (np.max(data[1]['dvec'][ind][:,2]))