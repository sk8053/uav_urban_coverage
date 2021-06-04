import  numpy as np
import sys
sys.path.append('/home/sk8053/uavchanmod/mmwchanmod/sim/')
class drone_antenna_gain():
    def __init__(self):
        p = np.loadtxt('/home/sk8053/uavchanmod/mmwchanmod/sim/azi_ele_angles.txt')
        az, ele = p.T
        v = np.loadtxt('/home/sk8053/uavchanmod/mmwchanmod/sim/values.txt')
        az, ele = np.array(az.T, dtype = int), np.array(ele.T, dtype = int)
        self.gain = dict()
        for i, (a, e) in enumerate(zip(az, ele)):
            self.gain[(str(a), str(e))] =v[i]

