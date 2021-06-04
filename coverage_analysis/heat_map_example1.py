import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from coverage_analysis.heat_map import Heat_Map
import numpy as np
import seaborn as h_map

f = Heat_Map(mod_name='uav_boston', bs_type = 'Terrestrial',
             npts=100, nsect = 3,
             horizontal_axis_of_net = np.linspace(-200, 200, 31), #41
             vertical_axis_of_net = np.linspace(0, 150,31),  plane_shift=0) #31
#f.net_work_area_verti = np.linspace (30, 150, 10)
#f.net_work_area_hori = np.linspace(0, 100, 10)
#f.plane_shift = 0
#f.plot_heat_map(bs_type="Aerial", tilt_angle= 45, get_link_state= True,
      #          annot= True, plane_type='xy', nsect=3, cdf_prob=0.3)
#f.plot_heat_map(bs_type="Aerial", tilt_angle= 45, get_link_state= True, annot= False, nsect =3, plane_type= 'xy')
f.plane_type = 'xz'
f.npts = 100

rx_matrix2= f.plot_heat_map(bs_type="Terrestrial", tilt_angle= -12, get_link_state=False, annot=False, nsect =3,
        plane_type= 'xz', bs_height=0)

np.savetxt('std_aoa_theta.txt', rx_matrix2)
plt.show()