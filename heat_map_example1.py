import matplotlib.pyplot as plt
from heat_map import Heat_Map
import numpy as np

f = Heat_Map(mod_name='uav_beijing', bs_type = 'Aerial',
             npts=100, nsect = 3, cdf_prob = 0.5,
             horizontal_axis_of_net = np.linspace(-100, 100, 10),
             vertical_axis_of_net = np.linspace(-100, 100,10),  plane_shift=200)
#f.net_work_area_verti = np.linspace (30, 150, 10)
#f.net_work_area_hori = np.linspace(0, 100, 10)
#f.plane_shift = 0
#f.plot_heat_map(bs_type="Aerial", tilt_angle= 45, get_link_state= True,
      #          annot= True, plane_type='xy', nsect=3, cdf_prob=0.3)
#f.plot_heat_map(bs_type="Aerial", tilt_angle= 45, get_link_state= True, annot= False, nsect =3, plane_type= 'xy')
f.plane_type = 'xy'
f.horizontal_axis_of_net=np.linspace(0, 300, 11)
f.vertical_axis_of_net = np.linspace(0, 160, 11)
f.cdf_prob = 0.5 # take meadian value
#f.get_association(aerial_height=10, tilt_angel_t=-22, tilt_angle_a=45, plane_shift=30)
f.plot_heat_map(bs_type="Aerial", tilt_angle= -12, get_link_state= True, annot= False, nsect =3, plane_type= 'xz', bs_height=10)
#f.get_association(aerial_height=10, tilt_angel_t=-22, tilt_angle_a=45)
plt.show()