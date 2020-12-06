import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time

class PPP_deployment():
    args_default = {'x_max': 1000, 'x_min': 0, 'y_max': 1000, 'y_min': 0,
            'z_t_min': 2.0, 'z_t_max': 5.0, 'z_a_min': 30, 'z_a_max': 60,
            'uav_height': 30, 'uav_number': 100,
            'lambda_t': 4 / (np.pi * 200 ** 2), 'lambda_a': 0.0, 'min_dist': 10.0}

    def __init__(self, args = args_default):

        self.args = args
        self.xDelta = args['x_max'] -args['x_min']
        self.yDelta = args['y_max'] - args['y_min']
        self.num_uav = args['uav_number']  # number of UAVs
        self.zDelta_tr = args['z_t_max'] - args['z_t_min']
        self.zDelta_aeri = args['z_a_max'] - args['z_a_min']

        self.areaTotal = self.xDelta * self.yDelta
        self.num_terr = scipy.stats.poisson( args['lambda_t'] * self.areaTotal).rvs()  # Poisson number of points
        self.num_aeri = scipy.stats.poisson(args['lambda_a'] * self.areaTotal).rvs()  # Poisson number of points
        self.uav_indices_to_delete =[]

    def get_UAV_cooridinates(self):
        ## UAV
        #random.seed(time.time())
        xx3 = self.xDelta * scipy.stats.uniform.rvs(0, 1, ((self.num_uav, 1))) + self.args['x_min']
        yy3 = self.yDelta * scipy.stats.uniform.rvs(0, 1, ((self.num_uav, 1))) + self.args['y_min']
        zz3 = np.array([self.args['uav_height']] * self.num_uav)
        self.xx_u, self.yy_u, self.zz_u = xx3, yy3, zz3

        return self.xx_u, self.yy_u, self.zz_u

    def get_terrestril_BS(self,lambda_t):
        ## terrestrial BS
        self.num_terr = scipy.stats.poisson(lambda_t * self.areaTotal).rvs()  # Poisson number of points
        xx = self.xDelta * scipy.stats.uniform.rvs(0, 1, ((self.num_terr, 1))) + self.args['x_min']  # x coordinates of Poisson points
        yy = self.yDelta * scipy.stats.uniform.rvs(0, 1, ((self.num_terr, 1))) + self.args['y_min']  # y coordinates of Poisson points
        zz = self.zDelta_tr * scipy.stats.uniform.rvs(0, 1, ((self.num_terr, 1))) + self.args['z_t_min']  ## uniform distribution (Z_tr_min ~ Z_tr_max)

        self.xx_t, self.yy_t, self.zz_t = self.check_inter_distance(xx, yy, zz, self.args['min_dist'])
        self.xx_u, self.yy_u, self.zz_u = self.check_inter_distance_BS_UAV(
            self.xx_t, self.yy_t, self.zz_t, self.xx_u, self.yy_u, self.zz_u, self.args['min_dist']
        )
        self.num_uav = len(self.xx_u)
        return self.xx_t, self.yy_t, self.zz_t

    def get_aerial_BS(self, lambda_aeri):
        ## Aerial BS
        self.num_aeri = scipy.stats.poisson(lambda_aeri * self.areaTotal).rvs(
            random_state=int(time.time()))  # Poisson number of points
        xx2 = self.xDelta * scipy.stats.uniform.rvs(0, 1, ((self.num_aeri, 1))) + self.args['x_min']  # x coordinates of Poisson points
        yy2 = self.yDelta * scipy.stats.uniform.rvs(0, 1, ((self.num_aeri, 1))) + self.args['y_min']  # y coordinates of Poisson points
        zz2 = self.zDelta_aeri * scipy.stats.uniform.rvs(0, 1,((self.num_aeri, 1))) + self.args['z_a_min']  # uniform distribution (100 ~200)

        self.xx_a, self.yy_a, self.zz_a = self.check_inter_distance(xx2, yy2, zz2, self.args['min_dist'])
        self.xx_u, self.yy_u, self.zz_u = self.check_inter_distance_BS_UAV(
           self.xx_a, self.yy_a, self.zz_a, self.xx_u, self.yy_u, self.zz_u, self.args['min_dist'])
        self.num_uav = len(self.xx_u)
        #print(self.num_uav)
        return self.xx_a, self.yy_a, self.zz_a


    def distance(self, x, y, z, x1, y1, z1):
        return np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2)

    ## this function guarantees that the inter-distance is larger than minimum value
    def check_inter_distance(self, x, y, z, min_inter_distance):
        x_copy = []
        y_copy = []
        z_copy = []
        while (len(x) != 0):
            x_p = x[0]
            y_p = y[0]
            z_p = z[0]
            indices = [0]
            for j in range(len(x)):
                if self.distance(x_p, y_p, z_p, x[j], y[j], z[j]) < min_inter_distance:
                    indices.append(j)
            x = np.delete(x, indices)
            y = np.delete(y, indices)
            z = np.delete(z, indices)
            x_copy = np.append(x_copy, x_p)
            y_copy = np.append(y_copy, y_p)
            z_copy = np.append(z_copy, z_p)
        return x_copy, y_copy, z_copy

    def plot (self):
        # Plotting
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.scatter3D(self.xx_t, self.yy_t, self.zz_t, alpha=0.5, marker='o', edgecolor='g', facecolor='g', label='terrestrial BS')
        ax.scatter3D(self.xx_a, self.yy_a, self.zz_a, alpha=0.5, marker='o', edgecolor='b', facecolor='b', label='aerial BS')
        ax.scatter3D(self.xx_u, self.yy_u, self.zz_u, alpha=0.5, marker='o', edgecolor='r', facecolor='r', label='UAV')
        plt.legend()
        plt.title("locations")
        plt.savefig('ppp_deployment.png', bbox_inches='tight')

    def check_inter_distance_BS_UAV(self,x_b, y_b, z_b, x_u, y_u, z_u, min_inter_distance):
        vec_b = np.column_stack((x_b, y_b, z_b))
        vec_u = np.column_stack((x_u, y_u, z_u))
        d_vec = vec_b - vec_u[:,None]
        distance = np.linalg.norm(d_vec, axis=-1)
        ind = np.where (distance < min_inter_distance)[0]
        self.uav_indices_to_delete = ind
        x_u = np.delete(x_u,ind)
        y_u = np.delete(y_u, ind)
        z_u = np.delete (z_u, ind)

        return x_u, y_u, z_u

    def get_uav_num(self):
        return self.num_uav, self.uav_indices_to_delete
    def update_UAVs(self):
        return  self.xx_u, self.yy_u, self.zz_u
    def set_uav_num(self, num_uav):
        self.num_uav = num_uav