import scipy.constants
from mmwchanmod.datasets.download import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML
import tensorflow.keras.backend as K
fig = plt.figure()
ax = plt.axes(projection='3d')

class Channel_Info(object):
    def __init__(self):
        self.frequency = 28e9
        self.lambda_ = scipy.constants.speed_of_light / self.frequency
        self.NF = 6
        self.KT = -174
        self.TX_power = 23
        self.BW = 400e6
        K.clear_session
        self.channel = load_model('uav_lon_tok', src='remote')
        self.channel.load_link_model()
        self.channel.load_path_model()

class Network(object):
    def __init__(self, X_MAX = 100, X_MIN = -100, Y_MAX = 100, Y_MIN = -100,Z_MAX=60, Z_MIN = 0):
        self.coordinate = {'X_MAX':X_MAX,
                           'X_MIN':X_MIN,
                           'Y_MAX':Y_MAX,
                           'Y_MIN':Y_MIN,
                            'Z_MAX':Z_MAX,
                           'Z_MIN':Z_MIN}

    def drop(self, UAVs = [], BSs = []):
        for obj in BSs:
            obj.set_locations(self.coordinate)
        for key in UAVs.keys():
            UAVs[key].set_locations(self.coordinate)
            UAVs[key].set_BS_info(BSs)
        self.Cell_types = UAVs['0'].cell_types
        self.N_UAV = len(UAVs)

        self.BS_locations = UAVs['0'].bs_locations

        return UAVs, BSs

    def randomWalk_3D(self,i):
        # visualize the random walk for all UAVs
        UAV_locations = self.data_t[str(i)]['UAV_locations']
        serving_BS = self.data_t[str(i)]['serving_BS']
        ax.cla()  # clear the previous plot
        # set boundaries
        ax.set_xlim(self.coordinate['X_MIN'], self.coordinate['X_MAX'])
        ax.set_ylim(self.coordinate['Y_MIN'], self.coordinate['Y_MAX'])
        ax.set_zlim(self.coordinate['Z_MIN'], self.coordinate['Z_MAX'])
        line, = ax.plot([], [], lw=2)

        for ind in range(len(self.BS_locations)):
            if self.Cell_types[ind] == 1:
                tr = ax.scatter3D(self.BS_locations[ind, 0], self.BS_locations[ind, 1], self.BS_locations[ind, 2], marker='o',
                                  edgecolor='k', facecolor='k')
            else:
                ar = ax.scatter3D(self.BS_locations[ind, 0], self.BS_locations[ind, 1], self.BS_locations[ind, 2], marker='H',
                                  edgecolor='b', facecolor='b')

        for u in range(self.N_UAV):
            uav = ax.scatter3D(UAV_locations[u, 0], UAV_locations[u, 1], UAV_locations[u, 2], marker='D', edgecolor='r',
                               facecolor='r')
            ax.text(UAV_locations[u, 0], UAV_locations[u, 1], UAV_locations[u, 2], '%s' % (str(u)), size=10, zorder=1,
                    color='k')
            b_ind = serving_BS[u]
            ax.plot([UAV_locations[u][0], self.BS_locations[b_ind][0]], [UAV_locations[u][1], self.BS_locations[b_ind][1]],
                    [UAV_locations[u][2], self.BS_locations[b_ind][2]], 'b-.')

        ax.legend((tr, ar, uav), ('terrestrial BS', 'aerial BS', 'UAV'))
        return line

    def visualization(self, run_time = 300, data_t=None):
        self.data_t = data_t
        anim = animation.FuncAnimation(fig, self.randomWalk_3D, frames=run_time, interval=100)
        #HTML(anim.to_jshtml())

        #anim.save('vis.mp4', writer=writer)
        #anim.save('vis.mp4')

        #writervideo = animation.FFMpegFileWriter(fps=60)
        #anim.save('random_walk.mp4', writer=writervideo)
        #anim.save('random_walk.mp4')
    def plot_SINR(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=[20, 10])
        fig.text(0.5, 0.07, 'Time', ha='center', fontsize=16)
        fig.text(0.09, 0.5, 'SINR', va='center', rotation='vertical', fontsize=16)
        for n in range(self.N_UAV):
            ax = axes[n // 5, n % 5]
            ax.plot(self.UAV_set[str(n)].SINRs)
            ax.set_title('UAV ' + str(n))
            ax.grid()