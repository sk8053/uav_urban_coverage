import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle

class snr_plot(object):
    """
    This class plots the CDF of SNR for various deployment of UAVs and BSs
    And also plots the ration of link types (LOS, NLOS, Outage)
    """
    scenario={'isd_t':200,'isd_a':[np.inf,400, 200, 100], 'uav_heights':[30,60, 120]}
    dir = 'snr_data/'
    def __init__(self, scenario={'isd_t':200,'isd_a':[np.inf,400, 200, 100], 'uav_heights':[30,60, 120]},
                dir = 'snr_data/', enabl_log = False):

        self.scenario = scenario
        self.dir = dir
        self.ISD_t = scenario['isd_t']
        self.DATA = dict()
        self.enable_log = enabl_log
        self.figure_n = 0 # figure number
        for h in scenario['uav_heights']:
            data_all_dic, data_los, data_nlos, data_outage, link_ratio, connected_ar = \
                self.read_data(uav_height= h, ISD_t= 200, dir = self.dir)
            self.DATA[str(h)] = {'all':data_all_dic, 'los':data_los, 'nlos':data_nlos,'outage':data_outage,
                                 'link_ratio':link_ratio, 'connected_ar':connected_ar}

    def read_data(self, uav_height = 30, ISD_t= 200, dir=' '):
        """
        Read the data of SNR values and store it into a variable, DATA
        Parameters
        ----------
        uav_height: UAV height for which you want to read data
        ISD_t : Inter-site distance between terrestrial BSs
        dir: directory for reading data

        Returns
        -------
        data_all_dic, data_los, data_nlos, data_outage: SNR data classified for every case
        link_ratio: Link ratio of LOS, NLOS and outage links
        connected_ar: the percentage of connected Aerial BSs
        """
        data_los, data_nlos, data_outage = dict(), dict(), dict()
        data_all_dic = dict()
        connected_ar = dict()
        link_ratio = np.array([])
        for ISD_a in self.scenario['isd_a']:
            if ISD_a == np.inf:
                ISD_a =0
            file_name = dir + 'SNR_ISD_t_' + str(ISD_t) + '_ISD_a_' + str(ISD_a) + '_height_' + str(uav_height) + '.p'
            f = gzip.open(file_name)
            data = pickle.load(f)
            data_all = np.append(data['los'], data['nlos'])
            data_all = np.append(data_all, data['outage'])
            data_los[str(ISD_a)] = data ['los']
            data_nlos[str(ISD_a)] = data['nlos']
            data_outage[str(ISD_a)] = data ['outage']
            if ISD_a !=0:
                connected_ar[str(ISD_a)] = data['connected_ar']

            data_all_dic [str(ISD_a)] = data_all

            L = len(data_all)
            link_ratio = np.append(link_ratio, np.array([len(data['los']) / L, len(data['nlos']) / L,
                                                         (L - len(data['los']) - len(data['nlos'])) / L], dtype=float))
            if self.enable_log is True:
                print(f'UAV Height = {uav_height}, ISD_t and ISD a = {ISD_t, ISD_a}'
                      f', # of LOS, NLOS, and All = {len(data["los"]), len(data["nlos"]), len(data_all)}')

        link_ratio = link_ratio.reshape(len(self.scenario['isd_a']),-1).T

        return data_all_dic, data_los, data_nlos, data_outage, link_ratio, connected_ar

    def plot_snr(self, uav_height = 30, save_dir = ' ', single_plot = False):
        """
        Plot CDF of SNR based on the data for each case of UAV height
        Parameters
        ----------
        uav_height : UAV height
        save_dir: directory for saving data
        single_plot: determine if you want to plot individual plots or integrated ones

        Returns
        -------

        """
        data_all = self.DATA[str(uav_height)]['all']
        data_los = self.DATA[str(uav_height)]['los']
        data_nlos = self.DATA[str(uav_height)]['nlos']
        if single_plot == False:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            fig.set_size_inches(12.5,4)
            self.sub_graphs_plotting(ax1, data_all,  'All Case')
            self.sub_graphs_plotting(ax2, data_los, 'LOS Case')
            self.sub_graphs_plotting(ax3, data_nlos, 'NLOS Case')
            fig.suptitle ('UAV Height = '+ str(uav_height))
            plt.show()
            plt.savefig(save_dir+ 'CDF of SNR UAV Height = '+ str(uav_height))
        else:
            titles =[ 'All Case', 'LOS Case', 'NLOS Case']
            for i, case in enumerate(['all', 'los', 'nlos']):
                plt.figure (self.figure_n)
                self.sub_graphs_plotting(plt, self.DATA[str(uav_height)][case],titles[i])
                self.figure_n += 1
            #fig.suptitle ('UAV Height = '+ str(uav_height))
            plt.show()
            plt.savefig(save_dir+ 'CDF of SNR UAV Height = '+ str(uav_height))

    def sub_graphs_plotting(self, ax, data,  title):
        ## This function plot sub plots for integrating all plots
        for case in data.keys():
            d = data[case]
            p = np.arange(len(d)) / float(len(d))
            if case == '0':
                case = '$\infty$'
            ax.plot(np.sort(d), p, label='ISD_a = ' + case)
        ax.grid()
        if ax != plt:
            ax.set_title(title)
        else:
            ax.title (title)
        ax.legend() #loc=4

    def plot_connected_aerial(self, save_dir =  ' '):
        # plot the percentage of connected aerial BSs
        X = np.array([0, 0.6, 1.2])
        plt.figure (self.figure_n)
        loc = 0.0
        objects = ['ISD_a=400', 'ISD_a=200', 'ISD_a=100']
        colors = ['r','g','b']
        for i,h in enumerate(self.scenario['uav_heights']):
            c = self.DATA[str(h)]['connected_ar']
            plt.bar(X+loc, list(c.values()), width=0.15,color=colors[i], label =str(h)+'m')
            loc += 0.15
        plt.grid()
        plt.legend(bbox_to_anchor=(0.0, -0.129, 1.0, -0.129), loc='lower right', ncol=3,
                       borderaxespad=0.)
        plt.title ("Percentage of Connected Aerial BSs ")
        plt.xticks (X + 0.15, objects)
        plt.savefig(save_dir+"connected_aerial_BS.png")
        self.figure_n+=1

    def plot_link_ratio(self, sav_dir = ' '):
        # plot the link ratio between LOS, NLOS, and outage links
        objects = ['ISD_a=$\infty$', 'ISD_a=400', 'ISD_a=200', 'ISD_a=100']
        X = np.array([0, 0.6, 1.2, 1.8])
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        fig.set_size_inches(16, 4.5 )
        uav_height_axis = zip (self.scenario['uav_heights'], [ax1,ax2,ax3])

        for uav_h, ax in uav_height_axis:
            link_ratio = self.DATA[str(uav_h)]['link_ratio']

            ax.bar(X + 0.00, link_ratio[0], color='b', width=0.15, label='LOS')
            ax.bar(X + 0.15, link_ratio[1], color='g', width=0.15, label="NLOS")
            ax.bar(X + 0.30, link_ratio[2], color='r', width=0.15, label='outage')

            plt.setp(ax, xticks = X +0.15, xticklabels = objects)
            ax.set_yscale ('log')
            ax.set_title( 'UAV height = ' + str(uav_h) + 'm ')
            ax.legend(bbox_to_anchor=(0.0, -0.139, 1.0, -0.139), loc='lower right', ncol=3,
                       borderaxespad=0.)  # mode ='expand',
            ax.grid()
        fig.suptitle('The Percentage of Link States (LOS, NLOS, outage), when ISD_t = ' + str(self.ISD_t))
        plt.savefig(sav_dir + 'log_scale_bar' + '_height_' + str(uav_h) + '_isdt_' + str(self.ISD_t) + '_.png',
                        bbox_inches='tight')
        self.figure_n +=3
'''
import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--h',action='store',default=30,type= int,\
    help='uav_height')

args = parser.parse_args()
h = args.h
s = snr_plot(scenario={'isd_t':200,'isd_a':[np.inf,400, 200, 100], 'uav_heights':[30,60,120]},
                dir = 'snr_data/multi_sect_dir_simple/', enabl_log=True)

s.plot_link_ratio()
s.plot_connected_aerial()
#data_all, data_los, data_nlos, data_outage, link_ratio=s.read_data(uav_height=60)
s.plot_snr(uav_height= h, single_plot= False)
'''