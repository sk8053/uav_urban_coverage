import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle
import pandas as pd
class snr_plot(object):
    """
    This class plots the CDF of SNR for various deployment of UAVs and BSs
    And also plots the ration of link types (LOS, NLOS, Outage)
    """
    scenario={'isd_t':400,'isd_a':[np.inf,400, 200, 100], 'uav_heights':[30,60, 120]}
    dir = 'snr_data/'
    def __init__(self, scenario={'isd_t':400,'isd_a':[np.inf,400, 200, 100], 'uav_heights':[30,60, 120]},
                dir = 'snr_data/', enabl_log = False,isd_t = 200):

        self.scenario = scenario
        self.dir = dir
        self.ISD_t = scenario['isd_t']
        self.DATA = dict()
        self.enable_log = enabl_log
        self.figure_n = 0 # figure number
        for h in scenario['uav_heights']:
            data_all_dic, data_los, data_nlos, data_outage, link_ratio, connected_ar = \
                self.read_data(uav_height= h, ISD_t= isd_t, dir = self.dir)
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
            file_name = dir + 'DATA_ISD_t_' + str(ISD_t) + '_ISD_a_' + str(ISD_a) + '_height_' + str(uav_height) + '.txt'
            df = pd.read_csv(file_name, delimiter='\t', index_col=False)

            d_los= np.array(df[df['link_state']==1]['SNR'])
            d_nlos = np.array(df[df['link_state']==2]['SNR'])
            d_outage = np.array(df[df['link_state'] == 0]['SNR'])

            d_all = np.array(df['SNR'])
            #d_all = np.append(d_los, d_nlos)
            #d_all = np.append(d_all, d_outage)
            data_los[str(ISD_a)] = d_los
            data_nlos[str(ISD_a)] = d_nlos
            data_outage[str(ISD_a)] = d_outage

            if ISD_a !=0:
                connected_ar[str(ISD_a)] = np.sum(df['BS_type']==0)/len(df)

            data_all_dic [str(ISD_a)] = d_all

            L = len(d_all)
            link_ratio = np.append(link_ratio, np.array([len(d_los) / L, len(d_nlos) / L,
                                                         (L - len(d_los) - len(d_nlos)) / L], dtype=float))
            if self.enable_log is True:
                print(f'UAV Height = {uav_height}, ISD_t and ISD a = {ISD_t, ISD_a}'
                      f', # of LOS, NLOS, and All = {len(d_los), len(d_nlos), len(d_all)}')

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
            self.sub_graphs_plotting(ax1, data_all,  'All case')
            ax1.set_ylabel ('CDF', fontsize = 14)
            self.sub_graphs_plotting(ax2, data_los, 'LOS case')
            self.sub_graphs_plotting(ax3, data_nlos, 'NLOS case')
           # fig.suptitle ('UAV Height = '+ str(uav_height))

            plt.show()
            plt.savefig(save_dir+ 'CDF of SNR UAV Height = '+ str(uav_height))
        else:
            titles =[ 'All Case', 'LOS Case', 'NLOS Case']
            for i, case in enumerate(['all']):#, 'los', 'nlos']):
                plt.figure (self.figure_n)
                self.sub_graphs_plotting(plt, self.DATA[str(uav_height)][case],titles[i])
                self.figure_n += 1
                plt.xlabel('SNR (dB)', fontsize=14)
                plt.ylabel('CDF', fontsize=14)
                plt.legend(fontsize=12)  # loc=4
                if case == 'all':
                    plt.savefig('CDF_SNR_'+ str(uav_height)+'m.png', format = 'png', dpi = 800)


    def sub_graphs_plotting(self, ax, data,  title):
        ## This function plot sub plots for integrating all plots
        c =['b','g','k','r']
        line_type = {'0':'solid', '800':'dashed', '400':'dotted','200':'dashdot'}
        for i, case in enumerate(data.keys()):
            d = data[case]
            lt = line_type[case]
            if case == '0':
                d_0 = d[d>0]
            #d = d[d<100]

            p = np.arange(len(d)) / float(len(d))
            if case == '0':
                case = '$\infty$'
                ax.plot(np.sort(d), p, label='ISD$_d$, ' + case, color = c[i], linestyle = lt,linewidth = 2)
            else:
                ax.plot(np.sort(d), p, label='ISD$_d$, ' + case+' m', color = c[i], linestyle = lt, linewidth = 2)
            #ax.set_xticklabels(fontsize = 13)
            #ax.set_yticklabels(fontsize=13)
        ax.grid()
        if ax != plt:
            ax.set_xlabel('SNR (dB)', fontsize = 14)
        else:
            #if title == 'All Case':
            #ax.title ('UAV height, 120 m')
            ax.xticks(fontsize = 13)
            ax.yticks(fontsize = 13)
            ax.xlim (-10,60)

            ax.xlabel ('SNR (dB)', fontsize = 14)
            ax.ylabel ('CDF', fontsize =14)
            
        ax.legend() #loc=4
        ax.tight_layout()
    def plot_connected_aerial(self, save_dir =  ' '):
        # plot the percentage of connected aerial BSs
        X = np.array([0, 0.6, 1.2])
        #X = np.array([0, 0.6])
        plt.figure (self.figure_n)
        loc = 0.0
        objects = ['800', '400','200']
        colors = ['black','gray','white']
        for i,h in enumerate(self.scenario['uav_heights']):
            c = self.DATA[str(h)]['connected_ar']
            plt.bar(X+loc, list(c.values()), width=0.15,color=colors[i], label ='UAV height, '+str(h)+' m',edgecolor = 'black')
            loc += 0.15
        plt.grid()
        plt.xlabel ('ISD$_d$ (m)', fontsize = 14)
        plt.xticks(fontsize = 13)
        plt.yticks (fontsize = 13)
        plt.ylabel ('Fraction of UAVs', fontsize = 14)
        #bbox_to_anchor=(0.99, -0.089, 0.1089, 0.55)
        plt.legend( bbox_to_anchor=(-0.0, 0.45, 0.1089, 0.55),loc='upper left', ncol=1,
                    borderaxespad=0., fontsize = 13)
        #plt.title ("Percentage of Connected Aerial BSs ")
        plt.xticks (X + 0.15, objects)
        plt.tight_layout()
        plt.savefig(save_dir+"connected_aerial_BS.png")
        self.figure_n+=1

    def plot_link_ratio(self, sav_dir = ' '):
        # plot the link ratio between LOS, NLOS, and outage links
        #objects = ['ISD_a=$\infty$', 'ISD_a=400', 'ISD_a=200', 'ISD_a=100']
        objects = ['$\infty$', '800', '400', '200']
        X = np.array([0, 0.6, 1.2, 1.8])
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        fig.set_size_inches(18, 4.5 )
        uav_height_axis = zip (self.scenario['uav_heights'], [ax1,ax2,ax3])

        for uav_h, ax in uav_height_axis:
            link_ratio = self.DATA[str(uav_h)]['link_ratio']

            ax.bar(X + 0.00, link_ratio[0], color='b', width=0.15, label='LOS')
            ax.bar(X + 0.15, link_ratio[1], color='g', width=0.15, label="NLOS")
            ax.bar(X + 0.30, link_ratio[2], color='r', width=0.15, label='outage')

            plt.setp(ax, xticks = X +0.15, xticklabels = objects)
            ax.set_yscale ('log')
            #ax.set_title( 'UAV height = ' + str(uav_h) + 'm ')
            ax.set_title( str(uav_h) + ' m')
            ax.set_xlabel ('ISD$_d$ (m)')

            if ax == ax3:
                ax.legend( bbox_to_anchor=(1.136, 0.7, 0.2, 0.3),loc='upper right', ncol=1,
                        borderaxespad=0.)  # mode ='expand',
            if ax == ax1:
                ax.set_ylabel('Fraction of Link State')
            ax.grid()
            ax.tight_layout()
        #fig.suptitle('The Percentage of Link States (LOS, NLOS, outage), when ISD_t = ' + str(self.ISD_t))
        plt.savefig(sav_dir + 'log_scale_bar' + '_height_' + str(uav_h) + '_isdt_' + str(self.ISD_t) + '_.png',
                        bbox_inches='tight')
        self.figure_n +=3
    def single_plot_link_ratio(self, uav_h = 30, sav_dir = ' '):
        # plot the link ratio between LOS, NLOS, and outage links
        #objects = ['ISD_a=$\infty$', 'ISD_a=400', 'ISD_a=200', 'ISD_a=100']
        objects = ['$\infty$', '800', '400','200']
        X = np.array([0, 0.6, 1.2, 1.8])
        #X = np.array([0, 0.6,1.2])
        #plt.figure(figsize=(8, 8))
        link_ratio_30 = self.DATA[str(30)]['link_ratio']
        link_ratio_60 = self.DATA[str(60)]['link_ratio']
        link_ratio_120 = self.DATA[str(120)]['link_ratio']

        plt.bar(X + 0.00, link_ratio_30[1], color='black',edgecolor = 'black', width=0.15, label='UAV height, 30 m')
        plt.bar(X + 0.15, link_ratio_60[1], color='gray', edgecolor = 'black', width=0.15, label='UAV height, 60 m')
        plt.bar(X + 0.30, link_ratio_120[1], color='white', edgecolor = 'black', width=0.15, label='UAV height, 120 m')

        plt.xticks (X +0.15,  objects, fontsize = 14)
        plt.yticks(fontsize = 14)
        #plt.yscale ('log')
        #ax.set_title( 'UAV height = ' + str(uav_h) + 'm ')
       # plt.title( str(uav_h) + ' m')
        plt.xlabel ('ISD$_d$ (m)', fontsize = 14)
        #bbox_to_anchor=(0.99, -0.089, 0.1089, 0.55)
        plt.legend( bbox_to_anchor=(0.89, 0.45, 0.1089, 0.55),loc='upper right', ncol=1,
                    borderaxespad=0., fontsize = 10)  # mode ='expand',
        plt.ylabel('Fraction of NLOS Link', fontsize = 14)
        plt.grid()
        plt.tight_layout()
        #fig.suptitle('The Percentage of Link States (LOS, NLOS, outage), when ISD_t = ' + str(self.ISD_t))
        plt.savefig('log_scale_bar' + '_height_' + str(uav_h) + '.png',
                        bbox_inches='tight', dpi = 800)

    def plot_tr_only(self, uav_height=30):
        #titles = [ 'LOS', 'NLOS']
        c = ['r','b','k']
        for i, uav_height in enumerate([30, 60,120]):
            d= self.DATA[str(uav_height)]['nlos']['0']

            plt.plot (np.sort(d), np.linspace(0,1, len(d)), label = str(uav_height)+'m', color = c[i])
            plt.xlabel('SNR (dB)', fontsize=14)
            plt.ylabel('CDF', fontsize=14)

        plt.legend(fontsize=12)  # loc=4
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize =14)
        plt.grid()
        plt.tight_layout()
        plt.savefig('tr_only_all_case_'+str(uav_height), dpi =800)

import argparse
parser = argparse.ArgumentParser(description='')

parser.add_argument('--h',action='store',default=30,type= int,\
    help='uav_height')
parser.add_argument('--isd_t', action = 'store', default = 200, type = int, help = 'inter site distance for terrestrial BSs' )

args = parser.parse_args()
h = args.h
ISD_t = args.isd_t
s = snr_plot(scenario={'isd_t':200,'isd_a':[np.inf,800, 400,200], 'uav_heights':[30,60,120]},
                dir = 'snr_data/', enabl_log=True, isd_t = ISD_t)

#s.single_plot_link_ratio(uav_h=h)
#s.plot_connected_aerial()
#data_all, data_los, data_nlos, data_outage, link_ratio=s.read_data(uav_height=60)
'''
data  = np.loadtxt('distance_vector_snr_60.txt')
dist_vect = data[:,0:2]
distance = np.linalg.norm(dist_vect, axis = 1)

los_type = data[:,-16]

dist_los = distance[los_type==0]
dist_vlos = distance[los_type==1]
plt.figure (3)
plt.plot (np.sort(dist_los), np.linspace(0,1,len(dist_los)),'r', label = 'real los')
plt.plot (np.sort(dist_vlos), np.linspace(0,1,len(dist_vlos)),'b', label = 'virtual los')
plt.legend(fontsize = 12)
plt.grid()
plt.xlabel ('3D distance', fontsize =13)
plt.ylabel ('CDF', fontsize = 13)
plt.xticks(fontsize = 12)
plt.yticks (fontsize = 12)
plt.tight_layout()
'''
#s.plot_tr_only()
s.plot_snr(uav_height= h, single_plot=True)
plt.show()
#
