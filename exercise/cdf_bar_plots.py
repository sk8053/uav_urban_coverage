import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import gzip, pickle

f = open ('log.csv','wt', encoding='utf-8', newline="")
log_writer = csv.writer(f)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--h',action='store',default=120,type= int,\
    help='uav_height')
parser.add_argument('--isd_t',action='store',default=200,type= int,\
    help='inter site distance for terrestrial cells')

args = parser.parse_args()
uav_height = args.h
isd_t = args.isd_t


def plot_graphs(n_fig, data, label, title,
                xlabel = 'SNR (dB)', ylabel = 'CDF', x_lim = [-30,40]):
    plt.figure(n_fig)
    p  = np.arange(len(data)) / float(len(data))
    plt.plot(np.sort(data), p, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim (x_lim)
    plt.legend()
    #plt.grid()


Isd_list = [[200,0],[200,400],[200,200],[200,100]]
fig_n = 0
for uav_height in [30 ,60, 120]:
    link_ratio = np.array([])
    for ind, (ISD_t, ISD_a) in enumerate(Isd_list):
        # read data from pickle files
        dir = 'snr_data/'
        file_name = dir + 'SNR_ISD_t_' + str(ISD_t) + '_ISD_a_' + str(ISD_a) + '_height_' + str(uav_height) + '.p'
        f = gzip.open(file_name)
        data = pickle.load(f)

        data_los = data ['los']
        data_nlos = data['nlos']
        data_outage = data ['outage']

        data_all = np.append(data_los,data_nlos)
        data_all = np.append(data_all, data_outage)

        L = len(data_all)
        print (f'UAV Height = {uav_height}, ISD_t and ISD a = {ISD_t, ISD_a}'
               f', # of LOS, NLOS, and All = {len(data_los), len(data_nlos), len(data_all)}')
        link_ratio = np.append(link_ratio, np.array([len(data_los)/L , len(data_nlos)/L ,
                                                     (L-len(data_los)-len(data_nlos))/L ], dtype=float))

    #####################################################
        n_fig  = 1+fig_n
        data  = data_nlos
        title = "CDF of SNR in NLOS Case\n UAV Height = " + str(uav_height)+"m"
        file_name = "plots/CDF of SNR in NLOS Case UAV Height = " + str(uav_height) + "m"
        if ISD_a == 0:
            label = "terrestrial BS only, ISD_a= $\infty$"
        else:
            label = "terrestrial and aerial BS,  ISD_a= " + str(ISD_a)
        plot_graphs(n_fig, data, label, title, xlabel='SNR (dB)', ylabel='CDF', x_lim=[-30, 40])
        if ind == len(Isd_list)-1:
            plt.grid()
            plt.savefig(file_name)

##############################################
        n_fig = 2 + fig_n
        data = data_los
        title = "CDF of SNR in LOS Case\n UAV Height = " + str(uav_height) + "m"
        file_name = "plots/CDF of SNR in LOS Case UAV Height = " + str(uav_height) + "m"
        if ISD_a == 0:
            label = "terrestrial BS only, ISD_a= $\infty$"
        else:
            label = "terrestrial and aerial BS,  ISD_a= " + str(ISD_a)
        plot_graphs(n_fig, data, label, title,  xlabel='SNR (dB)', ylabel='CDF', x_lim=[-30, 40])
        if ind == len(Isd_list) - 1:
            plt.grid()
            plt.savefig(file_name)
################################
        n_fig  = 3+fig_n
        data  = data_all

        title = "CDF of SNR in All Cases\n UAV Height = " + str(uav_height)+"m"
        file_name = "plots/CDF of SNR in All Case UAV Height = " + str(uav_height) + "m"
        if ISD_a == 0:
            label = "terrestrial BS only, ISD_a= $\infty$"
        else:
            label = "terrestrial and aerial BS,  ISD_a= " + str(ISD_a)
        plot_graphs(n_fig, data, label, title,xlabel='SNR (dB)', ylabel='CDF', x_lim=[-30, 40])
        if ind == len(Isd_list) - 1:
            plt.grid()
            plt.savefig(file_name)

    fig_n += 4

    '''
    # plot bar graphs
    link_ratio = link_ratio.reshape(4, -1).T
    #link_ratio = link_ratio.reshape(2, -1).T
    objects = ('ISD_a = $\infty$', 'ISD_a = 400', 'ISD_a = 200', 'ISD_a=100')
    X = np.array([0, 0.6, 1.2, 1.8])

    fig = plt.figure(4+fig_n)

    plt.bar(X + 0.00, link_ratio[0], color='b', width=0.15, label='LOS')
    plt.bar(X + 0.15, link_ratio[1], color='g', width=0.15, label="NLOS")
    plt.bar(X + 0.30, link_ratio[2], color='r', width=0.15, label='outage')
    if uav_height ==30:
        plt.scatter(0, 0.001)

    plt.xticks(X + 0.15, objects)
    plt.yscale('log') #, subs = [0,-1,-2])
    plt.title('The Percentage of Link States (LOS, NLOS, outage)\n ( ISD_t = ' + str(ISD_t)
              + 'm UAV height = ' + str(uav_height) + 'm )')
    plt.legend(bbox_to_anchor=(0.0, -0.139, 1.0, -0.139), loc='lower right', ncol=3,
               borderaxespad=0.)  # mode ='expand',
    plt.grid()
    plt.savefig('plots/log_scale_bar' + '_height_' + str(uav_height) + '_isdt_' + str(ISD_t) + '_.png', bbox_inches='tight')
    '''






