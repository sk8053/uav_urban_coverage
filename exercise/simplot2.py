import numpy as np
import matplotlib.pyplot as plt

dist_30 = np.loadtxt('distance_vector_30m_with_aerial.txt')
dist_60 = np.loadtxt('distance_vector_60m_with_aerial.txt')
dist_120 = np.loadtxt('distance_vector_120m_with_aerial.txt')

plt.plot(np.sort(dist_30), np.linspace(0,1,len(dist_30)), label = '30m')
plt.plot(np.sort(dist_60), np.linspace(0,1,len(dist_60)), label = '60m')
plt.plot(np.sort(dist_120), np.linspace(0,1,len(dist_120)), label = '120m')
plt.xlabel('3D distance',fontsize=14)
plt.ylabel('CDF', fontsize =14)
plt.grid()
plt.legend(fontsize = 13)
plt.show()
