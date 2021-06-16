import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('text', usetex = True)




def height_plot(directory, cut_off_time, start_time):


    timedata = np.genfromtxt(directory+ 'times.csv')
    midpoints = np.genfromtxt(directory+ 'midpoints.csv')
    print(timedata)


    stopInd = np.where(timedata[:,0]>cut_off_time)[0][0]
    startInd = np.where(timedata[:,0]>start_time)[0][0]
    return timedata[startInd:stopInd,0]-timedata[stopInd,0], np.asarray(midpoints[startInd:stopInd])


directorys = ['/Users/carmenlee/Desktop/13082020_pip1_1/', '/Users/carmenlee/Desktop/13082020_pip1_2/', '/Users/carmenlee/Desktop/13082020_pip1_3/', '/Users/carmenlee/Desktop/13082020_pip3/', '/Users/carmenlee/Desktop/12032019_zoom1/', '/Users/carmenlee/Desktop/12032019_zoom2/']
cut = [175,74, 55, 111.6, 48, 29.05]
start = [142,54, 45.2, 84.7, 8.20,22.48]
film = [2.55/2*1.39, 4.85*1.39/2, 1.64*1.39/2,]
volumes = []
figmidpoint, axmid = plt.subplots(figsize = (4,3))
figmidpoint.subplots_adjust(top=0.99, bottom=0.28, left=0.3, right=0.99)
for k in range(len(directorys)):
    t, h = height_plot(directorys[k], cut[k], start[k])
    axmid.plot(-t, h, '.', label = k)
axmid.set_xlabel(r'$t \left[\textrm{s}\right]$',fontsize = 24)
axmid.set_ylabel(r'$h \left[\mu\textrm{m}\right]$',fontsize = 24)
axmid.set_xlim([0,15])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

plt.show()
directory = '/Users/carmenlee/Desktop/12032019_zoom2/'
