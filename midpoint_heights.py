import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc('text', usetex = True)
from scipy import optimize



def height_plot(directory, cut_off_time, start_time, mag):


    timedata = np.genfromtxt(directory+ 'times.csv')
    midpoints = np.genfromtxt(directory+ 'midpoints.csv')
    pip = np.genfromtxt(directory+'pipette2.csv')
    midpointloc = np.genfromtxt(directory+'midpointsloc.csv')

    #print(timedata)
    positions = np.genfromtxt(directory+'widths.csv')

    stopInd = np.where(timedata[:,0]>cut_off_time)[0][0]
    startInd = np.where(timedata[:,0]>start_time)[0][0]
    r = np.polyval(pip, midpointloc[startInd:stopInd]/mag)*mag/2
    #print(r)
    R = np.asarray(positions[startInd:stopInd,0])/2
    #R = (positions[startInd:stopInd,1]-positions[startInd:stopInd,0])/2
    return timedata[startInd:stopInd,0]-timedata[startInd,0], np.asarray(midpoints[startInd:stopInd]),r, R


directorys = ['/Users/carmenlee/Desktop/13082020_pip1_1/', '/Users/carmenlee/Desktop/13082020_pip1_2/', '/Users/carmenlee/Desktop/13082020_pip1_3/', '/Users/carmenlee/Desktop/13082020_pip3/', '/Users/carmenlee/Desktop/12032019_zoom1/', '/Users/carmenlee/Desktop/12032019_zoom2/']
cut = [174.5,75.3, 55.2, 111.3, 48.1, 29.1]
start = [142,54, 45.2, 84.7, 8.20,22.48]
film = [2.55/2*1.39, 4.85*1.39/2, 1.64*1.39/2,]
mag = [1.39, 1.39, 1.39, 1.39, 3.34, 3.34]
arbshift = [30,18.17,7.01, 24.66,38.24, 5.54]
labels = [r'$\alpha = 0.766^\circ, \Omega = 3.9966$', r'$\alpha = 0.766^\circ, \Omega = 4.237$',r'$\alpha = 0.766^\circ, \Omega = 7.528$', r'$\alpha = 0.673^\circ, \Omega = 3.841$',r'$\alpha = 1.36^\circ, \Omega = 4.2102$',r'$\alpha = 1.36^\circ, \Omega = 3.5678$']
volumes = []

def theory(t, a,b, c):
    return a*(t+b)*np.log(t+b)+c
figmidpoint, axmid = plt.subplots(figsize = (4,3))
figmidpoint.subplots_adjust(top=0.99, bottom=0.28, left=0.25, right=0.94)
for k in range(len(directorys)):
    t, h,r, R = height_plot(directorys[k], cut[k], start[k], mag[k])
    #fit, cov = optimize.curve_fit(theory, t*22e-3/4.85*R*1e-6, h*1e-6)

    #print(4.85/(970*22e-3*h*10e-6)**0.5)
    #print(fit, cov)
    axmid.plot(((t-arbshift[k]))*22e-3/(R[0]*1e-6*4.85)+0.0002*1e6, (h)/R[0], '.', label = labels[k])
    #axmid.plot(((np.asarray(t)-arbshift[k])/R)/(4.85/2.2e-3)*np.log(((np.asarray(t)-arbshift[k])/R)/(4.85/2.2e-3)), (h+r)/R, '.', label = labels[k])
    #axmid.plot(t, theory(t*22e-3/4.85*R*1e-6, fit[0], fit[1], fit[2]))
#axmid.plot(np.linspace(1, 20), np.linspace(1, 20), label = r'$h \sim t$')
#xmid.plot(np.linspace(1, 20), np.linspace(1, 20)**10, label = r'$h \sim t^2$')
#axmid.plot(np.linspace(1, 20), np.linspace(1, 20)**(2/3), label = r'$h \sim t^{2/3}$')
#axmid.plot(np.linspace(1, 20), np.linspace(50, 70)*np.log(np.linspace(50, 70)), label = r'$h \sim t \textrm{ln}t$')
axmid.set_xlabel(r'$t \gamma / R_0 \eta $',fontsize = 24)
axmid.set_ylabel(r'$(h_\textrm{n}-r)/ R_0$',fontsize = 24)
axmid.set_xlim([0, 0.00025e6])
#axmid.set_xlim([0,0.00019])
axmid.set_xticks([0, 0.0001e6, 0.0002e6])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

plt.show()


figmidpoint2, axmid2 = plt.subplots(figsize = (4,3))
figmidpoint2.subplots_adjust(top=0.99, bottom=0.28, left=0.3, right=0.99)
for k in range(len(directorys)):
    t, h,r, R = height_plot(directorys[k], cut[k], start[k], mag[k])
    v = np.gradient(h)

    axmid2.loglog((t-arbshift[k])+20, v, '.', label = labels[k])
#axmid.plot(np.linspace(0, 7.5), np.log(np.linspace(0, 7.5)))
axmid2.set_xlabel(r'$h \left[\textrm{s}\right]$',fontsize = 24)
axmid2.set_ylabel(r'$v_\textrm{n} \left[\mu\textrm{m}\right]$',fontsize = 24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

plt.show()

exit()
figmidpoint, axmid = plt.subplots(figsize = (4,3))
figmidpoint.subplots_adjust(top=0.99, bottom=0.28, left=0.3, right=0.99)
for k in range(len(directorys)):
    t, h = height_plot(directorys[k], cut[k], start[k])
    axmid.semilogy(-t, h, '.', label = k)
axmid.set_xlabel(r'$t \left[\textrm{s}\right]$',fontsize = 24)
axmid.set_ylabel(r'$h_\textrm{m} \left[\mu\textrm{m}\right]$',fontsize = 24)
axmid.set_xlim([0,15])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

plt.show()
figmidpoint, axmid = plt.subplots(figsize = (4,3))
figmidpoint.subplots_adjust(top=0.99, bottom=0.28, left=0.3, right=0.99)
for k in range(len(directorys)):
    t, h = height_plot(directorys[k], cut[k], start[k])
    axmid.semilogx(-t, h, '.', label = k)
axmid.set_xlabel(r'$t \left[\textrm{s}\right]$',fontsize = 24)
axmid.set_ylabel(r'$h_\textrm{m} \left[\mu\textrm{m}\right]$',fontsize = 24)
axmid.set_xlim([0,15])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()

plt.show()
directory = '/Users/carmenlee/Desktop/12032019_zoom2/'
