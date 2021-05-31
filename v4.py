#This code calculates the velocity from pipette data and the measured velocity


#required modules###################################

import numpy as np
import scipy as sc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex = True)
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


########################Functions to deal with selecting data from data calculated from Analysis.py#############################################################


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []


    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def SelectData(data,times, drops, heights, gradients, rs, pos2):


    print("\nSelect points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")
    position = []
    height = []
    grad = []
    radii = []
    position2 = []
    for k in range(drops):

        if __name__ == '__main__':

            fig, ax = plt.subplots()
            #xs = np.repeat(np.arange(data.shape[0]),data.shape[1])
            spots = ax.scatter(times, data)

            selector = SelectFromCollection(ax, spots)
            ax.set_ylim([0, 1300])

            plt.show()

            selector.disconnect()

            # After figure is closed print the coordinates of the selected points

            datam = selector.xys[selector.ind]
            new_height = heights.flat[selector.ind]
            new_grad = gradients.flat[selector.ind]
            new_pos = (datam[:,0], datam[:,1])
            new_rad = rs.flat[selector.ind]
            new_pos2 = pos2.flat[selector.ind]
            position.append(new_pos)
            height.append(new_height)
            grad.append(new_grad)
            radii.append(new_rad)
            position2.append(new_pos2)
    #plt.show()

    '''fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    for m in range(len(position)):
        ax2.plot(position[m][0], height[m])

    plt.show()'''


    return position, height, grad, radii, position2


##############################################################################
#defining velocity calcuation functions

def velocitycalc(times,positions,heights, rad, grad, average_number):
    newtimes = []
    newposition = []
    newvelocity = []
    newheights = []
    newrad = []
    newgrad = []
    for k in range(len(positions)-average_number-1):
        p = np.polyfit(times[k:k+average_number], positions[k:k+average_number], 1)
        a = np.polyfit(times[k:k+average_number], heights[k:k+average_number], 1)
        b = np.polyfit(times[k:k+average_number], rad[k:k+average_number], 1)
        c = np.polyfit(times[k:k+average_number], grad[k:k+average_number], 1)
        time = (times[k]+times[k+average_number])/2
        newtimes.append(time)
        #newposition.append(np.polyval(p, time))
        newposition.append(np.polyval(p, time))
        newvelocity.append(np.polyval(np.polyder(p), time))
        newheights.append(np.polyval(a, time))
        newrad.append(np.polyval(b, time))
        newgrad.append(np.polyval(c, time))

    return(newtimes, newposition, newvelocity, newheights, newrad, newgrad)

def velocitycalc_noheight(times,positions, average_number):
    newtimes = []
    newposition = []
    newvelocity = []
    newheights = []
    for k in range(len(positions)-average_number-1):
        p = np.polyfit(times[k:k+average_number], positions[k:k+average_number], 1)

        time = (times[k]+times[k+average_number])/2
        newtimes.append(time)
        #newposition.append(np.polyval(p, time))
        newposition.append(((time)*p[0]+(p[1])))
        newvelocity.append(2*p[0])


    return(newtimes, newposition, newvelocity)

def velocitycalcsmooth(times,positions, heights):
    newtimes = times



    p = np.polyfit(times, (positions), 2)

    newposition = np.polyval(p, times)
    a = np.polyfit(times, heights, 2)
    newvelocity=(np.polyval(np.polyder(p), newposition))
    newheight = np.polyval(a,times)

    return(newtimes, newposition, newvelocity, newheight )

'''data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/500_500ms_22012018_1/position2.csv')
data_height = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/500_500ms_22012018_1/heights2.csv')
data_pipette = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/500_500ms_22012018_1/pipette2.csv')'''
'''data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/1000_500ms_23012018_2_pipette/position2.csv')
data_height = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/1000_500ms_23012018_2_pipette/heights2.csv')
data_pipette = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/1000_500ms_23012018_2_pipette/pipette2.csv')'''
#data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/1500_500ms_23012019_1/output.csv')
directory = '/Users/carmenlee/Desktop/13082020_pip1_1/'
data = np.genfromtxt(directory+'drop_positions.csv')
data_height = np.genfromtxt(directory+'drop_height.csv')
data_pipette = np.genfromtxt(directory+'pipette2.csv')
gradients = np.genfromtxt(directory+'gradients.csv')
rs = np.genfromtxt(directory+'drop_piprad.csv')
timedata = np.genfromtxt(directory+ 'times.csv')
print(timedata)
#data2 = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/01022019/zoom/drops.csv')
'''
data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/01022019/zoom_single/drop_positions.csv')
data_height = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/01022019/zoom_single/drop_height.csv')
data_pipette = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/28022018/1500_1s_01022019_zoom_single_2/pipette2.csv')
gradients = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/01022019/zoom_single/gradients.csv')
rs = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/01022019/zoom_single/drop_piprad.csv')
data2 = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/01022019/zoom_single/drop_positions.csv')
print(data.shape)
'''
#data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/1500_500ms_23012019_1/output.csv')

#data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/2000_500ms_22012018_3_pipette/position.csv')
#data_height = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/2000_500ms_22012018_3_pipette/heights.csv')

#data_pipette = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/22012019/2000_500ms_22012018_3_pipette/pipette.csv')

plt.plot(timedata, data, '.')
#plt.plot(range(len(data2)), data2)
plt.plot(range(len(data)), np.polyval(data_pipette, data))
plt.show()



drop_num = int(input('How many droplets do you see?'))

position, height, grad, radii, position_2 = SelectData(data,timedata, drop_num, data_height,gradients, rs, data)
print(position_2[0])
fig = plt.figure(1)
ax = fig.add_subplot(111)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)


'''fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)'''

ra = np.r_[np.linspace(0,0.9, len(position)), np.linspace(0, 0.9, len(position))]
c = plt.get_cmap("viridis")
colors = c(ra)
prefactor = [6,6,6]
for m in range(drop_num):
    time, positions, velocity, heights, rad, grads = velocitycalc(position[m][0], position[m][1], height[m], radii[m], grad[m], 10)
    #time2, position2, velocity2, heights2, rad2, grads2 = velocitycalc(position[m][0], position[m][1], height[m], radii[m], grad[m], 5)
    #time, positions, velocity, heights = velocitycalcsmooth(position[m][0], position[m][1], height[m])

    #ax.plot(time, position[m][1], color = colors[m])
    #ax.plot(time, position_2[m])
    ax.plot(time, positions, '.', color = colors[m])
    print(data_pipette)
    r = np.polyval(data_pipette, positions)
    grade = np.polyval(np.polyder(data_pipette), positions)
    #ax2.plot(positions, r)
    #ax2.plot(positions, rad)
    ax2.plot(positions, grade)
    ax2.plot(positions, grads)
    #ax3.plot(positions, velocity,color = colors[m], label = str(m))
    #ax3.plot(positions, heights, color = colors[m], label = str(m))
    ax3.plot(position[m][1], height[m] , '.', color = colors[m], label = str(m))
    if m ==2:

        #velocity.append(-0.71)
        #time.append(time[-1]+1)
        ax4.plot(np.asarray(time)/1000, np.asarray(velocity)*1.39, '.', color = colors[m],  label = r'\textrm{Data}')
    #ax4.plot(np.asarray(positions)*3.69,7.5*(0.02033/0.005)*(np.asarray(heights)*3.69)*grads/rad , color = colors[m])
    #ax4.plot(positions,1.75*(0.02033/0.005)*(np.asarray(heights))*grade/r)
        #ax5.plot(np.asarray(time)/0.5, np.asarray(velocity)*3.69/0.5, '.', color = colors[m], label = r'\textrm{Data}')
        ax4.plot(np.asarray(time)/1000,prefactor[m]*(0.02033)*(np.asarray(heights)*grads)/rad, color = 'k', label = r'\textrm{Model}')
        #ax5.plot(time2, velocity2)

    else:
        ax4.plot(np.asarray(time)/1000, np.asarray(velocity)*1.39, '.', color = colors[m])
    #ax4.plot(np.asarray(positions)*3.69,7.5*(0.02033/0.005)*(np.asarray(heights)*3.69)*grads/rad , color = colors[m])
    #ax4.plot(positions,1.75*(0.02033/0.005)*(np.asarray(heights))*grade/r)
        #ax5.plot(np.asarray(time)/0.5, np.asarray(velocity)*3.69/0.5, '.', color = colors[m])
        #ax5.plot(time2, velocity2)
        ax4.plot(np.asarray(time)/1000,prefactor[m]*(0.02033)*(np.asarray(heights)*grads)/rad, color = 'k')
    #ax5.plot(np.asarray(time2)/0.5,prefactor[m]*(0.02033/0.005)*(np.asarray(heights2))*grads2/rad2, color = colors[m])

#ax5.set_xlabel(r'$\textrm{Time} /\textrm{s}$', fontsize = 24)
ax.set_ylabel('Position from pipette tip')
ax3.set_ylabel(r'$h$', fontsize = 24,)
ax4.set_xlabel(r'$\textrm{Time} /\textrm{s}$ ', fontsize = 24)
ax4.set_ylabel(r'$\textrm{Velocity}$ / $\mu \textrm{m s}^{-1}$', fontsize = 24)
#ax4.set_ylim(3, 14)
ax4.legend(loc = 2, fontsize = 24, frameon = False)
#ax5.legend(loc = 2, fontsize = 24, frameon = False)
#ax5.set_ylim(3, 14)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.show()
