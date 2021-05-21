#velocity calculation

#currently no smoothing, purely discrete




import numpy as np
import scipy as sc
import matplotlib.pyplot as plt




data = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/Glycerol/23102018_fiber2_singledrop_gylcerol/single_1/output.csv', delimiter = ',', usecols = 0)


'''p = np.polyfit(range(len(data)), data, 5)
spline_positions = np.polyval(p, range(len(data)))'''
plt.plot(range(len(data)), data, '.')
#plt.plot(range(len(data)), spline_positions)


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
def SelectData(plot):


    print("\nSelect points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")
    position = []
    frames = []
    velocity = []

    for k in range(3):

        if __name__ == '__main__':

            fig, ax = plt.subplots()
            #xs = np.repeat(np.arange(data.shape[0]),data.shape[1])
            ax.imshow(np.flipud(plot), origin = 'lower')
            spots = ax.scatter(1280, 1024)

            selector = SelectFromCollection(ax, spots)
            ax.set_ylim([0, 1024])
            
            plt.show()

            selector.disconnect()

            # After figure is closed print the coordinates of the selected points
        
            datam = selector.xys[selector.ind]
            print(selector.ind)
            
            new_pos = (datam[:,0], datam[:,1])
        
            frames.append(datam[5:-5:,0])
            position.append(new_pos)
            
    #plt.show()

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        for m in range(len(frames)):
            ax2.plot(frames[m], position[m])

        plt.show()


ps = []
newpos = []
for k in range(len(data)-10):
    p = np.polyfit(np.arange(k, k+11), data[k:k+11], 2)
    ps.append(p)
    newpos.append(np.polyval(p, k+5))


plt.plot(np.asarray(range(len(data)-10))+5, np.asarray(newpos))
plt.xlabel('frame')
plt.ylabel('position')
plt.show()

velocity = []
spl = []

for spot in range(len(newpos)-1):
    spl.append((newpos[spot+1] - newpos[spot])/0.75)
                

plt.plot(np.asarray(range(len(spl)))+5, spl, '.')
plt.xlabel('frame')
plt.ylabel('velocity')
plt.show()

