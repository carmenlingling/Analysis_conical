import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
from scipy import ndimage as ndi
from skimage import feature, measure
from skimage.filters import roberts, sobel, scharr, prewitt
import scipy as sc
import csv
from itertools import zip_longest

#selects files for analyzing
root = tk.Tk()
root.withdraw()
root.update()
file_path = filedialog.askopenfilename() #asks which file you want to analyze and records the filepath and name
root.destroy()

#List all of the image files
directory = os.path.split(file_path)[0]
fileNames = []
for files in [f for f in os.listdir(directory) if f.endswith('.tif')]:
    fileNames.append(files)
fileNames.sort()

flip = False
###############################Function definition section###########################
def crop(img):
    """
    Crop the image to select the region of interest
    """
    x_min = 0
    x_max = 1280
    y_min = 0
    y_max = 1048
    return img[y_min:y_max,x_min:x_max]

#A function that creates a mask of the edges of the pipette
def fill_pipette(edges, threshold):
    filled = np.empty((edges.shape[0],edges.shape[1]))
    for k in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[k,j]< threshold:
                filled[k,j] = 0
            else:
                filled[k,j] = 1
    return filled

#A function that checks where the pipette + droplets are, makes an array of the appropriate size
def refill_array(top, bottom, x, y, nextim):
    new_array = np.empty((nextim.shape[0],len(x)+1))
    print(new_array.shape, nextim.shape)
    for value in x[0:-1]:
        for ys in range(nextim.shape[1]):
            if ys < y[value] and ys < bottom[value]:
                new_array[value, (ys - bottom[value]+int(y[value]))] = nextim[value, ys]
            elif ys > y[value] and ys > top[value]:
                new_array[value, (ys - top[value] + int(y[value]))] = nextim[value, ys]
            else:
                pass

    return new_array[0:x, min(bottom):max(top)]

#for tracking the where valid points are
def collapse(masked):
    tops = []
    bottoms = []
    validx = []
    validy = []
    for column in range(masked.shape[1]):
        indices = np.where(masked[:,column] == 1)
        index = indices[0]

        if len(index) > 0:
            bottoms.append(min(index))
            tops.append(max(index))
        else:
            bottoms.append(5000)
            tops.append(5000)
    top = []
    bottom = []
    for m in range(masked.shape[1]):
        if tops[m] != 5000 and bottoms[m] != 5000:
             validx.append(m)
             validy.append((tops[m]+bottoms[m])/2)
             top.append(tops[m])
             bottom.append(bottoms[m])

    return validx, validy, top, bottom


def position_finder(x, y, threshold):
    indices = []
    for position in range(len(x)):
        if y[position] > threshold:
            indices.append(position)
        else:
            pass
    if len(indices) == 0:
        return [0]
    else:
        pass
    #print(indices)

    droplet_position_break = [0]

    for index in range(len(indices)-1):

        if indices[index+1] - indices[index] > 1:
            droplet_position_break.append(index)
    #print(droplet_position_break)
    heights = []
    widths = []
    position = []
    if droplet_position_break == [0] and len(indices) >1:
        p = np.polyfit(x[indices[0]:indices[-1]], y[indices[0]:indices[-1]], 2)
        if -p[1]/(2*p[0]) <= 0 or -p[1]/(2*p[0]) > ref.shape[1]:
             position.append(-555)
             widths.append(-555)
             heights.append(-555)
        else:
             position.append(-p[1]/(2*p[0]))
             heights.append(np.polyval(p, -p[1]/(2*p[0])))
             widths.append(-x[indices[0]]+x[indices[-1]])
        #print(1)
    elif len(droplet_position_break) > 1 and len(indices) >1:
        for droplet in range(len(droplet_position_break)-1):
            if droplet_position_break[droplet+1]-droplet_position_break[droplet] > 10:
                #print(indices[droplet_position_break[droplet]+1],indices[droplet_position_break[droplet+1]])
                p = np.polyfit(x[indices[droplet_position_break[droplet]+1]:indices[droplet_position_break[droplet+1]]], y[indices[droplet_position_break[droplet]+1]:indices[droplet_position_break[droplet+1]]], 2)
                if -p[1]/(2*p[0]) <= 0 or -p[1]/(2*p[0]) >1280:
                    position.append(-555)
                    widths.append(-555)
                    heights.append(-555)
                else:
                    position.append(-p[1]/(2*p[0]))
                    heights.append(np.polyval(p, -p[1]/(2*p[0])))
                    widths.append(-x[indices[droplet_position_break[droplet]+1]]+x[indices[droplet_position_break[droplet+1]]])
            else:
                position.append(-555)
                widths.append(-555)
                heights.append(-555)
        if len(x[indices[droplet_position_break[-1]+1]:indices[-1]]) > 0:
            p = np.polyfit(x[indices[droplet_position_break[-1]+1]:indices[-1]], y[indices[droplet_position_break[-1]+1]:indices[-1]], 2)
            if -p[1]/(2*p[0]) <= 0 or -p[1]/(2*p[0]) >1280:
                position.append(-555)
                widths.append(-555)
                heights.append(-555)
            else:
                position.append(-p[1]/(2*p[0]))
                heights.append(np.polyval(p, -p[1]/(2*p[0])))
                widths.append(-x[indices[droplet_position_break[-1]+1]]+x[indices[-1]])
                #print(2)
        else:
            position.append(-555)
            widths.append(-555)
            heights.append(-555)

    else:
        position.append(-555)
        widths.append(-555)
        heights.append(-555)
        #print(3)
    return(position, widths, heights)


############################## Reference image of bare pipette####################
#here you find the pipette radius and the gradient of the radius as a function of the horizontal position, z



#set up reference image
ref_path = directory + '/'+fileNames[0]
if flip ==True:
    ref = np.fliplr(crop(plt.imread(ref_path, 0)))
else:
    ref = crop((plt.imread(ref_path, 0)))
ref_img = plt.imshow(ref)
ref = (ref-ref.min())/ref.max()
edge_sobel = sobel(ref)

level = 0.25
binary = fill_pipette(edge_sobel/edge_sobel.max(), level)

plt.imshow(binary, alpha = 0.6)
plt.colorbar()
plt.show()


x,y, top, bottom = collapse(binary)
pipette = np.asarray(top)-np.asarray(bottom)

x = np.asarray(x)
plt.plot(x,pipette, '.r')

x1 = x[387:561]
x2 = x[1159:]
pip1 = pipette[387:561]
pip2 = pipette[1159:]
print(np.concatenate((x1, x2)))
#fit a polynomial to the pipette
fit = np.polyfit(np.concatenate((x1, x2)), np.concatenate((pip1, pip2)), 4)
#fit = np.polyfit(x, pipette,6)
plt.plot(x, np.polyval(fit, x))



def line(a,x,b):
    return a*x+b

p,k= sc.optimize.curve_fit(line, x, y)
plt.plot(np.asarray(x), p[0]*np.asarray(x)+p[1], '-b')
plt.show()

z = np.asarray(x) ###horizontal position
r = np.polyval(fit, z) #### radius
grad = np.polyval(np.polyder(fit), z) ###gradient of radius

#############################################################


################### Film thickness from the second image #############
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('Frame #')
ax2.set_ylabel('z (pixels)')

import pims
def ImportStack(directory):
    frames = pims.TiffStack(directory)
    return frames

def ImportSequence(directory):
    frames = pims.ImageSequence(os.path.join(directory,'*.tif'))
    return frames

#path_two = directory + '/' + fileNames[1]
#frames = ImportStack(path_two)

path_two = directory+'/'
frames = ImportSequence(path_two)
nextim = crop(frames[1])
if flip == True:

    nextsobel = sobel(np.fliplr(nextim))
    plt.imshow(nextsobel)
    plt.show()
else:
    nextsobel = sobel(nextim)
two  = fill_pipette(nextsobel/nextsobel.max(), 0.25)
plt.imshow(two)


x2,y2, top2, bottom2 = collapse(two)
line = (np.asarray(top2)+np.asarray(bottom2))/2

plt.plot(np.asarray(x2), np.asarray(top2)-np.asarray(bottom2)-np.polyval(fit, np.asarray(x2)))
film_thickness = np.average(np.asarray(top2)-np.asarray(bottom2)-np.polyval(fit,np.asarray(x2)))

print(film_thickness)
plt.show()

########################################################################

############################Finding droplet positions###################


slices = np.empty(((len(frames)-1)*3,binary.shape[1]))
positions = []
times = []
position = []
profile = []
#print((len(fileNames)-750)*3)
for k in range(len(frames)-1):
    if flip ==True:
        nextim = np.fliplr(crop(frames[k]))
    else:
        nextim = crop((frames[k+1]))
    #plt.imshow(nextim)
    #plt.show()
    nextsobel = sobel(nextim)
    two  = fill_pipette(nextsobel/nextsobel.max(), 0.25)
    #plt.imshow(two)


    x2,y2, top2, bottom2 = collapse(two)
    line = (np.asarray(top2)+np.asarray(bottom2))/2
    for item in range(len(line)):
        slices[(k*3):(k*3)+3,item]=nextim[int(line[item])-1:int(line[item])+2, item]
    #plt.plot(np.asarray(x2), np.asarray(top2)-np.asarray(bottom2)-np.polyval(fit, np.asarray(x2)))

    profile.append(np.ndarray.tolist(np.asarray(top2)-np.asarray(bottom2)-np.polyval(fit, np.asarray(x2))))
    position.append(x2)
    times.append(k)
print(position)
length = len(sorted(profile,key=len, reverse=True)[0])
print(length)
pos_array = y=np.array([xi+[-555]*(length-len(xi)) for xi in position])
profile_array = np.array([hi+[-555]*(length-len(hi)) for hi in profile])
#frames_array = np.array([hi+[-555]*(length-len(hi)) for hi in frames])
#print(pos_array)

np.savetxt(directory + '/' + 'profile.csv', profile_array)
np.savetxt(directory + '/' + 'horizontal.csv', pos_array)
np.savetxt(directory + '/' + 'frames.csv', times)
np.savetxt(directory + '/' + 'pipette2.csv', fit)

#plt.savefig(directory+ '/'+'positions.eps')
#plt.show()

ra = np.r_[np.linspace(0,0.9, int(len(frames)/50)), np.linspace(0, 0.9, int(len(frames)/50))]
c = plt.get_cmap("plasma")
colors = c(ra)
plt.plot(pos_array,profile_array)
plt.show()

np.savetxt(directory+'/'+'slices.csv', slices)
from mpl_toolkits.axes_grid1 import make_axes_locatable
################################################
#plotting visual of images and saving data#########
fig_im, ax2 = plt.subplots(figsize=(4, 7))
fig_im.subplots_adjust(top=0.96, bottom=0.2, left=0.21, right=0.96)
slices = np.genfromtxt(directory+'/slices.csv')
timedata = np.genfromtxt(directory+ '/times.csv')
print(len(timedata), len(slices[0:1113]),timedata[-1]-timedata[0], len(slices[0:1113])*1000/(timedata[-1,0]-timedata[0,0]), (timedata[-1,0]-timedata[0,0])/(1000*len(slices)))
ax2.imshow(slices[0:1113])
divider = make_axes_locatable(ax2)
# below height and pad are in inches
ax1 = divider.append_axes("top", 1.1, pad=0.1, sharex=ax2)
ax3 = divider.append_axes("bottom", 1.1, pad=0.1, sharex=ax2)

# make some labels invisible
ax1.xaxis.set_tick_params(labelbottom=False)
ax1.yaxis.set_tick_params(labelleft=False)
ax2.xaxis.set_tick_params(labelbottom=False)
ax3.yaxis.set_tick_params(labelleft=False)
ax2.set_yticks([0,345, 690, 1035])
ax2.set_yticklabels([r'$0$', r'$20$', r'$40$', r'$60$'], fontsize = 20)
ax2.set_ylabel(r'$t$ $\left[ \textrm{s}\right]$', fontsize = 24)
ax3.set_xticks([0, 500/1.39, 1000/1.39, 1500/1.39])

ax3.set_xticklabels([r'$0$', r'$500$', r'$1000$',  r'$1500$'], fontsize = 20)
ax3.set_xlabel(r'$x$ $\left[ \mu \textrm{m}\right]$', fontsize = 24)
#fig_im = plt.figure(figsize = (3,9))
#grid = plt.GridSpec(15, 1, wspace=1, hspace=0.2)

#ax1 =fig_im.add_subplot(grid[0:2])
#im1 = plt.imread(directory + '/'+fileNames[2], 0)
im1 = frames[1]
nextim = crop((im1))
#plt.imshow(nextim)
#plt.show()
nextsobel = sobel(nextim)
two  = fill_pipette(nextsobel/nextsobel.max(), 0.25)
x2,y2, top2, bottom2 = collapse(two)
line = (np.asarray(top2)+np.asarray(bottom2))/2
#ax1.xticks([0, 500, 1000],['','',''])
ax1.imshow(im1[int(line[0])-int(im1.shape[1]/6):int(line[0])+int(im1.shape[1]/6)])
#a2 = fig_im.add_subplot(grid[2:13])

#a2.xticks([0, 500, 1000], ['','',''])
#a3 = fig_im.add_subplot(grid[13:15,0])
#imlast = plt.imread(directory + '/'+fileNames[-1], 0)
imlast = frames[-1]
print(len(fileNames))
nextim = crop((imlast))
#plt.imshow(nextim)
#plt.show()
nextsobel = sobel(nextim)
two  = fill_pipette(nextsobel/nextsobel.max(), 0.25)
x2,y2, top2, bottom2 = collapse(two)
line = (np.asarray(top2)+np.asarray(bottom2))/2
#a3.set_xticks([0, 500, 1000],['0', str(500*1.39), str(1000*1.39)])
ax3.imshow(imlast[int(line[0])-int(im1.shape[1]/6):int(line[0])+int(im1.shape[1]/6)])




plt.savefig(directory+'/'+'droplets.eps')
plt.show()
