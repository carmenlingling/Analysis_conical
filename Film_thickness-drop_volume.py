import tkinter as tk
from tkinter import filedialog 
import numpy as np
import os 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2
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

#for tracking the where the droplet is 
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

#set up reference image
ref_path = directory + '/'+fileNames[0]
ref = plt.imread(ref_path, 0)
ref_img = plt.imshow(ref)
ref = (ref-ref.min())/ref.max()
edge_sobel = sobel(ref)

level = 0.21
binary = fill_pipette(edge_sobel/edge_sobel.max(), level)

plt.imshow(binary)
plt.colorbar()
plt.show()


x,y, top, bottom = collapse(binary)
pipette = np.asarray(top)-np.asarray(bottom)
x = np.asarray(x)
plt.plot(x,pipette, '.r')

#fit a polynomial to the pipette
fit = np.polyfit(x, pipette, 6)

plt.plot(x, np.polyval(fit, x))
plt.plot(x, np.polyval(np.polyder(fit), x))


def line(a,x,b):
    return a*x+b

p,k= sc.optimize.curve_fit(line, x, y)
plt.plot(np.asarray(x), p[0]*np.asarray(x)+p[1], '-b')
plt.show()







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
    
    position = []
    if droplet_position_break == [0] and len(indices) >1:
        p = np.polyfit(x[indices[0]:indices[-1]], y[indices[0]:indices[-1]], 2)
        if -p[1]/(2*p[0]) <= 0 or -p[1]/(2*p[0]) > ref.shape[1]:
             position.append([-555])
        else:
             position.append(-p[1]/(2*p[0]))
        #print(1)
    elif len(droplet_position_break) > 1 and len(indices) >1:
        for droplet in range(len(droplet_position_break)-1):
            if droplet_position_break[droplet+1]-droplet_position_break[droplet] > 10:
                #print(indices[droplet_position_break[droplet]+1],indices[droplet_position_break[droplet+1]])
                p = np.polyfit(x[indices[droplet_position_break[droplet]+1]:indices[droplet_position_break[droplet+1]]], y[indices[droplet_position_break[droplet]+1]:indices[droplet_position_break[droplet+1]]], 2)
                if -p[1]/(2*p[0]) <= 0 or -p[1]/(2*p[0]) >1280:
                    position.append(-555)
                else:
                    position.append(-p[1]/(2*p[0]))
            else: position.append(-555)
        if len(x[indices[droplet_position_break[-1]+1]:indices[-1]]) > 0:
            p = np.polyfit(x[indices[droplet_position_break[-1]+1]:indices[-1]], y[indices[droplet_position_break[-1]+1]:indices[-1]], 2)
            if -p[1]/(2*p[0]) <= 0 or -p[1]/(2*p[0]) >1280:
                position.append(-555)
            else:
                position.append(-p[1]/(2*p[0]))
                #print(2)
        else:
            position.append(-555)
    else:
        position.append(-555)
        #print(3)
    return(position)


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('Frame #')
ax2.set_ylabel('Position from base (pixels)')

slices = np.empty(((len(fileNames)-1)*3,binary.shape[1]))
positions = []
frames = []
maxlen = 0
#print((len(fileNames)-750)*3)
for k in range(1):
    next_path = directory + '/'+fileNames[k+1]
    nextim = plt.imread(next_path, 0)
    nextsobel = sobel(nextim)
    two  = fill_pipette(nextsobel/nextsobel.max(), 0.2)
    plt.imshow(two)
    
    
    x2,y2, top2, bottom2 = collapse(two)
    line = (np.asarray(top2)+np.asarray(bottom2))/2
    for item in range(len(line)):
        slices[(k*3):(k*3)+3,item]=nextim[int(line[item])-1:int(line[item])+2, item]
    #(np.asarray(top2)-np.asarray(bottom2) - np.polyval(fit, x2)))
    
    plt.plot(np.asarray(x2), np.asarray(top2)-np.asarray(bottom2)-np.polyval(fit, np.asarray(x2)))
    film_thickness = np.average(np.asarray(top2)-np.asarray(bottom2)-np.polyval(fit,np.asarray(x2)))
    z = np.asarray(x2)
    r = np.polyval(fit, z)
    grad = np.polyval(np.polyder(fit), z)
    print(film_thickness)                   
    plt.show()
    #position = position_finder(x2, np.asarray(top2)-np.asarray(bottom2) - np.polyval(fit, x2), 20)
    #print(position, k)
    #if len(position)>0:
    #    ax2.plot(np.full(len(position), k), position, '.')
    #else:
    #    pass
    
    #positions.append(position)
    #if len(position) > maxlen:
    #    maxlen = len(position)
#ax2.set_ylim[0,1280]
'''
pos_array = y=np.array([xi+[-555]*(maxlen-len(xi)) for xi in positions])
np.savetxt(directory + '/' + 'output.csv', pos_array)

plt.savefig(directory+ '/'+'positions.eps')
plt.show()
'''
''']with open(directory+'/'+"output.csv", "w", newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(positions)
'''
'''

#plotting visual of images
fig_im = plt.figure(figsize = (3,9))
grid = plt.GridSpec(15, 1, wspace=1, hspace=0.4)

ax1 =fig_im.add_subplot(grid[0:2,0])
im1 = plt.imread(directory + '/'+fileNames[1], 0)
ax1.imshow(im1[int(im1.shape[1]/3):int(2*im1.shape[1]/3)])
a2 = fig_im.add_subplot(grid[2:13,0])
a2.imshow(slices)
a3 = fig_im.add_subplot(grid[13:15,0])
imlast = plt.imread(directory + '/'+fileNames[-1], 0)
a3.imshow(imlast[int(imlast.shape[1]/3):int(2*imlast.shape[1]/3)])



plt.savefig(directory+'/'+'droplets.eps')
#
plt.show()
        '''

