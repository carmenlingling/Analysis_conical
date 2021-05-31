#looking at profile curves

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

#Switch these out for the appropriate files

directory = '/Users/carmenlee/Desktop/13082020_pip1_1/'
profile = np.genfromtxt(directory +'profile.csv')
horizontal = np.genfromtxt(directory +'horizontal.csv')
frame = np.genfromtxt(directory +'frames.csv')
pipette  =np.genfromtxt(directory +'pipette2.csv')
time_raw =directory+'metadata.txt'
import metadata_reader
timedata = metadata_reader.read_txtfile(time_raw)[200:]
#    #print(timedata[k] - timedata[k+1])
############################################
# Homemade functions
############################################
def find_max(arrayx, arrayy):
    p = np.polyfit(arrayx, arrayy, 2)
    #print(p)
    return (-p[1]/(2*p[0]), p[0]*(-p[1]/(2*p[0]))**2+p[1]*-p[1]/(2*p[0])+p[2])

def find_centroid(arrayx, arrayy):
    arrayx = np.asarray(arrayx)
    arrayy = np.asarray(arrayy)
    #print(arrayx, arrayy, np.sum((arrayx-arrayx[0])*arrayy)/np.sum(arrayy)+arrayx[0])
    #exit()
    return(np.sum((arrayx-arrayx[0])*arrayy)/np.sum(arrayy)+arrayx[0])
def find_nearest(array, value):
    arraything = np.asarray(array)
    idx = (np.abs(arraything - value)).argmin()
    return idx

def check_for_breaks(x):
    beginning = [0]
    end = []
    for k in range(len(x)-1):
        if x[k+1]-x[k]>1:
            beginning.append(k+1)
            end.append(k)
    end.append(len(x)-1)
    return beginning, end

def extrema_checker(horz, vert, averaging):
    derive = np.gradient(np.asarray(vert), np.asarray(horz))
    smooth_deriv = []
    xpos = []
    for x in range(len(vert)-averaging):
        a = np.polyfit(horz[x:x+averaging], derive[x:x+averaging], 1)
        smooth_deriv.append(np.polyval(a, horz[x+int(averaging/2)]))
        xpos.append(horz[x+int(averaging/2)])

    height = []
    height_loc = []
    fitheight = []
    fit_pos = []
    centroid = []
    smooth_deriv = np.asarray(smooth_deriv)
    doublederiv = np.gradient(smooth_deriv, np.asarray(xpos))
    averaging = 30
    smooth_dblderiv = []
    xposdd = []
    for m in range(len(smooth_deriv)-averaging):
        b = np.polyfit(xpos[m:m+averaging], doublederiv[m:m+averaging], 1)
        smooth_dblderiv.append(np.polyval(b, xpos[m+int(averaging/2)]))
        xposdd.append(xpos[m+int(averaging/2)])

    minima = np.where(abs(smooth_deriv)>0.18)[0]
    #print(minima)
    beginning, end = check_for_breaks(minima)
    print(beginning, end)
    #print(len(beginning), end)
    drop_start = []
    drop_end = []
    volume = []
    #print(beginning,end)
    if len(beginning)>3:
        print('path1')
        for k in range(len(beginning)):
            if smooth_deriv[minima[beginning[k]]] > 0:
                drop_start.append(minima[beginning[k]])
            if smooth_deriv[minima[end[k]-1]] <0:
                drop_end.append(minima[end[k]-1])
        for k in range(len(beginning)-1):
            if smooth_deriv[minima[end[k]-1]]>0 and smooth_deriv[minima[beginning[k+1]]]<0:
                x, h = find_max(horz[minima[end[k]]+int(averaging/2):minima[beginning[k+1]+int(averaging/2)]], vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)])
                fitheight.append(h)
                fit_pos.append(x)
                height.append(max(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)]))
                height_loc.append(np.argmax(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)])+minima[end[k]]+int(averaging/2))
        for m in range(len(drop_start)):
            xmass = find_centroid(horz[drop_start[m]+int(averaging/2):drop_end[m]+int(averaging/2)], vert[drop_start[m]+int(averaging/2):drop_end[m]+int(averaging/2)])
            centroid.append(xmass)
    elif len(beginning)==3:
        print('path2')
        drop_start= [minima[beginning[0]], minima[end[1]]]
        drop_end=[minima[end[1]],minima[end[2]]]
        for k in range(len(drop_start)):
            x, h = find_max(horz[drop_start[k]+int(averaging/2):drop_end[k]+int(averaging/2)], vert[drop_start[k]+int(averaging/2):drop_end[k]+int(averaging/2)])
            fitheight.append(h)
            fit_pos.append(x)
            height.append(max(vert[drop_start[k]+int(averaging/2):drop_end[k]+int(averaging/2)]))
            height_loc.append(np.argmax(vert[drop_start[k]+int(averaging/2):drop_end[k]+int(averaging/2)])+drop_start[k]+int(averaging/2))
        xmass = find_centroid(horz[drop_start[0]+int(averaging/2):drop_end[1]+int(averaging/2)], vert[drop_start[0]+int(averaging/2):drop_end[1]+int(averaging/2)])
        centroid.append(xmass)
        centroid.append(0)
        #volume.append(sum(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)]))'''
    else:
        print('path3')
        #if smooth_deriv[minima[0]:minima[-1]]
        drop_start.append(minima[0])
        drop_end.append(minima[-1])
        x, h = find_max(horz[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)], vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)])
        fitheight.append(h)
        fit_pos.append(x)
        height.append(max(vert[minima[0]+int(averaging/2):minima[end[-1]]+int(averaging/2)]))
        height_loc.append(np.argmax(vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)])+minima[0]+int(averaging/2))
        for m in range(len(drop_start)):
            xmass = find_centroid(horz[drop_start[m]+int(averaging/2):drop_end[m]+int(averaging/2)], vert[drop_start[m]+int(averaging/2):drop_end[m]+int(averaging/2)])
            centroid.append(xmass)


        #volume.append(sum(vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)]))
    #print(xpos, smooth_deriv, drop_start, drop_end, height, height_loc, volume)
    '''figs,[axes1, axes2, axes3] = plt.subplots(nrows=3)
    axes1.plot(horz, vert)
    for k in range(len(beginning)):
        axes1.plot(horz[minima[beginning[k]]], vert[minima[beginning[k]]], '*')
        axes1.plot(horz[minima[end[k]]], vert[minima[end[k]]], 'o')
    #axes1.plot(horz[beginning[0]+height_loc[0]], height[0], 's')
    axes1.vlines(centroid, 0, 100)
    axes2.plot(xpos, smooth_deriv)
    axes3.plot(xpos, doublederiv)
    axes3.plot(xposdd, smooth_dblderiv)'''
    #plt.show()
    return(xpos, smooth_deriv, drop_start, drop_end, height, height_loc, fit_pos, fitheight, centroid)


########Dealing with data



positions = []
heights = []
grads = []
rs = []
vol = []
drops_positions =[]
times = []

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


start = 1
avg_interval = 30
half_interval = 15
smooth_interval= 30
half_smo_interval = 15
for k in range(268):
#for k in range(int(len(profile))-start):

    prof_smooth = []
    hzt = []
    thresh = []
    horz_t = []


    '''fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)'''

    for m in range(len(profile[k+start])-avg_interval):
        p = np.polyfit(horizontal[k+start][m:m+avg_interval], profile[k+start][m:m+avg_interval], 1)
        height = np.polyval(p,horizontal[k+start][m+half_interval])
        prof_smooth.append(height)
        hzt.append(horizontal[avg_interval][m+half_interval])
        #thresh.append(height)
        #horz_t.append(horizontal[10][m+5])





    ax1.plot(hzt, prof_smooth)


    h = []
    r = []
    grad = []
    pos = []
    drop_pos = []
    volumes = []
    timer = []
    spot=extrema_checker(hzt, prof_smooth,smooth_interval)
    ax2.plot(spot[0], spot[1])
    x = [len(spot[3]),len(spot[2]), len(spot[4]), len(spot[5])]
    #print(x)
    w = min(x)

    for z in range(w):
        ax2.plot(spot[0][spot[2][z]], spot[1][spot[2][z]], 'o')
        ax2.plot(spot[0][spot[3][z]], spot[1][spot[3][z]], 's')
        ax1.plot(hzt[half_smo_interval+spot[3][z]], prof_smooth[half_smo_interval+spot[3][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[3][z]]),'ro')
        ax1.plot(hzt[half_smo_interval+spot[2][z]], prof_smooth[half_smo_interval+spot[2][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[2][z]]),'bo')
        ax1.plot(hzt[spot[5][z]], spot[4][z]+np.polyval(pipette, hzt[half_smo_interval+spot[5][z]]),'ks')
        ax1.plot([hzt[half_smo_interval+spot[3][z]],hzt[half_smo_interval+spot[2][z]]], [prof_smooth[half_smo_interval+spot[3][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[3][z]]),prof_smooth[half_smo_interval+spot[2][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[2][z]])])
        grad.append((prof_smooth[half_smo_interval+spot[3][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[3][z]])-prof_smooth[half_smo_interval+spot[2][z]]-np.polyval(pipette, hzt[half_smo_interval+spot[2][z]]))/(hzt[half_smo_interval+spot[3][z]]-hzt[half_smo_interval+spot[2][z]]))
        h.append(spot[7][z]+np.polyval(pipette, hzt[half_smo_interval]+spot[6][z]))
        #drop_pos.append((spot[0][spot[2][z]]+spot[0][spot[3][z]])/2)
        drop_pos.append(spot[8][z])
        ax1.plot(hzt[spot[5][z]], spot[4][z],'*')
        ax1.plot(spot[6][z], spot[7][z],"o")
        pos.append(spot[6][z])
        r.append(np.polyval(pipette, hzt[half_smo_interval+spot[5][z]]))
        ax1.plot(hzt, prof_smooth+np.polyval(pipette, hzt))
        ax1.vlines(spot[8][z], 0, 100)
        timer.append(float(timedata[k+start-1,0]))
        #volumes.append(spot[6][z])
    ax1.set_xlim(0,1280)
    ax2.set_xlim(0,1280)
    #plt.title(str(k+start))
    plt.show()

    heights.append(h)
    rs.append(r)
    grads.append(grad)
    positions.append(pos)
    drops_positions.append(drop_pos)
    vol.append(volumes)
    times.append(timer)




length = len(sorted(grads,key=len, reverse=True)[0])
pos_array= np.array([xi+[0]*(length-len(xi)) for xi in positions])
grad_array =np.array([xi+[0]*(length-len(xi)) for xi in grads])
rs_array =np.array([xi+[0]*(length-len(xi)) for xi in rs])
heights_array =np.array([xi+[0]*(length-len(xi)) for xi in heights])
drop_pos_array = np.array([xi+[0]*(length-len(xi)) for xi in drops_positions])
time_array = np.array([xi+[0]*(length-len(xi)) for xi in times])
print(grad_array)
print(pos_array.shape)
figv = plt.figure(3)
axv = figv.add_subplot(111)
for k in range(pos_array.shape[1]):
    axv.plot(time_array[:,k], pos_array[:,k], '.')
figm = plt.figure(4)
axm = figm.add_subplot(111)
for k in range(pos_array.shape[1]):
    axm.plot(time_array[:,k], drop_pos_array[:,k], '.')
plt.show()

np.savetxt(directory +'drop_positions.csv', pos_array)
np.savetxt(directory +'drop_com.csv', drop_pos_array)
np.savetxt(directory +'drop_height.csv', heights_array)
np.savetxt(directory +'gradients.csv', grad_array)
np.savetxt(directory +'drop_piprad.csv', rs_array)
np.savetxt(directory +'times.csv', time_array)
