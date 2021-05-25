#looking at profile curves

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

#Switch these out for the appropriate files

directory = '/Users/carmenlee/Desktop/12032019_zoom1/'
profile = np.genfromtxt(directory +'profile.csv')
horizontal = np.genfromtxt(directory +'horizontal.csv')
frame = np.genfromtxt(directory +'frames.csv')
pipette  =np.genfromtxt(directory +'pipette2.csv')
time_raw =directory+'zoom1_meta.csv'
import metadata_reader
timedata = metadata_reader.scrapy(time_raw)
print(len(profile), len(timedata))
############################################
# Homemade functions
############################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def check_for_breaks(x):
    beginning = [0]
    end = []
    for k in range(len(x)-1):
        if x[k+1]-x[k]>1:
            beginning.append(k+1)
            end.append(k)
    end.append(len(x))
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

    smooth_deriv = np.asarray(smooth_deriv)

    minima = np.where(abs(smooth_deriv)>0.2)[0]

    beginning, end = check_for_breaks(minima)
    #print(len(beginning), end)
    drop_start = []
    drop_end = []
    volume = []
    #print(beginning,end)
    if len(beginning)>3:
        for k in range(len(beginning)):
            if smooth_deriv[minima[beginning[k]]] > 0:
                drop_start.append(minima[beginning[k]])
            if smooth_deriv[minima[end[k]-1]] <0:
                drop_end.append(minima[end[k]-1])
        for k in range(len(beginning)-1):
            if smooth_deriv[minima[end[k]-1]]>0 and smooth_deriv[minima[beginning[k+1]]]<0:
                height.append(max(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)]))
                height_loc.append(np.argmax(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)])+minima[end[k]-1]+int(averaging/2))
        #volume.append(sum(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)]))
    else:
        drop_start.append(minima[0])
        drop_end.append(minima[-1])
        height.append(max(vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)]))
        height_loc.append(np.argmax(vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)])+minima[0]+int(averaging/2))
        #volume.append(sum(vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)]))
    #print(xpos, smooth_deriv, drop_start, drop_end, height, height_loc, volume)
    return(xpos, smooth_deriv, drop_start, drop_end, height, height_loc)


########Dealing with data



positions = []
heights = []
grads = []
rs = []
vol = []
drops_pos =[]
times = []

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
fig2 =plt.figure(2)
ax = fig2.add_subplot(111)

start = 121
avg_interval = 10
half_interval = 5
smooth_interval= 30
half_smo_interval = 15
for k in range(int(len(profile))-start):
    prof_smooth = []
    hzt = []
    thresh = []
    horz_t = []

    for m in range(len(profile[k+start])-avg_interval):
        p = np.polyfit(horizontal[k+start][m:m+avg_interval], profile[k+start][m:m+avg_interval], 2)
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
        ax1.plot(hzt[spot[5][z]], spot[4][z]+np.polyval(pipette, hzt[15+spot[5][z]]),'ks')
        grad.append((hzt[half_smo_interval+spot[3][z]]+np.polyval(pipette,hzt[half_smo_interval+spot[3][z]])-prof_smooth[half_smo_interval+spot[2][z]]-np.polyval(pipette, hzt[half_smo_interval+spot[2][z]]))/(hzt[15+spot[3][z]]-hzt[15+spot[2][z]]))
        h.append(spot[4][z]+np.polyval(pipette, hzt[half_smo_interval+spot[5][z]]))
        drop_pos.append((spot[0][spot[2][z]]+spot[0][spot[3][z]])/2)
        ax1.plot((spot[0][spot[2][z]]+spot[0][spot[3][z]])/2, spot[4][z],'*')

        pos.append(hzt[spot[5][z]])
        r.append(np.polyval(pipette, hzt[half_smo_interval+spot[5][z]]))
        ax1.plot(hzt, prof_smooth+np.polyval(pipette, hzt))
        timer.append(float(timedata[k+start-1,0]))
        #volumes.append(spot[6][z])
    ax1.set_xlim(0,1280)
    ax2.set_xlim(0,1280)

    plt.show()

    heights.append(h)
    rs.append(r)
    grads.append(grad)
    positions.append(pos)
    drops_pos.append(drop_pos)
    vol.append(volumes)
    times.append(timer)

'''figv = plt.figure(3)
axv = figv.add_subplot(111)
for k in range(len(vol)):
    axv.plot(positions[k], vol[k], '.')
plt.show()'''

length = len(sorted(grads,key=len, reverse=True)[0])
pos_array= np.array([xi+[0]*(length-len(xi)) for xi in positions])
grad_array =np.array([xi+[0]*(length-len(xi)) for xi in grads])
rs_array =np.array([xi+[0]*(length-len(xi)) for xi in rs])
heights_array =np.array([xi+[0]*(length-len(xi)) for xi in heights])
vol_array = np.array([xi+[0]*(length-len(xi)) for xi in vol])
time_array = np.array([xi+[0]*(length-len(xi)) for xi in times])
print(grad_array)

np.savetxt(directory +'drop_positions.csv', pos_array)
np.savetxt(directory +'drop_height.csv', heights_array)
np.savetxt(directory +'gradients.csv', grad_array)
np.savetxt(directory +'drop_piprad.csv', rs_array)
np.savetxt(directory +'times.csv', time_array)
