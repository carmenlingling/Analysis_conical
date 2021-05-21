#looking at profile curves

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

################

profile = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/profile.csv')
horizontal = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/horizontal.csv')
frame = np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/frames.csv')
pipette  =np.genfromtxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/pipette2.csv')


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
    drop_start = []
    drop_end = []
    drop_start_2 =[]
    drop_end_2 = []
    #print(beginning,end)
    if len(beginning)>3:
        for k in range(len(beginning)):
            if smooth_deriv[minima[beginning[k]]] > 0:
                drop_start.append(minima[beginning[k]])
            if smooth_deriv[minima[end[k]-1]] <0:
                drop_end.append(minima[end[k]-1])
        for k in range(len(beginning)-1):
            if smooth_deriv[minima[end[k]-1]]>0 and smooth_deriv[minima[beginning[k+1]]]<0:
                height.append(max(vert[minima[end[k]]+15:minima[beginning[k+1]]+15]))
                height_loc.append(np.argmax(vert[minima[end[k]]+15:minima[beginning[k+1]]+15])+minima[end[k]-1]+15)
    else:
        drop_start.append(minima[0])
        drop_end.append(minima[-1])
        height.append(max(vert[minima[0]+15:minima[-1]+15]))
        height_loc.append(np.argmax(vert[minima[0]+15:minima[-1]+15])+minima[0]+15)
    '''for k in range(len(beginning)):
        if smooth_deriv[minima[beginning[k]]] > 0:
            drop_start.append(minima[beginning[k]])
        if smooth_deriv[minima[end[k]-1]] <0:
            drop_end.append(minima[end[k]-1])
    for k in range(len(beginning)-1):
        if smooth_deriv[minima[end[k]-1]]>0 and smooth_deriv[minima[beginning[k+1]]]<0:
            height.append(max(vert[minima[end[k]]+15:minima[beginning[k+1]]+15]))
            height_loc.append(np.argmax(vert[minima[end[k]]+15:minima[beginning[k+1]]+15])+minima[end[k]-1]+15)'''
        #height.append(max(vert[minima[beginni
    #print(len(drop_start), len(drop_end),len(height), len(height_loc))
    #minima = argrelextrema(np.asarray(smooth_deriv), np.less)
    '''if len(np.ndarray.tolist(maxima[0]))>0:
        maxima = min(np.ndarray.tolist(maxima[0]))
    else:
        maxima = 0
    if len(np.ndarray.tolist(minima[0]))>0:
        minima = max(np.ndarray.tolist(minima[0]))
    else:
        minima = len(xpos)
    #print(maxima, minima)
    if maxima ==0 and minima == 0:
        pass
    else:
        width = xpos[minima-1]-xpos[maxima]
        diff = vert[minima+int(averaging/2)]-vert[maxima+int(averaging/2)]
        #print('Width = ', width)
        #print('Diff = ', diff )
    return(xpos, smooth_deriv, maxima, minima)'''
    return(xpos, smooth_deriv, drop_start, drop_end, height, height_loc)

def height_and_pipette(horz, vert, pipette):
    height = max(vert)
    height_loc = horz[np.argmax(vert)]
    r = np.polyval(pipette, height_loc)
    qmax1 = find_nearest(vert[0:np.argmax(vert)], height/4)
    qmax2 = find_nearest(vert[np.argmax(vert):-1], height/4)
    full_width = horz[qmax2+np.argmax(vert)]-horz[qmax1]
    h1 = vert[qmax1]+np.polyval(pipette, horz[qmax1])
    h2 = vert[qmax2+np.argmax(vert)] + np.polyval(pipette, horz[qmax2+np.argmax(vert)])
    print(height)
    return(height, height_loc, r, h1, h2, full_width, qmax1, qmax2+np.argmax(vert))


smoothed_profiles = []
horizontals = []
thresholded = []
horz_thresh = []
positions = []
heights = []
grads = []
rs = []
threshold = 0

drops_pos =[]
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
fig2 =plt.figure(2)
ax = fig2.add_subplot(111)
for k in range(int(len(profile))-1):
    prof_smooth = []
    hzt = []
    thresh = []
    horz_t = []
    
    #frame.append(k*100)
    for m in range(len(profile[k])-10):
        p = np.polyfit(horizontal[k][m:m+10], profile[k][m:m+10], 2)
        height = np.polyval(p,horizontal[k][m+5])
        prof_smooth.append(height)
        hzt.append(horizontal[10][m+5])
        thresh.append(height)
        horz_t.append(horizontal[10][m+5])
        '''if height > threshold:
            thresh.append(height)
            horz_t.append(horizontal[50][m+25])'''
    
    '''fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)'''
    beginning, end = check_for_breaks(horz_t)
    from scipy.signal import argrelextrema
    #print(beginning, end)
    ax1.plot(horz_t, thresh)
    
    
    h = []
    r = []
    grad = []
    pos = []
    drop_pos = []
    for m in range(len(end)):
        if end[m]-beginning[m]>3:
            spot=(extrema_checker(horz_t[beginning[m]:end[m]], thresh[beginning[m]:end[m]],30))
            ax2.plot(spot[0], spot[1])
            x = [len(spot[3]),len(spot[2]), len(spot[4]), len(spot[5])]
            w = min(x)
            
            for z in range(w):
                ax2.plot(spot[0][spot[2][z]], spot[1][spot[2][z]], 'o')
                ax2.plot(spot[0][spot[3][z]], spot[1][spot[3][z]], 's')
                ax1.plot(horz_t[beginning[m]+5+spot[3][z]], thresh[beginning[m]+5+spot[3][z]]+np.polyval(pipette, horz_t[beginning[m]+5+spot[3][z]]),'ro')
                ax1.plot(horz_t[beginning[m]+5+spot[2][z]], thresh[beginning[m]+5+spot[2][z]]+np.polyval(pipette, horz_t[beginning[m]+15+spot[2][z]]),'bo')
                ax1.plot(horz_t[spot[5][z]], spot[4][z]+np.polyval(pipette, horz_t[beginning[m]+15+spot[5][z]]),'ks')
                grad.append((thresh[beginning[m]+5+spot[3][z]]+np.polyval(pipette,horz_t[beginning[m]+5+spot[3][z]])-thresh[beginning[m]+5+spot[2][z]]-np.polyval(pipette, horz_t[beginning[m]+5+spot[2][z]]))/(horz_t[beginning[m]+5+spot[3][z]]-horz_t[beginning[m]+5+spot[2][z]]))
                h.append(spot[4][z]+np.polyval(pipette, horz_t[beginning[m]+5+spot[5][z]]))
                drop_pos.append((spot[0][spot[2][z]]+spot[0][spot[3][z]])/2)
                ax1.plot((spot[0][spot[2][z]]+spot[0][spot[3][z]])/2, spot[4][z],'*')
                #print(drop_pos)
                pos.append(horz_t[spot[5][z]])
                r.append(np.polyval(pipette, horz_t[beginning[m]+5+spot[5][z]]))
            ax1.plot(horz_t[beginning[m]:end[m]], thresh[beginning[m]:end[m]]+np.polyval(pipette, horz_t[beginning[m]:end[m]]))
            
            '''if spot[2] == 0 and spot[3] ==0:
                grad.append(0)
            else:
                #ax2.plot(spot[0][spot[2]], spot[1][spot[2]], 'o')
                #ax1.plot(horz_t[beginning[m]+15+spot[2]], thresh[beginning[m]+15+spot[2]]+np.polyval(pipette, horz_t[beginning[m]+15+spot[2]]), 'ro')
        
        #print(spot[0][spot[3][0]],horz_t[beginning[m]+15+spot[3][0]])
                #ax2.plot(spot[0][spot[3]-1], spot[1][spot[3]-1], 'o')
                #ax1.plot(horz_t[beginning[m]+15+spot[3]], thresh[beginning[m]+15+spot[3]]+np.polyval(pipette, horz_t[beginning[m]+15+spot[3]]), 'ko')
                grad.append((thresh[beginning[m]+15+spot[3]]+np.polyval(pipette,horz_t[beginning[m]+15+spot[3]])-thresh[beginning[m]+15+spot[2]]-np.polyval(pipette, horz_t[beginning[m]+15+spot[2]]))/(horz_t[beginning[m]+15+spot[3]]-horz_t[beginning[m]+15+spot[2]]))'''
            #height, height_loc, rad, h1, h2, full_width, qmax1, qmax2 = height_and_pipette(horz_t[beginning[m]:end[m]], thresh[beginning[m]:end[m]],pipette)
       # ax1.plot(info[1], info[0]+info[2], 's')
            #h.append(height+rad)
            #r.append(rad)
            #pos.append(height_loc)
            #ax1.plot(horz_t[beginning[m]+qmax1], h1, 's')
            #ax1.plot(horz_t[beginning[m]+qmax2], h2, 's')
            
            #grad.append((h2-h1)/(horz_t[beginning[m]+qmax2]-horz_t[beginning[m]+qmax1]))
            
        else:
            pass
    #print(grad)
    
    '''ax.plot(np.full(len(h), k), h, 'k.', label = 'height')
    ax.plot(np.full(len(r), k), r, 'r.', label = 'pipette radius')
    ax.plot(np.full(len(grad), k), grad, 'b.', label = 'gradient')'''
    #plt.legend(loc=0)
    #ax2.plot(horz_t,, '.')
    ax1.set_xlim(0,1280)
    ax2.set_xlim(0,1280)
    #plt.show()
    #plt.show()
    heights.append(h)
    rs.append(r)
    grads.append(grad)
    positions.append(pos)
    drops_pos.append(drop_pos)
    
    thresholded.append(thresh)
    horz_thresh.append(horz_t)
    smoothed_profiles.append(prof_smooth)
    horizontals.append(hzt)

'''

figv = plt.figure(3)
axv = figv.add_subplot(111)
for k in range(len(grads)):
    axv.plot(positions[k], np.asarray(heights[k])*np.asarray(grads[k])/np.asarray(rs[k]), '.')'''

#print(grads)
length = len(sorted(grads,key=len, reverse=True)[0])
pos_array= np.array([xi+[0]*(length-len(xi)) for xi in positions])
grad_array =np.array([xi+[0]*(length-len(xi)) for xi in grads])
rs_array =np.array([xi+[0]*(length-len(xi)) for xi in rs])
heights_array =np.array([xi+[0]*(length-len(xi)) for xi in heights])
drop_array = np.array([xi+[0]*(length-len(xi)) for xi in drops_pos])
print(grad_array)

np.savetxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/drop_positions_raw.csv', pos_array)
np.savetxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/drop_height_raw.csv', heights_array)
np.savetxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/gradients_raw.csv', grad_array)
np.savetxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/drop_piprad_raw.csv', rs_array)
np.savetxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/frame_raw.csv', frame)
np.savetxt('/Users/sflee/Desktop/Research/Plateau-Rayleigh project/Data OM/PDMS5000/12032019/5000_01s_12032019_zoom_1/zoom/drops_raw.csv', drop_array)
'''
fig = plt.figure(1)
ax = fig.add_subplot(111)
for k in range(int(len(frame))):
    ax.plot(horizontal[k], profile[k])
    ax.plot(horizontals[k], smoothed_profiles[k])
plt.show()
'''



