#looking at profile curves

import numpy as np
import scipy as sc

import matplotlib.pyplot as plt

#Switch these out for the appropriate files

directory = '/Users/carmenlee/Desktop/13082020_pip2_2/'
profile = np.genfromtxt(directory +'profile.csv')
horizontal = np.genfromtxt(directory +'horizontal.csv')
frame = np.genfromtxt(directory +'frames.csv')
pipette  =np.genfromtxt(directory +'pipette2.csv')
time_raw =directory+'metadata.txt'
import metadata_reader
timedata = metadata_reader.read_txtfile(time_raw)[249:]
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
    #print(x)
    beginning = [0]
    end = []
    for k in range(len(x)-1):
        if x[k+1]-x[k]>1 and k+1!=len(x)-1 and k not in beginning:

            beginning.append(k+1)
            end.append(k)
        else: pass
    end.append(len(x)-1)

    return beginning, end

def min_height(drop_start, drop_end, horz, vert, averaging):
    midpoint_height=[]
    midpoint_heightloc = []
    if len(drop_end) ==2:
        if drop_start[1]==drop_end[0]:
            min_x, min_h = find_max(horz[drop_end[0]:drop_end[0]+2*averaging],vert[drop_end[0]:drop_end[0]+2*averaging])
            midpoint_height.append(min_h)
            midpoint_heightloc.append(min_x)
        else:
            min_x, min_h = find_max(horz[drop_end[0]+averaging:drop_start[1]+averaging],vert[drop_end[0]+averaging:drop_start[1]+averaging])
            midpoint_height.append(min_h)
            midpoint_heightloc.append(min_x)
    else:
        midpoint_height.append(0)
        midpoint_heightloc.append(0)

    return(midpoint_height, midpoint_heightloc)
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
    #averaging = 30
    smooth_dblderiv = []
    xposdd = []
    for m in range(len(smooth_deriv)-averaging):
        b = np.polyfit(xpos[m:m+averaging], doublederiv[m:m+averaging], 1)
        smooth_dblderiv.append(np.polyval(b, xpos[m+int(averaging/2)]))
        xposdd.append(xpos[m+int(averaging/2)])
    smooth_dblderiv=np.asarray(smooth_dblderiv)
    #minima = np.where(abs(smooth_deriv)>0.18)[0]
    minima = np.where(np.asarray(vert)<0)[0]

    print(minima)
    beginning, end = check_for_breaks(minima)
    #print(beginning, end)
    '''figs,[axes1, axes2, axes3] = plt.subplots(nrows=3)
    axes1.plot(horz, vert)
    for k in range(len(beginning)):
        axes1.plot(horz[minima[beginning[k]]+averaging], vert[minima[beginning[k]]+averaging], '*')
        axes1.plot(horz[minima[end[k]]+averaging], vert[minima[end[k]]+averaging], 'o')
    #axes1.plot(midpoint_heightloc, midpoint_height, 's')
    #axes1.plot(horz[beginning[0]+height_loc[0]], height[0], 's')
    #axes1.vlines(centroid, 0, 100)
    axes2.plot(xpos, smooth_deriv)
    axes3.plot(xpos, doublederiv)
    axes3.plot(xposdd, smooth_dblderiv)
    plt.show()'''

    drop_start = []
    drop_end = []
    drop_startIND = []
    drop_endIND = []
    volume = []
    #@midpoint_height = []
    #midpoint_heightloc = []
    #print(beginning,end)
    if len(beginning)>3:
        print('path1')
        for k in range(len(beginning)-1):

            if smooth_deriv[minima[beginning[k]]+int(averaging/2)]>0 and xposdd[minima[beginning[k+1]]]-xposdd[minima[end[k]]]>2*averaging:
                if k ==0:
                    drop_start.append(find_max(xposdd[minima[beginning[k]]:minima[end[k]]], vert[minima[beginning[k]]+averaging:minima[end[k]]+averaging]))
                    drop_startIND.append(np.argmax(smooth_dblderiv[minima[beginning[k]]:minima[end[k]]])+minima[beginning[k]])
                else:
                    drop_startIND.append(np.argmax(smooth_dblderiv[minima[beginning[k]]:minima[end[k]]])+minima[beginning[k]])
                    drop_start.append(find_max(xposdd[minima[beginning[k]]:minima[end[k]]], vert[minima[beginning[k]]+averaging:minima[end[k]]+averaging]) )
                if k !=0:
                    if beginning[k+1] != end[k+1]:
                        drop_end.append(find_max(xposdd[minima[beginning[k+1]]:minima[end[k+1]]],vert[minima[beginning[k+1]]+averaging:minima[end[k+1]]+averaging]))
                        drop_endIND.append(np.argmax(smooth_dblderiv[minima[beginning[k+1]]:minima[end[k+1]]])+minima[beginning[k+1]])
                    else:
                        drop_end.append(find_max(xposdd[minima[beginning[k+1]-averaging:minima[beginning[k+1]]+averaging]],vert[minima[beginning[k+1]]-averaging:minima[beginning[k+1]]+averaging]))
                        drop_endIND.append(minima[beginning[k+1]])
                else:
                    if beginning[k+1] != end[k+1]:
                        drop_end.append(find_max(horz[minima[beginning[k+1]]+averaging:minima[end[k+1]]+averaging],vert[minima[beginning[k+1]]+averaging:minima[end[k+1]]+averaging]))
                        drop_endIND.append(np.argmax(smooth_dblderiv[minima[beginning[k+1]]:minima[end[k+1]]])+minima[beginning[k+1]])
                    else:
                        drop_end.append(find_max(xposdd[minima[beginning[k+1]-averaging:minima[beginning[k+1]]+averaging]],smooth_dblderiv[minima[beginning[k+1]]-averaging:minima[beginning[k+1]]+averaging]))
                        drop_endIND.append(minima[beginning[k+1]])
            elif len(drop_end)!=0 and smooth_deriv[minima[beginning[k]]+int(averaging/2)]<0 and xposdd[minima[beginning[k+1]]]-xposdd[minima[end[k]]]>2*averaging and vert[minima[end[k]]+averaging]>40:
                #drop_start.append(np.argmax(smooth_dblderiv[minima[beginning[k]]:minima[end[k]]])+minima[beginning[k]])
                drop_start.append(drop_end[-1])
                drop_startIND.append(drop_endIND[-1])
                if beginning[k+1] != end[k+1]:
                    drop_endIND.append(np.argmax(smooth_dblderiv[minima[beginning[k+1]]:minima[end[k+1]]])+minima[beginning[k+1]])
                    drop_end.append(find_max(xposdd[minima[beginning[k+1]]:minima[end[k+1]]], smooth_dblderiv[minima[beginning[k+1]]:minima[end[k+1]]]))
                else:
                    drop_end.append(xposdd[minima[beginning[k+1]]])
                    drop_endIND.append(minima[beginning[k+1]])
        #drop_end.append(np.argmax(smooth_dblderiv[minima[beginning[-1]]:minima[end[-1]]])+minima[beginning[-1]])
        #drop_start.append(np.argmax(smooth_dblderiv[minima[beginning[k]]:minima[end[k]]])+minima[beginning[k]])
        print(drop_start, drop_end)
        for k in range(len(drop_start)):
            height.append(max(vert[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)]))
            maxpos = np.argmax(vert[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)])+drop_startIND[k]+int(averaging)
            #if smooth_deriv[drop_start[k]+int(averaging/2)]>0 and smooth_deriv[drop_end[k]+int(averaging/2)]<0:# and horz[minima[end[k]]] - horz[minima[beginning[k]]]>averaging:
            x, h = find_max(horz[maxpos-averaging:maxpos+averaging], vert[maxpos-averaging:maxpos+averaging])
            fitheight.append(h)
            fit_pos.append(x)
            height_loc.append(np.argmax(vert[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)])+drop_startIND[k]+int(averaging))

        for m in range(len(drop_start)):
            #print(drop_start, drop_end)
            xmass = find_centroid(horz[drop_startIND[m]+int(averaging):drop_endIND[m]+int(averaging)], vert[drop_startIND[m]+int(averaging):drop_endIND[m]+int(averaging)])
            centroid.append(xmass)

    elif len(beginning)==3:
        print('path2')
        print(beginning, end)

        drop_startIND= [np.argmax(smooth_dblderiv[minima[beginning[0]]:minima[end[0]]])+minima[beginning[0]], np.argmax(smooth_dblderiv[minima[beginning[1]]:minima[end[1]]])+minima[beginning[1]]]
        drop_endIND=[np.argmax(smooth_dblderiv[minima[beginning[1]]:minima[end[1]]])+minima[beginning[1]], np.argmax(smooth_dblderiv[minima[beginning[2]]:minima[end[2]]])+minima[beginning[2]]]
        if find_max(xposdd[minima[beginning[1]]:minima[end[1]]],vert[minima[beginning[1]]+averaging:minima[end[1]]+averaging])[1] <40:
            drop_start= [find_max(horz[minima[beginning[0]]:minima[end[0]]],vert[minima[beginning[0]]:minima[end[0]]]), find_max(xposdd[minima[beginning[1]]:minima[end[1]]],vert[minima[beginning[1]]+averaging:minima[end[1]]+averaging])]
            drop_end=[find_max(xposdd[minima[beginning[1]]:minima[end[1]]],vert[minima[beginning[1]]+averaging:minima[end[1]]+averaging]), find_max(xposdd[minima[beginning[2]]:minima[end[2]]],vert[minima[beginning[2]]+averaging:minima[end[2]]+averaging])]
        else:
            x, loc = find_max(xposdd[minima[beginning[1]]:minima[end[1]]],smooth_dblderiv[minima[beginning[1]]:minima[end[1]]])
            print('x=',x)
            ind = np.where(horz>x)[0][0]
            poly = np.polyfit(horz[ind-10:ind+10], vert[ind-10:ind+10],1)
            h = np.polyval(poly, x)

            drop_start= [find_max(horz[minima[beginning[0]]:minima[end[0]]],vert[minima[beginning[0]]:minima[end[0]]]), [x,h]]
            drop_end=[[x,h], find_max(xposdd[minima[beginning[2]]:minima[end[2]]],vert[minima[beginning[2]]+averaging:minima[end[2]]+averaging])]

        #axes3.plot(drop_start[0], find_max(xposdd[minima[beginning[0]]:minima[end[0]]],smooth_dblderiv[minima[beginning[0]]:minima[end[0]]])[1], 'o')
        #axes3.plot(drop_end[1], find_max(xposdd[minima[beginning[2]]:minima[end[2]]],smooth_dblderiv[minima[beginning[2]]:minima[end[2]]])[1], 'o')

        for k in range(len(drop_start)):
            #print(drop_start[k], drop_end[k])
            height.append(max(vert[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)]))
            maxpos = np.argmax(vert[drop_startIND[k]+2*int(averaging):drop_endIND[k]+int(averaging)])+drop_startIND[k]+2*int(averaging)
            height_loc.append(np.argmax(vert[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)])+drop_startIND[k]+int(averaging))
            x, h = find_max(horz[maxpos-averaging:maxpos+averaging], vert[maxpos-averaging:maxpos+averaging])
            fitheight.append(h)
            fit_pos.append(x)

            xmass = find_centroid(horz[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)], vert[drop_startIND[k]+int(averaging):drop_endIND[k]+int(averaging)])
            centroid.append(xmass)
        #entroid.append(0)


        #midpoint_height.append(min_h)
        #midpoint_heightloc.append(min_x)#volume.append(sum(vert[minima[end[k]]+int(averaging/2):minima[beginning[k+1]]+int(averaging/2)]))'''
    else:
        print('path3')
        #if smooth_deriv[minima[0]:minima[-1]]
        drop_start.append(find_max(xposdd[minima[beginning[0]]:minima[end[0]]],vert[minima[beginning[0]]+averaging:minima[end[0]]+averaging]))
        drop_end.append(find_max(xposdd[minima[beginning[-1]]:minima[end[-1]]],vert[minima[beginning[-1]]+averaging:minima[end[-1]]+averaging]))
        #axes3.plot(xposdd[minima[beginning[0]]:minima[end[0]]],smooth_dblderiv[minima[beginning[0]]:minima[end[0]]],'.')
        #axes3.plot(find_max(xposdd[minima[beginning[0]]:minima[end[0]]],smooth_dblderiv[minima[beginning[0]]:minima[end[0]]])[0],find_max(xposdd[minima[beginning[0]]:minima[end[0]]],smooth_dblderiv[minima[beginning[0]]:minima[end[0]]])[1],'o')
        print(drop_start, drop_end)
        drop_startIND.append(minima[0])
        drop_endIND.append(minima[-1])
        maxpos = np.argmax(vert[drop_startIND[0]+int(averaging):drop_endIND[0]+int(averaging)])+drop_startIND[0]+int(averaging)
        x, h = find_max(horz[maxpos-averaging:maxpos+averaging], vert[maxpos-averaging:maxpos+averaging])
        fitheight.append(h)
        fit_pos.append(x)
        height.append(max(vert[minima[0]+int(averaging):minima[end[-1]]+int(averaging)]))
        height_loc.append(np.argmax(vert[minima[0]+int(averaging):minima[-1]+int(averaging)])+minima[0]+int(averaging))
        for m in range(len(drop_start)):
            xmass = find_centroid(horz[drop_startIND[m]+int(averaging):drop_endIND[m]+int(averaging)], vert[drop_startIND[m]+int(averaging):drop_endIND[m]+int(averaging)])
            centroid.append(xmass)
        #midpoint_height.append(vert[drop_end[0]+averaging])
        #midpoint_heightloc.append(horz[drop_end[0]+averaging])
    if len(drop_end)>=1:
        min_h, min_x = min_height(drop_startIND, drop_endIND, horz, vert, averaging)
        midpoint_height=min_h[0]
        midpoint_heightloc=min_x[0]
    else:
        midpoint_height=0
        midpoint_heightloc=0
        #volume.append(sum(vert[minima[0]+int(averaging/2):minima[-1]+int(averaging/2)]))
    #print(xpos, smooth_deriv, drop_start, drop_end, height, height_loc, volume)
    '''figs,[axes1, axes2, axes3] = plt.subplots(nrows=3)
    axes1.plot(horz, vert)
    for k in range(len(beginning)):
        axes1.plot(horz[minima[beginning[k]]+averaging], vert[minima[beginning[k]]+averaging], '*')
        axes1.plot(horz[minima[end[k]]+averaging], vert[minima[end[k]]+averaging], 'o')
    for m in range(len(drop_start)):
        axes1.plot(drop_start[m][0], drop_start[m][1], '^')
        axes1.plot(drop_end[m][0], drop_start[m][1], '+')
    axes1.plot(midpoint_heightloc, midpoint_height, 's')
    #axes1.plot(horz[beginning[0]+height_loc[0]], height[0], 's')
    axes1.vlines(centroid, 0, 100)
    axes2.plot(xpos, smooth_deriv)
    axes3.plot(xpos, doublederiv)
    axes3.plot(xposdd, smooth_dblderiv)
    #axes3.plot(xposdd[])
    plt.show()'''
    print(midpoint_height)
    return(xpos, smooth_deriv, drop_start, drop_end, height, height_loc, fit_pos, fitheight, centroid, midpoint_height, midpoint_heightloc)


########Dealing with data



positions = []
heights = []
grads = []
rs = []
vol = []
drops_positions =[]
times = []
midpoint_heights = []

fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


start = 1
cut_off_time = 173923
avg_interval = 30
half_interval = 15
smooth_interval= 30
half_smo_interval = 30
for k in range(200):
#for k in range(int(len(profile))-start):
    print(k)
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
    midpoint_height = []
    spot=extrema_checker(hzt, prof_smooth,smooth_interval)
    ax2.plot(spot[0], spot[1])
    x = [len(spot[3]),len(spot[2]), len(spot[4]), len(spot[5])]
    #print(x)
    w = min(x)

    for z in range(w):
        ax1.plot(spot[2][z][0], spot[2][z][1], 'o')
        ax1.plot(spot[3][z][0], spot[3][z][1], 's')
        #ax1.plot(spot[3][z], prof_smooth[half_smo_interval+spot[3][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[3][z]]),'ro')
        #ax1.plot(spot[2][z]], prof_smooth[half_smo_interval+spot[2][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[2][z]]),'bo')
        ax1.plot(hzt[spot[5][z]], spot[4][z]+np.polyval(pipette, hzt[half_smo_interval+spot[5][z]]),'ks')
        #ax1.plot([hzt[half_smo_interval+spot[3][z]],hzt[half_smo_interval+spot[2][z]]], [prof_smooth[half_smo_interval+spot[3][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[3][z]]),prof_smooth[half_smo_interval+spot[2][z]]+np.polyval(pipette, hzt[half_smo_interval+spot[2][z]])])

        grad.append((spot[3][z][1]+np.polyval(pipette, spot[3][z][0])-spot[2][z][1]-np.polyval(pipette, spot[2][z][0]))/(spot[3][z][0]-spot[2][z][0]))
        ax1.plot([spot[2][z][0],spot[3][z][0]], [spot[2][z][1]+np.polyval(pipette, spot[2][z][0]),spot[3][z][1]+np.polyval(pipette, spot[3][z][0])])
        h.append(spot[7][z])
        #drop_pos.append((spot[0][spot[2][z]]+spot[0][spot[3][z]])/2)
        drop_pos.append(spot[8][z]) #center of mass
        ax1.plot(hzt[spot[5][z]], spot[4][z],'*') #just max index, not fit position
        ax1.plot(spot[6][z], spot[7][z],"+") #fit to max position and location
        pos.append(spot[6][z])
        r.append(np.polyval(pipette, hzt[half_smo_interval+spot[5][z]]))
        ax1.plot(hzt, prof_smooth+np.polyval(pipette, hzt))
        ax1.vlines(spot[8][z], 0, 100)
        timer.append(float(timedata[k+start-1,0]))
        #volumes.append(spot[6][z])
    ax1.set_xlim(0,1280)
    ax2.set_xlim(0,1280)
    #ax1.set_ylim(0, 200)
    #plt.title(str(k+start))
    plt.show()

    heights.append(h)
    rs.append(r)
    grads.append(grad)
    positions.append(pos)
    drops_positions.append(drop_pos)
    vol.append(volumes)
    times.append(timer)
    midpoint_heights.append(spot[9])



length = len(sorted(grads,key=len, reverse=True)[0])
pos_array= np.array([xi+[0]*(length-len(xi)) for xi in positions])*1.39
grad_array =np.array([xi+[0]*(length-len(xi)) for xi in grads])/2
rs_array =np.array([xi+[0]*(length-len(xi)) for xi in rs])*1.39
heights_array =np.array([xi+[0]*(length-len(xi)) for xi in heights])*1.39/2
drop_pos_array = np.array([xi+[0]*(length-len(xi)) for xi in drops_positions])*1.39
time_array = np.array([xi+[0]*(length-len(xi)) for xi in times])/1000

'''for k in range(len(time_array[:,1])):
    if time_array[k,1] > cut_off_time:
        pos_array[k,1] = 0
        drop_pos_array[k,1] = 0
stopInd = np.where(time_array[:,0]>cut_off_time)[0][0]
print(stopInd, time_array[stopInd, 0])
figmidpoint, axmid = plt.subplots()
axmid.plot(time_array[0:stopInd,0], np.asarray(midpoint_heights[0:stopInd])/2)
plt.show()'''

figv = plt.figure(3)
axv = figv.add_subplot(111)
for k in range(pos_array.shape[1]):
    axv.plot(time_array[:,k], grad_array[:,k], '.')
figm = plt.figure(4)
axm = figm.add_subplot(111)
for k in range(pos_array.shape[1]):
    axm.plot(range(len(drop_pos_array[:,k])), drop_pos_array)
    #axm.plot(time_array[:,k], drop_pos_array[:,k], '.')
plt.show()

np.savetxt(directory +'drop_positions.csv', pos_array)
np.savetxt(directory +'drop_com.csv', drop_pos_array)
np.savetxt(directory +'drop_height.csv', heights_array)
np.savetxt(directory +'gradients.csv', grad_array)
np.savetxt(directory +'drop_piprad.csv', rs_array)
np.savetxt(directory +'times.csv', time_array)
np.savetxt(directory+'midpoints.csv', midpoint_heights)
