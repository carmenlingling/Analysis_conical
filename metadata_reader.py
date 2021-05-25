#metadata reader&viewer, hopefully

import numpy as np

def reformat(temp):
    loc_start = temp.find('=')+3
    loc_end = temp[loc_start:].find('\"')
    #print(temp[loc_start:loc_end+loc_start])
    return(temp[loc_start:loc_end+loc_start])
##If metadata is scraped from image j
def scrapy(filename):
    data = np.genfromtxt(filename, dtype = 'str', skip_header = 1, usecols = [1,6] )
    for m in range(len(data)):
        data[m,0] = float(reformat(data[m,0]))
        data[m,1] = int(reformat(data[m,1]))
    return data
