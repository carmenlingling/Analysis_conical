import tkinter as tk
from tkinter import filedialog 
import numpy as np
import os 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2

#selects files for analyzing
root = tk.Tk()
root.withdraw()
root.update()
file_path = filedialog.askopenfilename() #asks which file you want to analyze and records the filepath and name
root.destroy()

#figure

for i in range(149):
    B=cv2.imread(file_path,i)
    path = os.path.split(file_path)[0] + '/%i' % (i)
    fig = plt.imshow(B)
    plt.savefig(path)
    plt.clf()


