# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:53:32 2022

@author: Provvisorio
"""
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

#import Total Active Power for the different stations
energy = [[],[],[],[],[]]
energy[0] = np.loadtxt("C:/Users/Provvisorio/OneDrive - Politecnico di Milano/Desktop/SMLAB/press.csv", delimiter=",")
energy[1] = np.loadtxt("C:/Users/Provvisorio/OneDrive - Politecnico di Milano/Desktop/SMLAB/manual_energy.csv", delimiter=",")
energy[2] = np.loadtxt("C:/Users/Provvisorio/OneDrive - Politecnico di Milano/Desktop/SMLAB/front_cover.csv", delimiter=",")
energy[3] = np.loadtxt("C:/Users/Provvisorio/OneDrive - Politecnico di Milano/Desktop/SMLAB/camera.csv", delimiter=",")
energy[4] = np.loadtxt("C:/Users/Provvisorio/OneDrive - Politecnico di Milano/Desktop/SMLAB/back_cover.csv", delimiter=",")

#list to save the breakpoints
brk = []

model="rbf"
    
for i in range(len(energy)):
    algo = rpt.Pelt(model="rbf", min_size=3, jump=5).fit(energy[i])
    my_bkps= algo.predict(pen=5)
    brk.append(my_bkps)
    fig, (ax,) = rpt.display(energy[i], my_bkps, my_bkps,figsize=(10, 6))
    plt.show()






