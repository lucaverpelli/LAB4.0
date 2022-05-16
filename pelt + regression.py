# -*- coding: utf-8 -*-
"""
Created on Fri May 13 17:02:55 2022

@author: LVerpelli
"""

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
energy[0] = np.loadtxt("C:/Users/lverpelli/Desktop/csv/back_cover.csv", delimiter=",")
energy[1] = np.loadtxt("C:/Users/lverpelli/Desktop/csv/camera.csv", delimiter=",")
energy[2] = np.loadtxt("C:/Users/lverpelli/Desktop/csv/front_cover.csv", delimiter=",")
energy[3] = np.loadtxt("C:/Users/lverpelli/Desktop/csv/manual_energy.csv", delimiter=",")
energy[4] = np.loadtxt("C:/Users/lverpelli/Desktop/csv/press.csv", delimiter=",")

#list to save the breakpoints
brk = []

model="rbf"
    
for i in range(len(energy)):
    algo = rpt.Pelt(model="rbf", min_size=1, jump=1).fit(energy[i])
    my_bkps= algo.predict(pen=7)
    brk.append(my_bkps)
    fig, (ax,) = rpt.display(energy[i], my_bkps, my_bkps,figsize=(10, 6))
    plt.show()


#non linear regression

#back_cover
y_data=energy[0]
y=y_data[brk[0][0]:brk[0][1]]
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Back Cover")
plt.ylabel("Active Power")

linear_model=np.polyfit(x,y,5)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
print("BackCover:\n", linear_model)

plt.show()

#camera
y_data=energy[1]
y=y_data[brk[1][1]:brk[1][2]]
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Camera")
plt.ylabel("Active Power")

linear_model=np.polyfit(x,y,4)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
print("Camera:\n", linear_model)

plt.show()

#front cover
y_data=energy[2]
y=y_data[brk[2][0]:brk[2][1]] #Pelt non molto preciso
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Front Cover")
plt.ylabel("Active Power")

linear_model=np.polyfit(x,y,4)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
print("Front Cover:\n", linear_model)

plt.show()

#manual energy
y_data=energy[3]
y=y_data[0:brk[3][0]] 
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Manual Energy")
plt.ylabel("Active Power")

linear_model=np.polyfit(x,y,3)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
print("Manual Energy:\n", linear_model)

plt.show()

#press
y_data=energy[4]
y=y_data[brk[4][0]:brk[4][1]] #Pelt non preciso
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Press")
plt.ylabel("Active Power")

linear_model=np.polyfit(x,y,4)
linear_model_fn=np.poly1d(linear_model)
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
print("Press:\n", linear_model)

plt.show()