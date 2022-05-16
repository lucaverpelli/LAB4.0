# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:51:39 2022

@author: LVerpelli
"""

import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import pandas as pd
#import Total Active Power for the different stations
bcover = [[],[],[]]
bcover[0] = np.loadtxt("C:/Users/lverpelli/Desktop/SMLAB/11052022_1csv.csv", delimiter=",")
bcover[1] = np.loadtxt("C:/Users/lverpelli/Desktop/SMLAB/11052022_2csv.csv", delimiter=",")
bcover[2] = np.loadtxt("C:/Users/lverpelli/Desktop/SMLAB/11052022_3csv.csv", delimiter=",")


#list to save the breakpoints
brk = []

model="rbf"
    
for i in range(len(bcover)):
    algo = rpt.Pelt(model="rbf", min_size=1, jump=1).fit(bcover[i])
    my_bkps= algo.predict(pen=7)
    brk.append(my_bkps)
    fig, (ax,) = rpt.display(bcover[i], my_bkps, my_bkps,figsize=(10, 6))
    plt.show()


#non linear regression


linear_model = pd.DataFrame(columns = ['a', 'b', 'c'])

#back_cover 0
y_data=bcover[0]
y=y_data[brk[0][0]:brk[0][1]]
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Back Cover")
plt.ylabel("Active Power")

linear_model ['a']=np.polyfit(x,y,5)
linear_model_fn=np.poly1d(linear_model['a'])
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
plt.show()

#back_cover 1
y_data=bcover[1]
y=y_data[brk[1][0]:brk[1][1]]
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Back Cover")
plt.ylabel("Active Power")

linear_model ['b']=np.polyfit(x,y,5)
linear_model_fn=np.poly1d(linear_model['b'])
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
plt.show()

#back_cover 2
y_data=bcover[2]
y=y_data[brk[2][0]:brk[2][1]]
x=np.arange(len(y))
plt.scatter(x,y,color="red")
plt.title("Back Cover")
plt.ylabel("Active Power")

linear_model ['c']=np.polyfit(x,y,5)
linear_model_fn=np.poly1d(linear_model['c'])
x_s=np.arange(0,len(y))
plt.plot(x_s,linear_model_fn(x_s),color="b")
plt.show()


print("BackCover:\n", linear_model)

mean_values = linear_model.mean(axis = 1)
std_dev_values = linear_model.std(axis = 1)

