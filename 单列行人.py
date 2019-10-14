import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from IPython.core.pylabtools import figsize
import math

font1 = {'size': 20}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 35}
v = []
t = []
dis = []
t1 = []
for d in np.arange(0.44, 3.5, 0.1):
    dis.append(d)
    if d <= 1.1:
        vi = 1.16 * math.tanh(1.2*(d - 0.85) + 0.5)
    elif 3 > d > 1.1:
        vi = 1.212*(0.53*d - 0.58) - 0.47*d + 1.41
    else:
        vi = 1.212
    v.append(vi)
    t.append(d/vi)
for i in range(0, len(dis)-1):
    ti = (dis[i+1] - dis[i])/(v[i+1] - v[i])
    t1.append(ti)
print(i)
print(dis)
print(v)
print(1.2*(0.53*3 - 0.58) - 0.47*3 + 1.41)
print(1.16 * math.tanh(1.2*(1.1 - 0.85) + 0.5))
print(len(dis))
print(len(t1))
plt.plot(dis, v)
plt.xlabel('d(m)', font2)
plt.ylabel('speed(m/s)', font2)
# plt.title(x_label_legend1)
# ax.yaxis.set_major_locator(MultipleLocator(0.4))
plt.ylim(0, 1.8, 0.4)
plt.tick_params(labelsize=30)
plt.show()
plt.plot(t, v)
plt.show()
v.pop(-1)
plt.plot(t1, v)
plt.show()
