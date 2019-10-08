import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from IPython.core.pylabtools import figsize
import math

G = pd.read_csv("C:\\Users\shd\Desktop\\G1.csv")
COL = G.columns.tolist()
x_y_den = G[['dis', 'speed', 'age', 'gender', 'floor_xg']]
col = x_y_den.columns.tolist()
# 选取楼层号[126, 116, 106, 91, 81, 71, 61, 51, 41, 31, 21, 11, 7]
dis_index = [0.0, 132.2000303, 250.17433706167898, 439.0170540728317, 583.5717001609739, 711.7471658000001, 834.9927564,
             965.632752, 1104.087929, 1229.727254, 1357.04364, 1483.612839789369, 1537.907707]


# part_mean四维数组，mun切割,gender为性别
# def part_mean(data, num, gender):
def part_mean(data,mun, gender, age1, age2):
    global ii
    x_label = []
    y_mean = []
    y_std = []
    data = data.sort_values(by=col[0], ascending=True)
    # data = data[data['gender'] == gender]
    data = data[data['age'] >= age1]
    data = data[data['age'] <= age2]
    data = data.reset_index(drop=True)
    x = data.iloc[0:, 0]
    y = data.iloc[0:, 1]
    col_index = []
    for i in range(1, len(dis_index)):
        a = x[(dis_index[i-1] <= x) & (dis_index[i] >= x)].index.tolist()
        x_label.append((dis_index[i-1] + dis_index[i])/2)
        if a:
            col_index.append(a[0])
        '''else:
            print(i-1)'''
    col_index.append(len(x)-1)
    print(col_index)
    print(len(col_index))
    for ii in range(1, len(col_index)):
        y_mean_i = np.mean(y[col_index[ii - 1]:col_index[ii]])
        y_std_i = np.std(y[col_index[ii - 1]:col_index[ii]])
        y_mean.append(y_mean_i)
        y_std.append(y_std_i)
    y_mean = np.insert(y_mean, 2, 'nan')
    y_std = np.insert(y_std, 2, 'nan')
    print(y_std)
    x_label_legend = "age" + "(" + str(age1) + '-' + str(age2) + ")"
    return x_label, y_mean, y_std, x_label_legend


x_label1, y_mean1, y_std1, x_label_legend1 = part_mean(x_y_den, 7, 1, 20, 60)
figsize(6, 4)
plt.errorbar(x_label1, y_mean1, yerr=y_std1, color='r', alpha=1, label='male')
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
plt.xlabel('distant(m)', font2)
plt.ylabel('speed(m/s)', font2)
# plt.title(x_label_legend1)
# plt.ylim(0.2,1.4)
plt.show()