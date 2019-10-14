import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from IPython.core.pylabtools import figsize
import math

G = pd.read_csv("G1_t0.csv")
G2 = G[G['group'] == 2]
G3 = G[G['group'] == 3]
# data=G
COL = G.columns.tolist()
font1 = {'size': 20}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 35}
'''den = list(G['den'])
x = list(G['age'])
y = list(G['speed'])'''
# data_np = G.values.T
x_y_den2 = G2[['age', 'speed', 'gender', 'floor_xg']]
x_y_den3 = G2[['dis', 'speed', 'gender', 'floor_xg']]
x_y_den = G[['dis', 'speed', 'gender', 'floor_xg']]
col = x_y_den3.columns.tolist()


# print(data)
# data是三维数组，slice_num是划分区间数，num是要选择的区间数
def den_divide(data, slice_num, num):
    data = data.sort_values(by=col[2], ascending=True)
    step1 = math.floor(len(data['den']) / slice_num)
    devide_res = data.iloc[num * step1:(num + 1) * step1, 0:2]
    # print(devide_res)
    # print(devide_res.shape())
    return devide_res

# part_mean三维数组，mun切割,gender为性别
# def part_mean(data, num, gender):
def part_mean(data, num, gender, age1, age2):
    global ii
    x_label = []
    y_mean = []
    y_std = []
    print(col)
    data = data.sort_values(by=col[0], ascending=True)
    data = data[data['gender'] == gender]
    # data = data[data['age'] >= age1]
    # data = data[data['age'] <= age2]
    data = data.reset_index(drop=True)
    x = data.iloc[0:, 0]
    y = data.iloc[0:, 1]
    step = math.floor((np.max(x) - np.min(x)) / num)
    col_index = []
    for i in range(1, num + 2):
        a = x[(np.min(x) + step * (i - 1) <= x) & (np.min(x) + step * i >= x)].index.tolist()
        print(np.min(x) + step * i)
        col_index.append(a[0])
    col_index.append(len(x))
    for ii in range(1, len(col_index) - 2):
        y_mean_i = np.mean(y[col_index[ii - 1]:col_index[ii]])
        y_std_i = np.std(y[col_index[ii - 1]:col_index[ii]])
        y_mean.append(y_mean_i)
        y_std.append(y_std_i)
        x_label.append(np.min(x) + (ii - 1) * step + step / 2)

    x_label.append(np.max(x) - step / 2)
    y_mean.append(np.mean(y[col_index[-2]:col_index[-1]]))
    y_std.append(np.std(y[col_index[-2]:col_index[-1]]))
    x_label_legend = "age" + "(" + str(age1) + '-' + str(age2) + ")"
    return x_label, y_mean, y_std, x_label_legend


# s = den_divide(x_y_den, 4, 3)
# 距离执行以下两行
x_label1, y_mean1, y_std1, x_label_legend1 = part_mean(x_y_den, 6, 1, 20, 60)
x_label2, y_mean2, y_std2, x_label_legend2 = part_mean(x_y_den, 7, 0, 30, 60)
# 年龄执行以下两行
# x_label1, y_mean1, y_std1 = part_mean(x_y_den, 11, 1)
# x_label2, y_mean2, y_std2 = part_mean(x_y_den, 5, 0)
# t.scatter(res[0], res[1])
# t.show()
figsize(30, 28)
male = G[G['gender'] == 1]
female = G[G['gender'] == 0]
plt.scatter(male['dis'], male['speed'],color='r',alpha=0.05)
plt.scatter(female['dis'], female['speed'],color='b',alpha=0.1)
plt.errorbar(x_label1, y_mean1, yerr=y_std1, color='r', alpha=1, label='male')
plt.errorbar(x_label2, y_mean2, yerr=y_std2, color='b', alpha=1, label='female')
plt.xlabel('distant(m)', font2)
plt.ylabel('speed(m/s)', font2)
# plt.title(x_label_legend1)
# ax.yaxis.set_major_locator(MultipleLocator(0.4))
plt.ylim(0, 1.8, 0.4)
plt.tick_params(labelsize=30)
plt.legend(prop=font1, loc=1)
plt.show()
