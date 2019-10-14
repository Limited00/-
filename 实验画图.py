import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from IPython.core.pylabtools import figsize
import math

G = pd.read_csv("G1_t0.csv")
font1 = {'size': 20}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 35}
G2 = G[G['group'] == 2]
G2.reset_index(drop=True)
G3 = G[G['group'] == 3]
G3.reset_index(drop=True)
COL = G.columns.tolist()

# 距离-速度
x_y_den2 = G2[['dis', 'speed', 'age', 'gender', 'floor_xg']]
x_y_den3 = G3[['dis', 'speed', 'age', 'gender', 'floor_xg']]
col = x_y_den2.columns.tolist()
# 选取楼层号[126, 116, 106, 91, 81, 71, 61, 51, 41, 31, 21, 11, 7]
dis_3 = [0.0, 132.2000303, 250.17433706167898, 439.0170540728317, 583.5717001609739, 711.7471658000001, 834.9927564,
         965.632752, 1104.087929, 1229.727254, 1357.04364, 1483.612839789369, 1537.907707]
dis_2 = [0.0, 132.2000303, 250.17433706167898, 439.0170540728317, 583.5717001609739, 711.7471658000001, 834.9927564,
         965.632752, 1104.087929, 1229.727254, 1357.04364, 1483.612839789369]


# part_mean四维数组，mun切割,gender为性别
# def part_mean(data, num, gender):
def part_mean(data, mun, gender, age1, age2, dis_index):
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
        a = x[(dis_index[i - 1] <= x) & (dis_index[i] >= x)].index.tolist()
        x_label.append((dis_index[i - 1] + dis_index[i]) / 2)
        if a:
            col_index.append(a[0])
        # else:
        # print(i-1)
    col_index.append(len(x) - 1)
    for ii in range(1, len(col_index)):
        y_mean_i = np.mean(y[col_index[ii - 1]:col_index[ii]])
        y_std_i = np.std(y[col_index[ii - 1]:col_index[ii]])
        y_mean.append(y_mean_i)
        y_std.append(y_std_i)
    y_mean = np.insert(y_mean, 2, 'nan')
    y_std = np.insert(y_std, 2, 'nan')
    x_label_legend = "age" + "(" + str(age1) + '-' + str(age2) + ")"
    return x_label, y_mean, y_std, x_label_legend


'''x_label1, y_mean1, y_std1, x_label_legend1 = part_mean(x_y_den2, 7, 1, 20, 60, dis_2)
x_label2, y_mean2, y_std2, x_label_legend2 = part_mean(x_y_den3, 7, 1, 20, 60, dis_3)
for i in range(0, len(x_label2)):
    x_label2[i] = x_label2[i] + 10
figsize(30, 28)
# figure, ax = plt.subplots(figsize=figsize)
plt.scatter(x_y_den2['dis'], x_y_den2['speed'], color='r', alpha=0.09)
plt.scatter(x_y_den3['dis'], x_y_den3['speed'], color='b', alpha=0.09)
plt.errorbar(x_label1, y_mean1, yerr=y_std1, color='r', alpha=1, label='Group2')
plt.errorbar(x_label2, y_mean2, yerr=y_std2, color='b', alpha=1, label='Group3')
plt.xlabel('distant(m)', font2)
plt.ylabel('speed(m/s)', font2)
# plt.title(x_label_legend1)
# ax.yaxis.set_major_locator(MultipleLocator(0.4))
plt.ylim(0, 1.8, 0.4)
plt.tick_params(labelsize=30)
plt.legend(prop=font1, loc=1)
plt.show()'''


# 前向时间与speed
def T_speed(group, title):
    data = group
    data = data.sort_values(by='den', ascending=True)
    data = data.reset_index(drop=True)
    x = data['den']
    y = data['speed']
    step = []
    for i in np.arange(0, 10, 0.5):
        step.append(i)
    index = []
    index_labal = []
    x_label = []
    y_mean = []
    y_std = []
    for i in range(1, len(step)):
        b = x[(step[i - 1] <= x) & (step[i] >= x)].index.tolist()
        index_labal.append((step[i - 1] + step[i]) / 2)
        index.append(b[0])
    index.append(len(x) - 1)
    for ii in range(1, len(index)):
        y_mean_i = np.mean(y[index[ii - 1]:index[ii]])
        y_std_i = np.std(y[index[ii - 1]:index[ii]])
        y_mean.append(y_mean_i)
        y_std.append(y_std_i)
    plt.scatter(x, y, color='b', alpha=0.15)  # x，y散点图
    plt.errorbar(index_labal, y_mean, yerr=y_std, color='r', alpha=1, fmt='', elinewidth=2, capsize=4)
    # plt.plot(x, y, color='r', label='连线图')  # x,y线形图
    # plt.plot(x, pp1, color='b', label='拟合图')  # 100个x及对应y值绘制的曲线
    # 可应用于各个行业的数值预估
    plt.legend(loc='best')
    # plt.scatter(x, y, color=color,label='true')
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
    plt.xlabel('front time(s)', font2)
    plt.ylabel('speed(m/s)', font2)
    plt.title(title, font2)
    # plt.legend(group_name)
    plt.ylim(0, 1.8, 0.4)
    plt.tick_params(labelsize=30)
    plt.legend(prop=font1, loc=1)
    plt.show()


g2_del = G2[G2['den'] < 10]
g3_del = G3[G3['den'] < 10]


# T_speed(g2_del, 'speed-density of Group2')
# T_speed(g3_del, 'speed-density of Group3')

# 年龄——速度
def age_speed(name, title):
    data = name
    data = data.sort_values(by='age', ascending=True)
    data = data.reset_index(drop=True)
    x = data['age']
    y = data['speed']
    step = []
    for j in np.arange(20, 60, 5):
        step.append(j)
    print(step)
    index = []
    index_age = []
    y_mean = []
    y_std = []
    for j in range(1, len(step)):
        b = x[(step[j - 1] <= x) & (step[j] >= x)].index.tolist()
        index_age.append((step[j - 1] + step[j]) / 2)
        index.append(b[0])
    index.append(len(x) - 1)
    for ii in range(1, len(index)):
        y_mean_i = np.mean(y[index[ii - 1]:index[ii]])
        y_std_i = np.std(y[index[ii - 1]:index[ii]])
        y_mean.append(y_mean_i)
        y_std.append(y_std_i)
    plt.scatter(x, y, color='b', alpha=0.03)  # x，y散点图
    plt.errorbar(index_age, y_mean, yerr=y_std, color='r', alpha=1, fmt='', elinewidth=2, capsize=4)
    plt.legend(loc='best')
    # plt.scatter(x, y, color=color,label='true')
    plt.xlabel('age', font2)
    plt.ylabel('speed(m/s)', font2)
    plt.title(title, font2)
    # plt.legend(group_name)
    plt.ylim(0, 1.8, 0.4)
    plt.tick_params(labelsize=30)
    plt.legend(prop=font1, loc=1)
    plt.show()


# age_speed(G, '')

